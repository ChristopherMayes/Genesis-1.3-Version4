// Apple Silicon (Metal) GPU backend -- resident storage and host transfers.
//
// This file is compiled only when the project is configured with
// -DENABLE_METAL=ON, which also defines G4_METAL. See include/MetalEngine.h for
// why the data has to stay resident rather than being marshalled per step.
//
// Scope of this stage: device discovery, allocation of the resident FP32
// buffers and validated transfers. The kernels come next.

#include "MetalEngine.h"

#include "Beam.h"
#include "Field.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <sstream>

// Buffers are shared storage: on Apple Silicon the GPU and the CPU address the
// same physical pages, so there is no staging copy and no explicit
// synchronisation beyond command-buffer completion.
static const MTLResourceOptions kShared = MTLResourceStorageModeShared;

struct MetalEngine::Impl {
    id<MTLDevice> dev {nil};
    id<MTLCommandQueue> queue {nil};

    int nslice {0};       // time slices, all resident
    int npart {0};        // particles per slice, must be uniform
    double gref {0};      // reference energy for the stored gamma offsets

    // Beam, structure of arrays. gamma is stored as an offset from gref.
    id<MTLBuffer> bX, bY, bPX, bPY, bG, bT;
    id<MTLBuffer> bCurrent;

    // One field buffer per harmonic, each nslice * ngrid^2 complex<float>,
    // stored in the same slice order as Field::field so that the ring-buffer
    // index Field::first keeps its meaning.
    std::vector<id<MTLBuffer> > bField;
    std::vector<int> ngrid;

    size_t bytes {0};

    float *fx() const { return (float *)[bX contents]; }
    float *fy() const { return (float *)[bY contents]; }
    float *fpx() const { return (float *)[bPX contents]; }
    float *fpy() const { return (float *)[bPY contents]; }
    float *fg() const { return (float *)[bG contents]; }
    float *ft() const { return (float *)[bT contents]; }
};

MetalEngine::MetalEngine() : p_(new Impl) {}

MetalEngine::~MetalEngine()
{
    delete p_;
}

bool MetalEngine::available()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return (dev != nil) && [dev hasUnifiedMemory];
    }
}

std::string MetalEngine::deviceName()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            return std::string("none");
        }
        return std::string([[dev name] UTF8String]);
    }
}

double MetalEngine::gammaRef() const { return p_->gref; }
size_t MetalEngine::bytesResident() const { return p_->bytes; }

bool MetalEngine::init(Beam *beam, std::vector<Field *> *field, std::string &reason)
{
    reason.clear();

    p_->dev = MTLCreateSystemDefaultDevice();
    if (p_->dev == nil) {
        reason = "no Metal device";
        return false;
    }
    if (![p_->dev hasUnifiedMemory]) {
        // A discrete GPU would need staging copies, which is exactly the cost
        // this design exists to avoid.
        reason = "Metal device does not have unified memory";
        return false;
    }
    p_->queue = [p_->dev newCommandQueue];

    p_->nslice = static_cast<int>(beam->beam.size());
    if (p_->nslice < 1) {
        reason = "no beam slices";
        return false;
    }

    // The SoA layout needs a rectangular particle array. one4one runs let the
    // population vary from slice to slice, so they are not supported yet.
    p_->npart = static_cast<int>(beam->beam[0].size());
    for (int is = 0; is < p_->nslice; is++) {
        if (static_cast<int>(beam->beam[is].size()) != p_->npart) {
            std::ostringstream os;
            os << "particle count varies between slices (" << p_->npart << " vs "
               << beam->beam[is].size() << " in slice " << is
               << ") -- one4one is not supported";
            reason = os.str();
            return false;
        }
    }
    if (p_->npart < 1) {
        reason = "no particles";
        return false;
    }

    for (size_t i = 0; i < field->size(); i++) {
        Field *f = field->at(i);
        if (static_cast<int>(f->field.size()) != p_->nslice) {
            std::ostringstream os;
            os << "field harmonic " << f->getHarm() << " has " << f->field.size()
               << " slices but the beam has " << p_->nslice;
            reason = os.str();
            return false;
        }
        // The threadgroup-tiled source deposition works on 32 x 32 tiles.
        if ((f->ngrid % 32) != 0) {
            std::ostringstream os;
            os << "ngrid = " << f->ngrid << " is not a multiple of 32";
            reason = os.str();
            return false;
        }
    }

    // Reference energy for the gamma offsets: the mean over the whole beam.
    double sum = 0;
    for (int is = 0; is < p_->nslice; is++) {
        for (int ip = 0; ip < p_->npart; ip++) {
            sum += beam->beam[is][ip].gamma;
        }
    }
    p_->gref = sum / (static_cast<double>(p_->nslice) * p_->npart);

    const size_t np = static_cast<size_t>(p_->nslice) * p_->npart;
    p_->bytes = 0;
    auto alloc = [&](size_t n) -> id<MTLBuffer> {
        id<MTLBuffer> b = [p_->dev newBufferWithLength:n options:kShared];
        p_->bytes += n;
        return b;
    };

    p_->bX = alloc(np * sizeof(float));
    p_->bY = alloc(np * sizeof(float));
    p_->bPX = alloc(np * sizeof(float));
    p_->bPY = alloc(np * sizeof(float));
    p_->bG = alloc(np * sizeof(float));
    p_->bT = alloc(np * sizeof(float));
    p_->bCurrent = alloc(static_cast<size_t>(p_->nslice) * sizeof(float));

    p_->bField.clear();
    p_->ngrid.clear();
    for (size_t i = 0; i < field->size(); i++) {
        const int ng = field->at(i)->ngrid;
        p_->ngrid.push_back(ng);
        p_->bField.push_back(alloc(static_cast<size_t>(p_->nslice) * ng * ng *
                                   2 * sizeof(float)));
    }

    for (size_t i = 0; i < p_->bField.size(); i++) {
        if (p_->bField[i] == nil) {
            reason = "field buffer allocation failed";
            return false;
        }
    }
    if (p_->bX == nil || p_->bT == nil || p_->bCurrent == nil) {
        reason = "beam buffer allocation failed";
        return false;
    }
    return true;
}

void MetalEngine::upload(Beam *beam, std::vector<Field *> *field)
{
    const int ns = p_->nslice, np = p_->npart;
    float *x = p_->fx(), *y = p_->fy(), *px = p_->fpx(), *py = p_->fpy();
    float *g = p_->fg(), *t = p_->ft();
    const double gref = p_->gref;

    for (int is = 0; is < ns; is++) {
        const size_t o = static_cast<size_t>(is) * np;
        const Particle *src = beam->beam[is].data();
        for (int ip = 0; ip < np; ip++) {
            x[o + ip] = static_cast<float>(src[ip].x);
            y[o + ip] = static_cast<float>(src[ip].y);
            px[o + ip] = static_cast<float>(src[ip].px);
            py[o + ip] = static_cast<float>(src[ip].py);
            g[o + ip] = static_cast<float>(src[ip].gamma - gref);
            t[o + ip] = static_cast<float>(src[ip].theta);
        }
    }

    float *cur = (float *)[p_->bCurrent contents];
    for (int is = 0; is < ns; is++) {
        cur[is] = static_cast<float>(beam->current[is]);
    }

    for (size_t i = 0; i < p_->bField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        std::complex<float> *dst = (std::complex<float> *)[p_->bField[i] contents];
        for (int is = 0; is < ns; is++) {
            const std::complex<double> *src = field->at(i)->field[is].data();
            std::complex<float> *d = dst + static_cast<size_t>(is) * nn;
            for (size_t k = 0; k < nn; k++) {
                d[k] = std::complex<float>(src[k]);
            }
        }
    }
}

void MetalEngine::download(Beam *beam, std::vector<Field *> *field)
{
    const int ns = p_->nslice, np = p_->npart;
    const float *x = p_->fx(), *y = p_->fy(), *px = p_->fpx(), *py = p_->fpy();
    const float *g = p_->fg(), *t = p_->ft();
    const double gref = p_->gref;

    for (int is = 0; is < ns; is++) {
        const size_t o = static_cast<size_t>(is) * np;
        Particle *dst = beam->beam[is].data();
        for (int ip = 0; ip < np; ip++) {
            dst[ip].x = x[o + ip];
            dst[ip].y = y[o + ip];
            dst[ip].px = px[o + ip];
            dst[ip].py = py[o + ip];
            dst[ip].gamma = gref + static_cast<double>(g[o + ip]);
            dst[ip].theta = t[o + ip];
        }
    }

    for (size_t i = 0; i < p_->bField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        const std::complex<float> *src =
            (const std::complex<float> *)[p_->bField[i] contents];
        for (int is = 0; is < ns; is++) {
            std::complex<double> *dst = field->at(i)->field[is].data();
            const std::complex<float> *s = src + static_cast<size_t>(is) * nn;
            for (size_t k = 0; k < nn; k++) {
                dst[k] = std::complex<double>(s[k]);
            }
        }
    }
}

MetalEngine::SyncError MetalEngine::compare(Beam *beam,
                                            std::vector<Field *> *field) const
{
    SyncError e;
    const int ns = p_->nslice, np = p_->npart;
    const float *x = p_->fx(), *y = p_->fy(), *px = p_->fpx(), *py = p_->fpy();
    const float *g = p_->fg(), *t = p_->ft();
    const double gref = p_->gref;

    // Relative to the spread of each quantity rather than to its own value, so
    // that a coordinate passing through zero does not produce a meaningless
    // ratio. gamma is compared against its offset from gref, which is the
    // quantity actually stored.
    double scale[6] = {0, 0, 0, 0, 0, 0};
    for (int is = 0; is < ns; is++) {
        for (int ip = 0; ip < np; ip++) {
            const Particle &h = beam->beam[is][ip];
            scale[0] = std::max(scale[0], std::abs(h.x));
            scale[1] = std::max(scale[1], std::abs(h.y));
            scale[2] = std::max(scale[2], std::abs(h.px));
            scale[3] = std::max(scale[3], std::abs(h.py));
            scale[4] = std::max(scale[4], std::abs(h.gamma - gref));
            scale[5] = std::max(scale[5], std::abs(h.theta));
        }
    }
    for (int k = 0; k < 6; k++) {
        if (scale[k] == 0) {
            scale[k] = 1;
        }
    }

    for (int is = 0; is < ns; is++) {
        const size_t o = static_cast<size_t>(is) * np;
        for (int ip = 0; ip < np; ip++) {
            const Particle &h = beam->beam[is][ip];
            e.beam = std::max(e.beam, std::abs(h.x - x[o + ip]) / scale[0]);
            e.beam = std::max(e.beam, std::abs(h.y - y[o + ip]) / scale[1]);
            e.beam = std::max(e.beam, std::abs(h.px - px[o + ip]) / scale[2]);
            e.beam = std::max(e.beam, std::abs(h.py - py[o + ip]) / scale[3]);
            e.beam = std::max(e.beam,
                              std::abs((h.gamma - gref) - g[o + ip]) / scale[4]);
            e.beam = std::max(e.beam, std::abs(h.theta - t[o + ip]) / scale[5]);
        }
    }

    for (size_t i = 0; i < p_->bField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        const std::complex<float> *src =
            (const std::complex<float> *)[p_->bField[i] contents];
        double fmax = 0;
        for (int is = 0; is < ns; is++) {
            const std::complex<double> *h = field->at(i)->field[is].data();
            for (size_t k = 0; k < nn; k++) {
                fmax = std::max(fmax, std::abs(h[k]));
            }
        }
        if (fmax == 0) {
            fmax = 1;
        }
        for (int is = 0; is < ns; is++) {
            const std::complex<double> *h = field->at(i)->field[is].data();
            const std::complex<float> *s = src + static_cast<size_t>(is) * nn;
            for (size_t k = 0; k < nn; k++) {
                e.field = std::max(
                    e.field, std::abs(h[k] - std::complex<double>(s[k])) / fmax);
            }
        }
    }
    return e;
}
