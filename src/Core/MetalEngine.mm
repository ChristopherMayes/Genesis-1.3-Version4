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
#include "Undulator.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <sstream>

extern const double vacimp;
extern const double eev;

// Buffers are shared storage: on Apple Silicon the GPU and the CPU address the
// same physical pages, so there is no staging copy and no explicit
// synchronisation beyond command-buffer completion.
static const MTLResourceOptions kShared = MTLResourceStorageModeShared;

// The radix-16 transform decomposes 256 as 16 x 16: 16 threads per 256-point
// FFT, each holding 16 complex values in registers, with a single threadgroup
// exchange. That is 2.2x faster than a radix-2 shuffle ladder and also more
// accurate (two stages instead of eight). It is specific to ngrid = 256, which
// init() enforces.
//
// The propagation uses only two of the three transforms in the textbook form
//     field = IFFT(FFT(field)*expK + 2*FFT(src))/ngrid^2
// because IFFT(FFT(src))/ngrid^2 == src, so
//     field = IFFT(FFT(field)*expK)/ngrid^2 + 2*src.
// This is exact, not an approximation, and holds whenever the source filter is
// off. The expK multiply is fused into the inverse row pass and the source add
// into the inverse column pass, so the whole solve is four passes.
static NSString *kMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint N = 256, RF = 8, CC = 16;

inline float2 cmul(float2 a, float2 b){ return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }

inline void dft4(thread float2* a, float s){
    float2 t0=a[0]+a[2], t1=a[0]-a[2], t2=a[1]+a[3], d=a[1]-a[3];
    float2 t3 = float2(s*d.y, -s*d.x);
    a[0]=t0+t2; a[1]=t1+t3; a[2]=t0-t2; a[3]=t1-t3;
}
// 16-point DFT in registers, in place. Output is left permuted:
// slot j holds frequency perm(j) = 4*(j&3) + (j>>2). The caller folds that
// permutation into its store index, which avoids a 16-element temporary.
inline void dft16(thread float2* a, const device float2* W, float s){
    float2 t[4];
    for (uint n2=0;n2<4;n2++){
        for (uint j=0;j<4;j++) t[j]=a[4*j+n2];
        dft4(t,s);
        for (uint k1=0;k1<4;k1++){
            float2 w = W[(16u*n2*k1) & 255u]; w.y *= s;
            a[4*k1+n2] = cmul(t[k1], w);
        }
    }
    for (uint k1=0;k1<4;k1++){
        for (uint n2=0;n2<4;n2++) t[n2]=a[4*k1+n2];
        dft4(t,s);
        for (uint k2=0;k2<4;k2++) a[4*k1+k2]=t[k2];
    }
}
// 256-point FFT across 16 threads. On entry a[n1] = x[16*n1+lane];
// on exit slot j holds X[16*perm(j)+lane].
// The exchange index is XOR-swizzled to avoid 16-way bank conflicts without
// padding, which is what lets 16 columns share one threadgroup (exactly the
// 32 KB limit).
inline void fft256_r16(thread float2* a, threadgroup float2* s,
                       const device float2* W, uint lane, float sgn){
    dft16(a, W, sgn);
    for (uint j=0;j<16;j++){
        uint k1 = 4u*(j&3u) + (j>>2);
        float2 w = W[(lane*k1) & 255u]; w.y *= sgn;
        s[k1*16u + (lane ^ k1)] = cmul(a[j], w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n2=0;n2<16;n2++) a[n2] = s[lane*16u + (n2 ^ lane)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dft16(a, W, sgn);
}

kernel void r16_rows(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                     constant float& sgn [[buffer(2)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[RF*256];
    uint lane = t & 15u, r = t >> 4;
    device float2* p = d + (ulong)tg.y*(N*N) + (ulong)(tg.x*RF + r)*N;
    float2 a[16];
    for (uint n1=0;n1<16;n1++) a[n1] = p[16u*n1 + lane];
    fft256_r16(a, sh + r*256u, W, lane, sgn);
    for (uint j=0;j<16;j++) p[16u*(4u*(j&3u)+(j>>2)) + lane] = a[j];
}
kernel void r16_rows_mul(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                         constant float& sgn [[buffer(2)]], const device float2* expK [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[RF*256];
    uint lane = t & 15u, r = t >> 4;
    ulong row = (ulong)(tg.x*RF + r);
    device float2* p = d + (ulong)tg.y*(N*N) + row*N;
    const device float2* k = expK + row*N;
    float2 a[16];
    for (uint n1=0;n1<16;n1++) a[n1] = cmul(p[16u*n1 + lane], k[16u*n1 + lane]);
    fft256_r16(a, sh + r*256u, W, lane, sgn);
    for (uint j=0;j<16;j++) p[16u*(4u*(j&3u)+(j>>2)) + lane] = a[j];
}
kernel void r16_cols(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                     constant float& sgn [[buffer(2)]], constant float& scale [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[CC*256];
    uint c = t % CC, lane = t / CC;
    device float2* b = d + (ulong)tg.y*(N*N) + (ulong)tg.x*CC + c;
    float2 a[16];
    for (uint n1=0;n1<16;n1++) a[n1] = b[(ulong)(16u*n1+lane)*N];
    fft256_r16(a, sh + c*256u, W, lane, sgn);
    for (uint j=0;j<16;j++) b[(ulong)(16u*(4u*(j&3u)+(j>>2))+lane)*N] = a[j]*scale;
}
kernel void r16_cols_add(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                         constant float& sgn [[buffer(2)]], constant float& scale [[buffer(3)]],
                         const device float2* src [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[CC*256];
    uint c = t % CC, lane = t / CC;
    ulong off = (ulong)tg.y*(N*N) + (ulong)tg.x*CC + c;
    device float2* b = d + off; const device float2* sb = src + off;
    float2 a[16];
    for (uint n1=0;n1<16;n1++) a[n1] = b[(ulong)(16u*n1+lane)*N];
    fft256_r16(a, sh + c*256u, W, lane, sgn);
    for (uint j=0;j<16;j++){
        ulong q = (ulong)(16u*(4u*(j&3u)+(j>>2))+lane)*N;
        b[q] = a[j]*scale + 2.0f*sb[q];
    }
}

// ---------------- source deposition ----------------

struct DepPar {
    float gridmax, dgrid, gref, scl;
    float ax, ay, kx, ky, gradx, grady;
    uint  ngrid, npart, nslice, harm, first;
};

kernel void zero_src(device float* s [[buffer(0)]], uint i [[thread_position_in_grid]]){
    s[i] = 0.0f;
}

// One thread per particle. Mirrors FieldSolverFFT::advance: bilinear scatter of
// sqrt(faw2)*scl/gamma * (sin(h*theta), cos(h*theta)) onto the lower-left grid
// point and its three neighbours.
kernel void deposit(device atomic_float* src [[buffer(0)]],
                    const device float* X    [[buffer(1)]],
                    const device float* Y    [[buffer(2)]],
                    const device float* G    [[buffer(3)]],
                    const device float* TH   [[buffer(4)]],
                    const device float* CUR  [[buffer(5)]],
                    constant DepPar& P       [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]){
    uint is = gid / P.npart;
    if (is >= P.nslice) return;

    float x = X[gid], y = Y[gid];
    if (!(x > -P.gridmax && x < P.gridmax && y > -P.gridmax && y < P.gridmax)) return;

    float wx = (x + P.gridmax)/P.dgrid;
    float wy = (y + P.gridmax)/P.dgrid;
    float fx = floor(wx), fy = floor(wy);
    wx = 1.0f + fx - wx;
    wy = 1.0f + fy - wy;

    // The beam slice interacts with field slice (is + first) % nslice, so the
    // source is written straight into field-slice order.
    uint fs   = (is + P.first) % P.nslice;
    uint idx  = fs*(P.ngrid*P.ngrid) + uint(fx) + uint(fy)*P.ngrid;

    float dx = x - P.ax, dy = y - P.ay;
    float faw2 = 1.0f + P.kx*dx*dx + P.ky*dy*dy + 2.0f*(P.gradx*dx + P.grady*dy);
    float part = sqrt(faw2) * (P.scl*CUR[is]) / (P.gref + G[gid]);
    float th   = float(P.harm) * TH[gid];
    float vr = sin(th)*part, vi = cos(th)*part;

    float w;
    uint d;
    w = wx*wy;              d = 2u*idx;
    atomic_fetch_add_explicit(&src[d  ], w*vr, memory_order_relaxed);
    atomic_fetch_add_explicit(&src[d+1], w*vi, memory_order_relaxed);
    w = (1.0f-wx)*wy;       d = 2u*(idx+1u);
    atomic_fetch_add_explicit(&src[d  ], w*vr, memory_order_relaxed);
    atomic_fetch_add_explicit(&src[d+1], w*vi, memory_order_relaxed);
    w = wx*(1.0f-wy);       d = 2u*(idx+P.ngrid);
    atomic_fetch_add_explicit(&src[d  ], w*vr, memory_order_relaxed);
    atomic_fetch_add_explicit(&src[d+1], w*vi, memory_order_relaxed);
    w = (1.0f-wx)*(1.0f-wy); d = 2u*(idx+P.ngrid+1u);
    atomic_fetch_add_explicit(&src[d  ], w*vr, memory_order_relaxed);
    atomic_fetch_add_explicit(&src[d+1], w*vi, memory_order_relaxed);
}
)MSL";

// Mirrors the MSL struct above.
struct DepPar {
    float gridmax, dgrid, gref, scl;
    float ax, ay, kx, ky, gradx, grady;
    uint32_t ngrid, npart, nslice, harm, first;
};

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

    // Field solve scratch. The source is reused across harmonics because they
    // are propagated one at a time. expK depends on the harmonic (through xks)
    // and on the step length, so it is cached per harmonic and rebuilt when
    // delz changes.
    id<MTLBuffer> bSrc {nil};
    id<MTLBuffer> bW {nil};
    std::vector<id<MTLBuffer> > bExpK;
    std::vector<double> delzCached;

    id<MTLComputePipelineState> pZero, pDep, pRow, pRowM, pCol, pColA;

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
        // The radix-16 field solver is written for a 16 x 16 decomposition.
        if (f->ngrid != 256) {
            std::ostringstream os;
            os << "ngrid = " << f->ngrid
               << ": the Metal field solver currently implements only ngrid = 256";
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

    // Field solve scratch. One source buffer is enough because the harmonics
    // are propagated one after another.
    const int ng = p_->ngrid[0];
    const size_t nn = static_cast<size_t>(ng) * ng;
    p_->bSrc = alloc(static_cast<size_t>(p_->nslice) * nn * 2 * sizeof(float));
    p_->bW = alloc(static_cast<size_t>(ng) * 2 * sizeof(float));
    p_->bExpK.clear();
    p_->delzCached.assign(field->size(), -1.0);
    for (size_t i = 0; i < field->size(); i++) {
        p_->bExpK.push_back(alloc(nn * 2 * sizeof(float)));
    }
    if (p_->bSrc == nil || p_->bW == nil) {
        reason = "field solver buffer allocation failed";
        return false;
    }

    std::complex<float> *W = (std::complex<float> *)[p_->bW contents];
    for (int m = 0; m < ng; m++) {
        const double a = -2.0 * M_PI * m / ng;
        W[m] = std::complex<float>(static_cast<float>(cos(a)),
                                   static_cast<float>(sin(a)));
    }

    NSError *err = nil;
    id<MTLLibrary> lib = [p_->dev newLibraryWithSource:kMSL options:nil error:&err];
    if (lib == nil) {
        reason = std::string("shader compilation failed: ") +
                 [[err localizedDescription] UTF8String];
        return false;
    }
    bool ok = true;
    auto pso = [&](NSString *name) -> id<MTLComputePipelineState> {
        NSError *e = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        id<MTLComputePipelineState> s =
            (fn == nil) ? nil
                        : [p_->dev newComputePipelineStateWithFunction:fn error:&e];
        if (s == nil) {
            ok = false;
        }
        return s;
    };
    p_->pZero = pso(@"zero_src");
    p_->pDep = pso(@"deposit");
    p_->pRow = pso(@"r16_rows");
    p_->pRowM = pso(@"r16_rows_mul");
    p_->pCol = pso(@"r16_cols");
    p_->pColA = pso(@"r16_cols_add");
    if (!ok) {
        reason = "compute pipeline creation failed";
        return false;
    }
    return true;
}

void MetalEngine::fieldStep(Undulator *und, std::vector<Field *> *field,
                            double delz)
{
    @autoreleasepool {
        const int ns = p_->nslice;
        const int istep = und->getStep();
        const float fwd = 1.0f, inv = -1.0f, one = 1.0f;

        id<MTLCommandBuffer> cb = [p_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

        for (size_t ih = 0; ih < p_->bField.size(); ih++) {
            Field *f = field->at(ih);
            const int ng = p_->ngrid[ih];
            const size_t nn = static_cast<size_t>(ng) * ng;
            const float nrm = 1.0f / static_cast<float>(nn);

            // exp(K2*delz) only changes when the step length does.
            if (p_->delzCached[ih] != delz) {
                std::complex<float> *K =
                    (std::complex<float> *)[p_->bExpK[ih] contents];
                const double dk = 4.0 * asin(1.0) / (ng * f->dgrid);
                const double shift = -0.5 * (ng - 1);
                for (int iy = 0; iy < ng; iy++) {
                    const double dy = iy + shift;
                    for (int ix = 0; ix < ng; ix++) {
                        const double dx = ix + shift;
                        const int ii =
                            ((iy + (ng + 1) / 2) % ng) * ng + ((ix + (ng + 1) / 2) % ng);
                        const std::complex<double> v = std::exp(
                            std::complex<double>(0, -(dx * dx + dy * dy) * dk * dk /
                                                        2.0 / f->xks) *
                            delz);
                        K[ii] = std::complex<float>(v);
                    }
                }
                p_->delzCached[ih] = delz;
            }

            // Zero the source, then deposit if this harmonic couples. Even
            // harmonics have fc == 0 and are skipped, as on the CPU.
            [e setComputePipelineState:p_->pZero];
            [e setBuffer:p_->bSrc offset:0 atIndex:0];
            [e dispatchThreads:MTLSizeMake(static_cast<size_t>(ns) * nn * 2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            const int harm = f->getHarm();
            if (und->inUndulator() && f->isEnabled() && (harm % 2 == 1)) {
                double scl = und->fc(harm) * vacimp * f->xks * delz;
                scl /= 4 * eev * static_cast<double>(p_->npart) * f->dgrid * f->dgrid;

                DepPar P;
                P.gridmax = static_cast<float>(f->gridmax);
                P.dgrid = static_cast<float>(f->dgrid);
                P.gref = static_cast<float>(p_->gref);
                P.scl = static_cast<float>(scl);
                P.ax = static_cast<float>(und->ax[istep]);
                P.ay = static_cast<float>(und->ay[istep]);
                P.kx = static_cast<float>(und->kx[istep]);
                P.ky = static_cast<float>(und->ky[istep]);
                P.gradx = static_cast<float>(und->gradx[istep]);
                P.grady = static_cast<float>(und->grady[istep]);
                P.ngrid = static_cast<uint32_t>(ng);
                P.npart = static_cast<uint32_t>(p_->npart);
                P.nslice = static_cast<uint32_t>(ns);
                P.harm = static_cast<uint32_t>(harm);
                P.first = static_cast<uint32_t>(f->first);

                [e setComputePipelineState:p_->pDep];
                [e setBuffer:p_->bSrc offset:0 atIndex:0];
                [e setBuffer:p_->bX offset:0 atIndex:1];
                [e setBuffer:p_->bY offset:0 atIndex:2];
                [e setBuffer:p_->bG offset:0 atIndex:3];
                [e setBuffer:p_->bT offset:0 atIndex:4];
                [e setBuffer:p_->bCurrent offset:0 atIndex:5];
                [e setBytes:&P length:sizeof(P) atIndex:6];
                [e dispatchThreads:MTLSizeMake(static_cast<size_t>(ns) * p_->npart, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // field = IFFT(FFT(field)*expK)/ngrid^2 + 2*src, four passes.
            const MTLSize rowTG = MTLSizeMake(ng / 8, ns, 1);
            const MTLSize rowT = MTLSizeMake(8 * 16, 1, 1);
            const MTLSize colTG = MTLSizeMake(ng / 16, ns, 1);
            const MTLSize colT = MTLSizeMake(16 * 16, 1, 1);

            [e setComputePipelineState:p_->pRow];
            [e setBuffer:p_->bField[ih] offset:0 atIndex:0];
            [e setBuffer:p_->bW offset:0 atIndex:1];
            [e setBytes:&fwd length:4 atIndex:2];
            [e dispatchThreadgroups:rowTG threadsPerThreadgroup:rowT];

            [e setComputePipelineState:p_->pCol];
            [e setBuffer:p_->bField[ih] offset:0 atIndex:0];
            [e setBuffer:p_->bW offset:0 atIndex:1];
            [e setBytes:&fwd length:4 atIndex:2];
            [e setBytes:&one length:4 atIndex:3];
            [e dispatchThreadgroups:colTG threadsPerThreadgroup:colT];

            [e setComputePipelineState:p_->pRowM];
            [e setBuffer:p_->bField[ih] offset:0 atIndex:0];
            [e setBuffer:p_->bW offset:0 atIndex:1];
            [e setBytes:&inv length:4 atIndex:2];
            [e setBuffer:p_->bExpK[ih] offset:0 atIndex:3];
            [e dispatchThreadgroups:rowTG threadsPerThreadgroup:rowT];

            [e setComputePipelineState:p_->pColA];
            [e setBuffer:p_->bField[ih] offset:0 atIndex:0];
            [e setBuffer:p_->bW offset:0 atIndex:1];
            [e setBytes:&inv length:4 atIndex:2];
            [e setBytes:&nrm length:4 atIndex:3];
            [e setBuffer:p_->bSrc offset:0 atIndex:4];
            [e dispatchThreadgroups:colTG threadsPerThreadgroup:colT];
        }

        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
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

void MetalEngine::downloadField(std::vector<Field *> *field)
{
    for (size_t i = 0; i < p_->bField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        const std::complex<float> *src =
            (const std::complex<float> *)[p_->bField[i] contents];
        for (int is = 0; is < p_->nslice; is++) {
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
