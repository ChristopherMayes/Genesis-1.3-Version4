// NVIDIA (CUDA) GPU backend.
//
// Compiled only when the project is configured with -DENABLE_CUDA=ON, which
// also defines G4_CUDA. See include/GPUEngine.h for what the interface requires
// of any backend and why, include/CudaEngine.h for what is specific to this
// one, and manual/GPU.md for the physics each kernel reproduces.
//
// The arithmetic below is a transcription of src/Core/MetalEngine.mm and is
// deliberately identical to it, down to the order of operations inside the
// transform, so that the two backends agree to the last bit of the FP32
// rounding and the accuracy figures measured for one apply to the other. Three
// things about this file are not transcriptions, and they are the ones to read
// carefully.
//
// 1. The memory is not unified. Every buffer is a device allocation; the host
//    reaches it through a pinned staging area and an explicit copy on the
//    engine's stream. The places that copy every step are small and are listed
//    in manual/GPU.md; each is, however, an ordering constraint, and the rule
//    that keeps them correct is the same one Metal needed: nothing the host
//    writes may be overwritten while a copy that reads it is still queued, so
//    beamStep() drains the stream once, at the top, before it writes anything.
//
// 2. The transform shape is a template parameter rather than a preprocessor
//    macro handed to a runtime compiler. Every supported grid size is
//    instantiated at build time, which removes the startup cost that the Metal
//    backend pays for compiling its shader library.
//
// 3. The device is chosen from the rank's position within its node. See
//    selectDevice().
//
// The parameter structs are shared between host and device here, rather than
// written twice as they must be in Metal, which removes the whole class of
// mistake the comments in MetalEngine.mm warn about.

#include "CudaEngine.h"

// The transform, the complex helpers and the launch table it is dispatched
// through. In a header of its own so that tools/fftcheck.cu can exercise the
// same code without linking the rest of Genesis.
#include "CudaFFT.cuh"

#include "Beam.h"
#include "Field.h"
#include "TrackBeam.h"
#include "Undulator.h"

#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

extern const double vacimp;
extern const double eev;
extern bool MPISingle;

namespace {

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

// A CUDA call that fails here means the device is gone, the allocation was
// refused or a kernel faulted, none of which the tracking loop can carry on
// from. init() reports its own failures through 'reason' and returns false; a
// failure anywhere later is not recoverable and is reported rather than
// swallowed, because a run that continues past a faulted kernel writes an
// output file full of whatever was in the buffers.
bool g_cudaFailed = false;
std::string g_cudaError;

bool cuOK(cudaError_t e, const char *what)
{
    if (e == cudaSuccess) { return true; }
    if (!g_cudaFailed) {
        g_cudaFailed = true;
        g_cudaError = std::string(what) + ": " + cudaGetErrorString(e);
        std::fprintf(stderr, "\n*** CUDA error in %s: %s\n", what,
                     cudaGetErrorString(e));
    }
    return false;
}

#define CU(call) cuOK((call), #call)

// ---------------------------------------------------------------------------
// block reductions
// ---------------------------------------------------------------------------
//
// The reduction kernels all run with exactly kRedThreads threads, which is what
// makes the fold across warps a fixed-length loop that every thread executes,
// so that every thread leaves with the same answer. That property is used: the
// second pass of the moments needs the mean in every thread, not just in one.

enum { kRedThreads = 256, kNWarp = kRedThreads / 32 };

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
    return v;
}
__device__ __forceinline__ float warpMin(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fminf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}
__device__ __forceinline__ float warpMax(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

__device__ __forceinline__ float blockSum(float v, float *sh, unsigned t)
{
    v = warpSum(v);
    if ((t & 31u) == 0u) { sh[t >> 5] = v; }
    __syncthreads();
    float r = 0;
#pragma unroll
    for (int i = 0; i < kNWarp; i++) { r += sh[i]; }
    __syncthreads();
    return r;
}
__device__ __forceinline__ float blockMin(float v, float *sh, unsigned t)
{
    v = warpMin(v);
    if ((t & 31u) == 0u) { sh[t >> 5] = v; }
    __syncthreads();
    float r = sh[0];
#pragma unroll
    for (int i = 1; i < kNWarp; i++) { r = fminf(r, sh[i]); }
    __syncthreads();
    return r;
}
__device__ __forceinline__ float blockMax(float v, float *sh, unsigned t)
{
    v = warpMax(v);
    if ((t & 31u) == 0u) { sh[t >> 5] = v; }
    __syncthreads();
    float r = sh[0];
#pragma unroll
    for (int i = 1; i < kNWarp; i++) { r = fmaxf(r, sh[i]); }
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------------------
// source deposition
// ---------------------------------------------------------------------------

struct DepPar {
    float gridmax, dgrid, gref, scl;
    float ax, ay, kx, ky, gradx, grady;
    unsigned ngrid, npart, nslice, harm, first;
};

// One thread per particle. Mirrors FieldSolverFFT::advance: bilinear scatter of
// sqrt(faw2)*scl/gamma * (sin(h*theta), cos(h*theta)) onto the lower-left grid
// point and its three neighbours. CUDA has native float atomics in global
// memory, so this is four pairs of atomicAdd rather than Metal's
// compare-and-swap on the bit pattern.
__global__ void kDeposit(float *__restrict__ src, const float *__restrict__ X,
                         const float *__restrict__ Y, const float *__restrict__ G,
                         const float *__restrict__ TH, const float *__restrict__ CUR,
                         DepPar P, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }
    const unsigned is = static_cast<unsigned>(gid / P.npart);

    const float x = X[gid], y = Y[gid];
    if (!(x > -P.gridmax && x < P.gridmax && y > -P.gridmax && y < P.gridmax)) {
        return;
    }

    float wx = (x + P.gridmax) / P.dgrid;
    float wy = (y + P.gridmax) / P.dgrid;
    const float fx = floorf(wx), fy = floorf(wy);
    wx = 1.0f + fx - wx;
    wy = 1.0f + fy - wy;

    // The beam slice interacts with field slice (is + first) % nslice, so the
    // source is written straight into field-slice order.
    const unsigned fs = (is + P.first) % P.nslice;
    const unsigned idx =
        fs * (P.ngrid * P.ngrid) + static_cast<unsigned>(fx) +
        static_cast<unsigned>(fy) * P.ngrid;

    const float dx = x - P.ax, dy = y - P.ay;
    const float faw2 =
        1.0f + P.kx * dx * dx + P.ky * dy * dy + 2.0f * (P.gradx * dx + P.grady * dy);
    const float part = sqrtf(faw2) * (P.scl * CUR[is]) / (P.gref + G[gid]);
    const float th = static_cast<float>(P.harm) * TH[gid];
    float sth, cth;
    sincosf(th, &sth, &cth);
    const float vr = sth * part, vi = cth * part;

    float w;
    unsigned d;
    w = wx * wy;                  d = 2u * idx;
    atomicAdd(&src[d], w * vr);
    atomicAdd(&src[d + 1], w * vi);
    w = (1.0f - wx) * wy;         d = 2u * (idx + 1u);
    atomicAdd(&src[d], w * vr);
    atomicAdd(&src[d + 1], w * vi);
    w = wx * (1.0f - wy);         d = 2u * (idx + P.ngrid);
    atomicAdd(&src[d], w * vr);
    atomicAdd(&src[d + 1], w * vi);
    w = (1.0f - wx) * (1.0f - wy); d = 2u * (idx + P.ngrid + 1u);
    atomicAdd(&src[d], w * vr);
    atomicAdd(&src[d + 1], w * vi);
}

// ---------------------------------------------------------------------------
// transverse tracking
// ---------------------------------------------------------------------------

struct TrkPar {
    float delz, aw, qx, qy, xoff, yoff, gref;
    float cpx, cpy;   // corrector kick, already scaled by gamma_ref
    float cm[8];      // chicane map, rows 00 01 10 11 22 23 32 33
    unsigned mx, my;  // 0 = drift, 1 = focusing, 2 = defocusing
    unsigned doMap;   // 1 if the chicane map is to be applied
};

__global__ void kTrackBeam(float *__restrict__ X, float *__restrict__ Y,
                           float *__restrict__ PX, float *__restrict__ PY,
                           const float *__restrict__ G, TrkPar P, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }

    const float gam = P.gref + G[gid];
    float x = X[gid], y = Y[gid];
    float px = PX[gid], py = PY[gid];

    if (P.doMap != 0u) {
        // TrackBeam::applyChicane. Its gamma*beta_z leaves out aw, because a
        // chicane sits outside the undulator, so it is formed separately from
        // the one the transverse map below uses.
        const float rc = (1.0f + px * px + py * py) / (gam * gam);
        const float gc = gam * sqrtf(fmaxf(0.0f, 1.0f - rc));
        float t = x;
        x = P.cm[0] * t + P.cm[1] * px / gc;
        px = P.cm[2] * t * gc + P.cm[3] * px;
        t = y;
        y = P.cm[4] * t + P.cm[5] * py / gc;
        py = P.cm[6] * t * gc + P.cm[7] * py;
    }

    // TrackBeam::applyCorrector runs before the transverse map and therefore
    // before gamma*beta_z is formed, so the kick is added here rather than to
    // the stored momenta afterwards.
    px += P.cpx;
    py += P.cpy;

    // gamma*beta_z. Written as gamma*sqrt(1-r) rather than
    // sqrt(gamma^2-1-aw^2-p^2): at gamma = 11357 the FP32 quantum of gamma^2 is
    // 15, so the O(1) terms would be lost completely in the subtraction.
    const float r = (1.0f + P.aw * P.aw + px * px + py * py) / (gam * gam);
    const float gz = gam * sqrtf(fmaxf(0.0f, 1.0f - r));

    if (P.mx == 0u) {
        x += px * P.delz / gz;
    } else {
        const float foc = sqrtf(fabsf(P.qx) / gz), omg = foc * P.delz;
        float a1, a2, a3;
        if (P.mx == 1u) {
            a1 = cosf(omg);  a2 = sinf(omg) / foc;  a3 = -a2 * foc * foc;
        } else {
            a1 = coshf(omg); a2 = sinhf(omg) / foc; a3 = a2 * foc * foc;
        }
        const float xt = x - P.xoff;
        x = a1 * xt + a2 * px / gz + P.xoff;
        px = a3 * xt * gz + a1 * px;
    }
    if (P.my == 0u) {
        y += py * P.delz / gz;
    } else {
        const float foc = sqrtf(fabsf(P.qy) / gz), omg = foc * P.delz;
        float a1, a2, a3;
        if (P.my == 1u) {
            a1 = cosf(omg);  a2 = sinf(omg) / foc;  a3 = -a2 * foc * foc;
        } else {
            a1 = coshf(omg); a2 = sinhf(omg) / foc; a3 = a2 * foc * foc;
        }
        const float yt = y - P.yoff;
        y = a1 * yt + a2 * py / gz + P.yoff;
        py = a3 * yt * gz + a1 * py;
    }
    X[gid] = x;  PX[gid] = px;
    Y[gid] = y;  PY[gid] = py;
}

// ---------------------------------------------------------------------------
// longitudinal push
// ---------------------------------------------------------------------------

enum { kMaxHarm = 4 };

struct PushPar {
    const float2 *F[kMaxHarm];
    float delz, aw, xks, xku, gref, autophase;
    float ax, ay, kx, ky, gradx, grady;
    float gridmax, dgrid;
    unsigned ngrid, npart, nslice, nfld;
    unsigned useSR;          // 1 if a per-particle space-charge field is supplied
    float rtmp[kMaxHarm];    // und->fc(harm) / field->xks
    float rharm[kMaxHarm];
    unsigned first[kMaxHarm];
};

// The ODE is evaluated entirely in FP32, which forces two reformulations.
//
// 1. gamma is carried as an offset dg from a reference gref, because at
//    gamma = 11357 the FP32 quantum of absolute gamma (1.35e-3) is larger than
//    the per-step energy change (3.0e-4 at saturation, 5.5e-7 at seed).
//
// 2. The literal expression for the detuning,
//        k2pp = xks*(1 - 1/sqrt(1-u)) + xku,
//    is catastrophic in FP32: u = btper0/gamma^2 is 1.3e-8, below the FP32
//    epsilon of 1.2e-7, so (1-u) rounds to exactly 1 and the detuning collapses
//    to zero. The series 1 - 1/sqrt(1-u) = -(u/2)(1 + 3u/4) never forms (1-u)
//    and is accurate to ~5e-5 rad/m, against a physical detuning spread of
//    7.4e-2 rad/m for delgam = 1.
struct Acc {
    float gg, pp;
};

__device__ __forceinline__ Acc ode(float dg, float th, float btpar, float ez,
                                   const float2 *rpart, const PushPar &P, Acc k)
{
    float2 ctmp = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < kMaxHarm; i++) {
        if (static_cast<unsigned>(i) < P.nfld) {
            const float a = P.rharm[i] * th;
            float sa, ca;
            sincosf(a, &sa, &ca);
            const float2 e = make_float2(ca, -sa);
            ctmp += make_float2(rpart[i].x * e.x - rpart[i].y * e.y,
                                rpart[i].x * e.y + rpart[i].y * e.x);
        }
    }
    const float tgam = P.gref + dg;
    const float btper0 = btpar - (2.0f / P.xks) * ctmp.x;
    const float u = btper0 / (tgam * tgam);
    const float invb = 1.0f + 0.5f * u + 0.375f * u * u;    // 1/sqrt(1-u)
    k.pp += -P.xks * 0.5f * u * (1.0f + 0.75f * u) + P.xku; // dtheta/dz
    k.gg += ctmp.y * invb / tgam - ez;                      // dgamma/dz
    return k;
}

__global__ void kPushBeam(float *__restrict__ G, float *__restrict__ TH,
                          const float *__restrict__ X, const float *__restrict__ Y,
                          const float *__restrict__ PX, const float *__restrict__ PY,
                          const float *__restrict__ EZ, const float *__restrict__ EZP,
                          PushPar P, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }
    const unsigned is = static_cast<unsigned>(gid / P.npart);

    const float x = X[gid], y = Y[gid], px = PX[gid], py = PY[gid];
    const float dx = x - P.ax, dy = y - P.ay;
    const float awloc =
        1.0f + 0.5f * (P.kx * dx * dx + P.ky * dy * dy) + P.gradx * dx + P.grady * dy;
    const float btpar = 1.0f + px * px + py * py + P.aw * P.aw * awloc * awloc;
    // The long-range field is one number per slice; the short-range solve adds
    // one per particle, as BeamSolver::advance sums them.
    // The short-range buffer is allocated at full size whether or not the
    // solver is on, so this stays a plain add rather than a predicated load
    // that a compiler is free to hoist out of its guard.
    const float ez = EZ[is] + (P.useSR != 0u ? EZP[gid] : 0.0f);

    // Bilinear gather of each harmonic at the particle position.
    float2 rpart[kMaxHarm];
#pragma unroll
    for (int i = 0; i < kMaxHarm; i++) { rpart[i] = make_float2(0.0f, 0.0f); }

    const bool on = (x > -P.gridmax && x < P.gridmax && y > -P.gridmax && y < P.gridmax);
    if (on) {
        float wx = (x + P.gridmax) / P.dgrid, wy = (y + P.gridmax) / P.dgrid;
        const float fx = floorf(wx), fy = floorf(wy);
        wx = 1.0f + fx - wx;
        wy = 1.0f + fy - wy;
        const unsigned c =
            static_cast<unsigned>(fx) + static_cast<unsigned>(fy) * P.ngrid;
#pragma unroll
        for (int i = 0; i < kMaxHarm; i++) {
            if (static_cast<unsigned>(i) < P.nfld) {
                const float2 *F = P.F[i];
                const unsigned b =
                    ((is + P.first[i]) % P.nslice) * (P.ngrid * P.ngrid) + c;
                const float2 cp = F[b] * (wx * wy) +
                                  F[b + 1u] * ((1.0f - wx) * wy) +
                                  F[b + P.ngrid] * (wx * (1.0f - wy)) +
                                  F[b + P.ngrid + 1u] * ((1.0f - wx) * (1.0f - wy));
                // rtmp*awloc*conj(cpart)
                const float s = P.rtmp[i] * awloc;
                rpart[i] = make_float2(s * cp.x, -s * cp.y);
            }
        }
    }

    float dg = G[gid], th = TH[gid] + P.autophase;

    // Classic RK4, transcribed from BeamSolver::RungeKutta.
    Acc k2 = ode(dg, th, btpar, ez, rpart, P, Acc{0.0f, 0.0f});
    float stpz = 0.5f * P.delz;
    dg += stpz * k2.gg;  th += stpz * k2.pp;
    Acc k3 = k2;
    k2 = ode(dg, th, btpar, ez, rpart, P, Acc{0.0f, 0.0f});
    dg += stpz * (k2.gg - k3.gg);  th += stpz * (k2.pp - k3.pp);
    k3.gg /= 6.0f;  k3.pp /= 6.0f;
    k2.gg *= -0.5f; k2.pp *= -0.5f;
    k2 = ode(dg, th, btpar, ez, rpart, P, k2);
    stpz = P.delz;
    dg += stpz * k2.gg;  th += stpz * k2.pp;
    k3.gg -= k2.gg; k3.pp -= k2.pp;
    k2.gg *= 2.0f;  k2.pp *= 2.0f;
    k2 = ode(dg, th, btpar, ez, rpart, P, k2);
    dg += stpz * (k3.gg + k2.gg / 6.0f);
    th += stpz * (k3.pp + k2.pp / 6.0f);

    G[gid] = dg;
    TH[gid] = th;
}

// ---------------------------------------------------------------------------
// collective kicks
// ---------------------------------------------------------------------------

// Wakefield energy loss. The wake is a single number per slice, because it is
// driven by the slice current rather than by individual particles, so every
// particle in a slice takes the same kick. The profile is built on the host,
// where the current of every slice is available through an MPI gather; all that
// is left here is the addition. G holds gamma as an offset from a reference
// energy, and the kick is a difference, so it applies unchanged to the offset.
__global__ void kApplyELoss(float *__restrict__ G, const float *__restrict__ DG,
                            unsigned npart, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }
    G[gid] += DG[gid / npart];
}

// Incoherent synchrotron radiation: one value per beamlet rather than per
// particle, because Incoherent::apply draws once every nbins particles and
// gives the whole beamlet the same kick, so that the shot noise the beamlet
// carries is not disturbed. The numbers themselves are drawn on the host from
// the generator Genesis already uses, in the order the CPU path consumes them,
// which is what lets a GPU run reproduce a CPU run exactly instead of merely
// statistically.
__global__ void kApplyISR(float *__restrict__ G, const float *__restrict__ DG,
                          unsigned npart, unsigned nbins, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }
    const unsigned nbeamlet = (npart + nbins - 1u) / nbins;
    G[gid] += DG[(gid / npart) * nbeamlet + (gid % npart) / nbins];
}

// Longitudinal phase shift through a chicane, TrackBeam::applyR56. r56 is
// already scaled by 2*pi/(lambda*gamma_ref); dgref is the difference between
// the energy the offsets are stored against and the lattice reference energy,
// so that G + dgref is the same gamma - gamma_ref the CPU path forms.
__global__ void kApplyR56(float *__restrict__ TH, const float *__restrict__ G,
                          float r56, float dgref, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }
    TH[gid] += r56 * (G[gid] + dgref);
}

// ---------------------------------------------------------------------------
// diagnostic reductions
// ---------------------------------------------------------------------------
//
// One block of kRedThreads per slice. The partial sums are folded first across
// each warp and then across the eight warps, which keeps the shared traffic to
// eight floats per quantity.

enum { kMaxBunchHarm = 8, kBeamStride = 48, kFieldStride = 16 };

struct BMPar {
    float gref;
    unsigned npart, nharm, doAux;
};

__global__ void kBeamMoments(const float *__restrict__ X, const float *__restrict__ Y,
                             const float *__restrict__ PX, const float *__restrict__ PY,
                             const float *__restrict__ G, const float *__restrict__ TH,
                             float *__restrict__ out, BMPar P)
{
    __shared__ float sh[kNWarp];
    const unsigned t = threadIdx.x, nt = blockDim.x, slice = blockIdx.x;
    const unsigned np = P.npart;
    const size_t o = static_cast<size_t>(slice) * np;

    // first pass: means, and the bunching factor which needs no centring
    float sx = 0, sy = 0, spx = 0, spy = 0, sg = 0;
    float2 b[kMaxBunchHarm];
#pragma unroll
    for (int h = 0; h < kMaxBunchHarm; h++) { b[h] = make_float2(0.0f, 0.0f); }
    for (unsigned i = t; i < np; i += nt) {
        sx += X[o + i]; sy += Y[o + i]; spx += PX[o + i]; spy += PY[o + i];
        sg += G[o + i];
        const float th = TH[o + i];
#pragma unroll
        for (int h = 0; h < kMaxBunchHarm; h++) {
            if (static_cast<unsigned>(h) < P.nharm) {
                const float a = static_cast<float>(h + 1) * th;
                float sa, ca;
                sincosf(a, &sa, &ca);
                b[h] += make_float2(ca, sa);
            }
        }
    }
    sx = blockSum(sx, sh, t);  sy = blockSum(sy, sh, t);
    spx = blockSum(spx, sh, t); spy = blockSum(spy, sh, t); sg = blockSum(sg, sh, t);
#pragma unroll
    for (int h = 0; h < kMaxBunchHarm; h++) {
        if (static_cast<unsigned>(h) < P.nharm) {
            b[h].x = blockSum(b[h].x, sh, t);
            b[h].y = blockSum(b[h].y, sh, t);
        }
    }

    const float n = 1.0f / static_cast<float>(np);
    const float mx = sx * n, my = sy * n, mpx = spx * n, mpy = spy * n, mg = sg * n;

    // second pass: centred second moments
    float cx = 0, cy = 0, cpx = 0, cpy = 0, cg = 0, cxpx = 0, cypy = 0;
    for (unsigned i = t; i < np; i += nt) {
        const float dx = X[o + i] - mx, dy = Y[o + i] - my;
        const float dpx = PX[o + i] - mpx, dpy = PY[o + i] - mpy, dg = G[o + i] - mg;
        cx += dx * dx; cy += dy * dy; cpx += dpx * dpx; cpy += dpy * dpy;
        cg += dg * dg;
        cxpx += dx * dpx; cypy += dy * dpy;
    }
    cx = blockSum(cx, sh, t);   cy = blockSum(cy, sh, t);
    cpx = blockSum(cpx, sh, t); cpy = blockSum(cpy, sh, t); cg = blockSum(cg, sh, t);
    cxpx = blockSum(cxpx, sh, t); cypy = blockSum(cypy, sh, t);

    float *r = out + static_cast<size_t>(slice) * kBeamStride;
    if (t == 0) {
        r[0] = mx; r[1] = my; r[2] = mpx; r[3] = mpy; r[4] = mg;
        r[5] = cx * n; r[6] = cy * n; r[7] = cpx * n; r[8] = cpy * n; r[9] = cg * n;
        r[10] = cxpx * n; r[11] = cypy * n;
#pragma unroll
        for (int h = 0; h < kMaxBunchHarm; h++) {
            if (static_cast<unsigned>(h) < P.nharm) {
                r[12 + 2 * h] = b[h].x * n;
                r[13 + 2 * h] = b[h].y * n;
            }
        }
    }

    if (P.doAux == 0u) { return; }
    float xmn = 1e30f, xmx = -1e30f, ymn = 1e30f, ymx = -1e30f;
    float pxmn = 1e30f, pxmx = -1e30f, pymn = 1e30f, pymx = -1e30f;
    float gmn = 1e30f, gmx = -1e30f;
    for (unsigned i = t; i < np; i += nt) {
        const float vx = X[o + i], vy = Y[o + i], vpx = PX[o + i], vpy = PY[o + i],
                    vg = G[o + i];
        xmn = fminf(xmn, vx); xmx = fmaxf(xmx, vx);
        ymn = fminf(ymn, vy); ymx = fmaxf(ymx, vy);
        pxmn = fminf(pxmn, vpx); pxmx = fmaxf(pxmx, vpx);
        pymn = fminf(pymn, vpy); pymx = fmaxf(pymx, vpy);
        gmn = fminf(gmn, vg); gmx = fmaxf(gmx, vg);
    }
    xmn = blockMin(xmn, sh, t); xmx = blockMax(xmx, sh, t);
    ymn = blockMin(ymn, sh, t); ymx = blockMax(ymx, sh, t);
    pxmn = blockMin(pxmn, sh, t); pxmx = blockMax(pxmx, sh, t);
    pymn = blockMin(pymn, sh, t); pymx = blockMax(pymx, sh, t);
    gmn = blockMin(gmn, sh, t); gmx = blockMax(gmx, sh, t);
    if (t == 0) {
        // Slot 28 and up, clear of the bunching factors, which reach slot 27 at
        // the eight harmonics the reduction supports.
        r[28] = xmn; r[29] = xmx; r[30] = ymn; r[31] = ymx;
        r[32] = pxmn; r[33] = pxmx; r[34] = pymn; r[35] = pymx;
        r[36] = gmn; r[37] = gmx;
    }
}

struct FMPar {
    unsigned ngrid, isfar;
    float shift, wscale;
};

// Intensity-weighted transverse moments of one slice. With isfar != 0 the input
// is the transform of the slice and the cell index is FFT-shifted, matching the
// far-field branch of DiagField::getValues.
//
// wscale exists because the transform is unnormalised: |FFT|^2 is up to ngrid^4
// times the cell intensity, and dx^2*|FFT|^2 then overflows FP32 outright at
// ngrid = 256. Every quantity derived from these sums is a ratio, so scaling
// them all by a constant is free.
//
// The near-field pass also copies out the on-axis cell. On a unified-memory
// device the host simply reads it out of the resident field; here that would be
// one strided PCIe read per slice, so the kernel that is already touching the
// slice writes it into the reduction output instead.
__global__ void kFieldMoments(const float2 *__restrict__ F, float *__restrict__ out,
                              FMPar P)
{
    __shared__ float sh[kNWarp];
    const unsigned t = threadIdx.x, nt = blockDim.x, slice = blockIdx.x;
    const unsigned ng = P.ngrid, nn = ng * ng, hng = (ng + 1u) / 2u;
    const float2 *f = F + static_cast<size_t>(slice) * nn;

    float p = 0, sx = 0, sy = 0, fr = 0, fi = 0;
    for (unsigned i = t; i < nn; i += nt) {
        const unsigned ix = i % ng, iy = i / ng;
        const unsigned j =
            (P.isfar != 0u) ? (((iy + hng) % ng) * ng + ((ix + hng) % ng)) : i;
        const float2 c = f[j] * P.wscale;
        const float w = c.x * c.x + c.y * c.y;
        const float dx = static_cast<float>(ix) + P.shift;
        const float dy = static_cast<float>(iy) + P.shift;
        p += w; sx += dx * w; sy += dy * w; fr += c.x; fi += c.y;
    }
    p = blockSum(p, sh, t); sx = blockSum(sx, sh, t); sy = blockSum(sy, sh, t);
    fr = blockSum(fr, sh, t); fi = blockSum(fi, sh, t);

    const float xc = (p > 0.0f) ? sx / p : 0.0f;
    const float yc = (p > 0.0f) ? sy / p : 0.0f;

    float cx = 0, cy = 0;
    for (unsigned i = t; i < nn; i += nt) {
        const unsigned ix = i % ng, iy = i / ng;
        const unsigned j =
            (P.isfar != 0u) ? (((iy + hng) % ng) * ng + ((ix + hng) % ng)) : i;
        const float2 c = f[j] * P.wscale;
        const float w = c.x * c.x + c.y * c.y;
        const float dx = static_cast<float>(ix) + P.shift - xc;
        const float dy = static_cast<float>(iy) + P.shift - yc;
        cx += dx * dx * w; cy += dy * dy * w;
    }
    cx = blockSum(cx, sh, t); cy = blockSum(cy, sh, t);

    if (t == 0) {
        float *r = out + static_cast<size_t>(slice) * kFieldStride +
                   (P.isfar != 0u ? 7u : 0u);
        r[0] = p; r[1] = sx; r[2] = sy; r[3] = cx; r[4] = cy;
        if (P.isfar == 0u) {
            r[5] = fr; r[6] = fi;
            // The cell on the axis, as DiagField::getValues picks it.
            const float2 m = f[static_cast<size_t>(ng / 2) * ng + (ng / 2)];
            r[12] = m.x; r[13] = m.y;
        }
    }
}

// ---------------------------------------------------------------------------
// short-range space charge
// ---------------------------------------------------------------------------
//
// EFieldSolver::shortRange, one block per slice. The solve is a set of
// azimuthal modes m and longitudinal modes l; for each pair the particles are
// binned in radius into a complex source term, a tridiagonal system is solved
// on the radial grid, and the result is gathered back onto the particles.
//
// Two things are host side. The radial grid spacing arrives per slice, because
// rmax grows over the slices in order and a slice must be solved on a grid that
// already holds the widest slice before it. The centroid and the largest radius
// come from kSCAnalyse below, whose only purpose is to give the host that
// growth without sending it the particles.

__global__ void kSCAnalyse(const float *__restrict__ X, const float *__restrict__ Y,
                           float *__restrict__ out, unsigned npart)
{
    __shared__ float sh[kNWarp];
    const unsigned t = threadIdx.x, nt = blockDim.x, slice = blockIdx.x;
    const size_t o = static_cast<size_t>(slice) * npart;

    float sx = 0, sy = 0;
    for (unsigned i = t; i < npart; i += nt) { sx += X[o + i]; sy += Y[o + i]; }
    sx = blockSum(sx, sh, t);
    sy = blockSum(sy, sh, t);
    const float xc = sx / static_cast<float>(npart);
    const float yc = sy / static_cast<float>(npart);

    float r2 = 0;
    for (unsigned i = t; i < npart; i += nt) {
        const float dx = X[o + i] - xc, dy = Y[o + i] - yc;
        r2 = fmaxf(r2, dx * dx + dy * dy);
    }
    r2 = blockMax(r2, sh, t);

    if (t == 0) {
        out[3 * slice + 0] = xc;
        out[3 * slice + 1] = yc;
        out[3 * slice + 2] = sqrtf(r2);
    }
}

// Radius bin and azimuth of every particle, against the centroid and spacing of
// its own slice. Held for the whole solve so that the atan2 is paid once rather
// than once per mode pair.
__global__ void kSCPrepare(const float *__restrict__ X, const float *__restrict__ Y,
                           const float *__restrict__ CEN, const float *__restrict__ DR,
                           float *__restrict__ PHI, unsigned *__restrict__ IDX,
                           unsigned npart, unsigned ng, size_t n)
{
    const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= n) { return; }
    const unsigned is = static_cast<unsigned>(gid / npart);
    const float dx = X[gid] - CEN[3 * is + 0];
    const float dy = Y[gid] - CEN[3 * is + 1];
    PHI[gid] = atan2f(dy, dx);
    // The grid is sized to hold the widest particle, so this cannot exceed the
    // last bin except by rounding at the very edge. The clamp is here because
    // the consequence on the device is a write outside shared memory rather
    // than a wrong number.
    IDX[gid] = min(static_cast<unsigned>(floorf(sqrtf(dx * dx + dy * dy) / DR[is])),
                   ng - 1u);
}

struct SCPar {
    float coef, econstScale;
    unsigned npart, ngrid, nz, nphi;
};

// The radial arrays live in shared memory, whose size is a run-time property of
// the deck, so the allocation is dynamic. Metal had to accumulate the source
// term through a compare-and-swap on the bit pattern because it has no
// threadgroup float atomic; here atomicAdd on a shared float is native, so the
// source is accumulated straight into csrc and the two scratch arrays Metal
// needed are gone.
__global__ void kSCSolve(const float *__restrict__ TH, const float *__restrict__ PHI,
                         const unsigned *__restrict__ IDX, const float *__restrict__ DR,
                         const float *__restrict__ CUR, float *__restrict__ EZ,
                         float *__restrict__ SCOUT, SCPar P)
{
    extern __shared__ char scRaw[];
    const unsigned ng = P.ngrid, np = P.npart;
    float2 *csrc = reinterpret_cast<float2 *>(scRaw);
    float2 *clow = csrc + ng;
    float2 *cmid = clow + ng;
    float2 *cupp = cmid + ng;
    float2 *celm = cupp + ng;
    float2 *cgam = celm + ng;
    float *vol = reinterpret_cast<float *>(cgam + ng);
    float *rlog = vol + ng;
    float *lmid = rlog + ng;
    float *ldig = lmid + ng;          // ng + 1 entries

    const unsigned t = threadIdx.x, nt = blockDim.x, slice = blockIdx.x;
    const size_t o = static_cast<size_t>(slice) * np;
    const float pi = 3.14159265358979323846f;
    const float dr = DR[slice];

    // constructLaplaceOperator, once per slice
    for (unsigned j = t; j < ng; j += nt) {
        vol[j] = (j == 0) ? pi * dr * dr : pi * dr * dr * static_cast<float>(2 * j + 1);
        ldig[j] = (j == 0) ? 0.0f : 2.0f * pi * static_cast<float>(j);
        rlog[j] = (j == 0) ? 0.5f
                           : logf(static_cast<float>(j + 1) / static_cast<float>(j));
    }
    if (t == 0) { ldig[ng] = 0.0f; }
    __syncthreads();

    const float econst = P.econstScale * CUR[slice] / static_cast<float>(np);

    for (unsigned i = t; i < np; i += nt) { EZ[o + i] = 0.0f; }

    for (int m = -static_cast<int>(P.nphi); m <= static_cast<int>(P.nphi); m++) {
        for (unsigned j = t; j < ng; j += nt) {
            lmid[j] = -ldig[j] - ldig[j + 1] -
                      2.0f * pi * static_cast<float>(m * m) * rlog[j];
        }
        __syncthreads();
        if (t == 0) { lmid[ng - 1] -= 2.0f * pi * static_cast<float>(ng); }
        __syncthreads();

        for (unsigned l = 1; l <= P.nz; l++) {
            for (unsigned j = t; j < ng; j += nt) { csrc[j] = make_float2(0.0f, 0.0f); }
            __syncthreads();

            // source term: sum of exp(-i(m*phi + l*theta)) over the particles
            // of each radial bin
            for (unsigned i = t; i < np; i += nt) {
                const float a = -(static_cast<float>(m) * PHI[o + i] +
                                  static_cast<float>(l) * TH[o + i]);
                float sa, ca;
                sincosf(a, &sa, &ca);
                const unsigned bin = IDX[o + i];
                atomicAdd(&csrc[bin].x, ca);
                atomicAdd(&csrc[bin].y, sa);
            }
            __syncthreads();

            for (unsigned j = t; j < ng; j += nt) {
                const float s = econst / static_cast<float>(l) / vol[j];
                const float2 raw = csrc[j];
                csrc[j] = make_float2(-raw.y * s, raw.x * s);   // multiply by i*s
                const float f = P.coef / static_cast<float>(l * l) / vol[j];
                clow[j] = make_float2(f * ldig[j], 0.0f);
                cmid[j] = make_float2(1.0f + f * lmid[j], 0.0f);
                cupp[j] = make_float2(f * ldig[j + 1], 0.0f);
            }
            __syncthreads();

            // Thomas algorithm. A recurrence of ng steps, so one thread runs it
            // while the rest wait; ng is small and the alternative costs more
            // than it saves.
            if (t == 0) {
                float2 bet = cmid[0];
                celm[0] = cdiv(csrc[0], bet);
                for (unsigned j = 1; j < ng; j++) {
                    cgam[j] = cdiv(cupp[j - 1], bet);
                    bet = cmid[j] - cmul(clow[j], cgam[j]);
                    celm[j] = cdiv(csrc[j] - cmul(clow[j], celm[j - 1]), bet);
                }
                for (int j = static_cast<int>(ng) - 2; j > -1; j--) {
                    celm[j] = celm[j] - cmul(cgam[j + 1], celm[j + 1]);
                }
                // The CPU writes this for every m at l == 1, so what survives is
                // the last one. Matching that matters: it is the SSCfield
                // diagnostic.
                if (l == 1 && m == static_cast<int>(P.nphi)) {
                    SCOUT[slice] = sqrtf(celm[0].x * celm[0].x + celm[0].y * celm[0].y);
                }
            }
            __syncthreads();

            for (unsigned i = t; i < np; i += nt) {
                const float a = static_cast<float>(m) * PHI[o + i] +
                                static_cast<float>(l) * TH[o + i];
                float sa, ca;
                sincosf(a, &sa, &ca);
                const float2 e = celm[IDX[o + i]];
                EZ[o + i] += 2.0f * (ca * e.x - sa * e.y);
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// device selection
// ---------------------------------------------------------------------------

// Genesis parallelises over slices with MPI, and each rank owns its own engine
// and its own slices, so several cards in a node are used by pointing a rank at
// each of them. The device has to come from the rank's position within its
// node, not from its rank in the job: with the global rank, every rank of a
// four-node job lands on device 0 of its node and three quarters of the
// hardware idles, which produces a correct answer at a quarter of the speed and
// is invisible unless someone looks.
//
// G4_CUDA_DEVICE overrides the choice for one process, which is what to use
// when the launcher already pins ranks to devices; CUDA_VISIBLE_DEVICES is
// handled by the runtime and needs nothing here, since it simply changes what
// this function can see.
struct DeviceChoice {
    int device {0};
    int localRank {0};
    int localSize {1};
    int visible {1};
    bool fromEnv {false};
};

DeviceChoice selectDevice(int visible)
{
    DeviceChoice c;
    c.visible = visible;

    int worldRank = 0;
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited != 0) { MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); }

    if (inited != 0 && !MPISingle) {
        // Collective, and safe here because every rank reaches init() together.
        // In MPISingle mode the ranks are running unrelated jobs and must not
        // meet in a collective, so that case falls through to the world rank
        // below, which is a local call.
        MPI_Comm node = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                            &node);
        MPI_Comm_rank(node, &c.localRank);
        MPI_Comm_size(node, &c.localSize);
        MPI_Comm_free(&node);
    } else {
        c.localRank = worldRank;
        c.localSize = 1;
    }

    c.device = (visible > 0) ? (c.localRank % visible) : 0;

    const char *env = std::getenv("G4_CUDA_DEVICE");
    if (env != nullptr && *env != '\0') {
        const int want = std::atoi(env);
        if (want >= 0 && want < visible) {
            c.device = want;
            c.fromEnv = true;
        }
    }
    return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// the engine
// ---------------------------------------------------------------------------

struct CudaEngine::Impl {
    int devId {-1};
    cudaDeviceProp prop {};
    cudaStream_t stream {nullptr};
    std::string desc {"none"};

    int nslice {0};   // time slices, all resident
    int npart {0};    // particles per slice, must be uniform
    double gref {0};  // reference energy for the stored gamma offsets

    // Beam, structure of arrays. gamma is stored as an offset from gref.
    float *dX {nullptr}, *dY {nullptr}, *dPX {nullptr}, *dPY {nullptr};
    float *dG {nullptr}, *dT {nullptr}, *dCurrent {nullptr};

    // One field buffer per harmonic, each nslice * ngrid^2 complex<float>,
    // stored in the same slice order as Field::field so that the ring-buffer
    // index Field::first keeps its meaning.
    std::vector<float2 *> dField;
    std::vector<int> ngrid;

    // FFT shape for this grid. lanes*regs == ngrid; rows and cols are how many
    // transforms share a block.
    int lanes {0}, regs {0}, rowsPerTG {0}, colsPerTG {0};
    FFTLaunch fft {};

    // Field solve scratch. The source is reused across harmonics because they
    // are propagated one at a time. expK depends on the harmonic (through xks)
    // and on the step length, so it is cached per harmonic and rebuilt when
    // delz changes.
    float2 *dSrc {nullptr};
    float2 *dW {nullptr};
    float *dEZ {nullptr};
    float *dELoss {nullptr};
    float *dISR {nullptr};

    // Beamlet size and count, for the incoherent radiation kick.
    int nbins {1}, nbeamlet {0};

    // Short-range space charge: the per-slice analysis, the per-particle
    // azimuth and radial bin it produces, and the field it leaves behind.
    float *dSCcen {nullptr}, *dSCdr {nullptr}, *dSCout {nullptr}, *dSCphi {nullptr};
    unsigned *dSCidx {nullptr};
    float *dEZP {nullptr};
    bool scOn {false};
    int scNgrid {2};
    size_t scShmem {0};

    // Source filter: real, shared by every slice, applied to the transformed
    // source term. When it is off the field solve takes the cheaper four-pass
    // route and this buffer is a placeholder.
    float *dFilt {nullptr};
    bool filterOn {false};
    std::vector<float2 *> dExpK;
    std::vector<double> delzCached;

    // Diagnostic reduction output, a fixed number of floats per slice.
    float *dBM {nullptr}, *dFM {nullptr};

    // Pinned host mirrors of everything the host touches every step. Pinned
    // rather than pageable so that the copies can be queued on the stream in
    // order with the kernels instead of synchronising the whole device.
    float *hEZ {nullptr}, *hELoss {nullptr}, *hISR {nullptr};
    float *hSCcen {nullptr}, *hSCdr {nullptr}, *hSCout {nullptr};
    float *hBM {nullptr}, *hFM {nullptr};

    // Staging for the bulk transfers, which happen at setup, at teardown and at
    // a dump. Sized for one whole beam array or one field slice, whichever is
    // larger, and reused: the conversion between the host's FP64 arrays and the
    // device's FP32 ones costs more than the transfer does, so there is nothing
    // to gain from a larger buffer.
    float *hStage {nullptr};
    size_t hStageFloats {0};
    int stageSlices {1};   // field slices the staging buffer holds at once

    size_t bytes {0};

    // Seconds the device spent executing. Each engine call that queues work is
    // bracketed by a pair of events and the elapsed time between them is
    // accumulated, so what is measured is the device timeline from the first
    // operation of a call to the last, and the host work between two calls --
    // the slippage, the sort, the diagnostics -- falls outside every span.
    //
    // Read the result as a saturation indicator rather than as a calibrated
    // fraction. Two things stop it being one: the event records cost device time
    // of their own, and the device timestamps and the host clock the tracking
    // loop is measured with need not agree to better than a few percent. A
    // saturated run reports between 100% and 106% here. The question the number
    // is for is 95% against 50%, not 95% against 100%.
    mutable double busy {0};
    mutable bool pending {false};

    // A pool of event pairs, one per span, drained at each sync. Six spans is
    // what a step opens; the pool is far larger so that a caller which never
    // reads anything back cannot run it dry, and running it dry only forces an
    // early drain in any case.
    struct Span {
        cudaEvent_t a {nullptr}, b {nullptr};
    };
    std::vector<Span> spans;
    mutable int spanUsed {0};
    mutable bool spanOpen {false};

    // The bulk transfers -- the upload before the loop, the download after it,
    // and the whole-state copies gpu_validate performs -- are device work but
    // they are not tracking work, and the figure is compared against the wall
    // clock of the tracking loop. Counting the initial upload alone was enough
    // to report a device 102% busy. The per-slice slippage transfers stay
    // counted, because those happen inside the loop and are part of what the
    // loop costs.
    mutable bool countBusy {true};

    struct NoBusy {
        const Impl *p;
        bool saved;
        explicit NoBusy(const Impl *i) : p(i), saved(i->countBusy)
        {
            p->sync();          // close the batch being measured before pausing
            p->countBusy = false;
        }
        ~NoBusy()
        {
            p->sync();
            p->countBusy = saved;
        }
    };

    ~Impl()
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        auto dfree = [](void *p) { if (p != nullptr) { cudaFree(p); } };
        auto hfree = [](void *p) { if (p != nullptr) { cudaFreeHost(p); } };
        dfree(dX); dfree(dY); dfree(dPX); dfree(dPY); dfree(dG); dfree(dT);
        dfree(dCurrent);
        for (float2 *f : dField) { dfree(f); }
        for (float2 *k : dExpK) { dfree(k); }
        dfree(dSrc); dfree(dW); dfree(dEZ); dfree(dELoss); dfree(dISR);
        dfree(dSCcen); dfree(dSCdr); dfree(dSCout); dfree(dSCphi); dfree(dSCidx);
        dfree(dEZP); dfree(dFilt); dfree(dBM); dfree(dFM);
        hfree(hEZ); hfree(hELoss); hfree(hISR); hfree(hSCcen); hfree(hSCdr);
        hfree(hSCout); hfree(hBM); hfree(hFM); hfree(hStage);
        for (Span &sp : spans) {
            if (sp.a != nullptr) { cudaEventDestroy(sp.a); }
            if (sp.b != nullptr) { cudaEventDestroy(sp.b); }
        }
        if (stream != nullptr) { cudaStreamDestroy(stream); }
    }

    // Opens a span if one is not already open. Every launch and every queued
    // copy calls this, so the event pair always brackets real work.
    void begin() const
    {
        if (spanOpen) { return; }
        if (spanUsed >= static_cast<int>(spans.size())) { sync(); }
        cudaEventRecord(spans[spanUsed].a, stream);
        spanOpen = true;
        pending = true;
    }

    // Closes the current span. Called at the end of each engine call that
    // queues work, so that host work between two calls -- the slippage, the
    // sort, whatever the diagnostics do -- falls outside every span rather than
    // inside the one that happens to still be open. Measuring only to the next
    // drain instead reported the 500-slice deck as 102% busy at one rank.
    void endSpan() const
    {
        if (!spanOpen) { return; }
        cudaEventRecord(spans[spanUsed].b, stream);
        spanUsed++;
        spanOpen = false;
    }

    // Everything the host does to a buffer -- reading a reduction, uploading,
    // downloading, comparing -- goes through here first.
    void sync() const
    {
        if (!pending) { return; }
        endSpan();
        CU(cudaStreamSynchronize(stream));
        if (countBusy) {
            for (int i = 0; i < spanUsed; i++) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, spans[i].a, spans[i].b) == cudaSuccess) {
                    busy += ms * 1e-3;
                }
            }
        }
        spanUsed = 0;
        pending = false;
    }

    void toDevice(void *d, const void *h, size_t bytes) const
    {
        begin();
        CU(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream));
    }
    void toHost(void *h, const void *d, size_t bytes) const
    {
        begin();
        CU(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, stream));
    }

    size_t nthread() const { return static_cast<size_t>(nslice) * npart; }
    static int blocks(size_t n, int tpb)
    {
        return static_cast<int>((n + tpb - 1) / tpb);
    }
};

CudaEngine::CudaEngine() : p_(new Impl) {}

CudaEngine::~CudaEngine()
{
    delete p_;
}

int CudaEngine::maxBunchHarm() { return kMaxBunchHarm; }

double CudaEngine::gammaRef() const { return p_->gref; }
size_t CudaEngine::bytesResident() const { return p_->bytes; }
double CudaEngine::deviceSeconds() const { return p_->busy; }

bool CudaEngine::available(std::string &reason)
{
    reason.clear();
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        reason = std::string("no usable CUDA device: ") + cudaGetErrorString(e);
        // A machine with the runtime but no driver reports this, and the
        // message is worth passing on verbatim rather than replacing.
        cudaGetLastError();
        return false;
    }
    if (n < 1) {
        reason = "the CUDA runtime reports no devices";
        return false;
    }
    return true;
}

std::string CudaEngine::deviceName() const { return p_->desc; }

bool CudaEngine::init(Beam *beam, std::vector<Field *> *field, std::string &reason)
{
    reason.clear();

    // selectDevice() is collective over MPI_COMM_WORLD, so it has to be reached
    // by every rank that gets this far, before anything that can fail on one
    // rank and not another. A rank that never constructs an engine at all --
    // because create() found no runtime -- would still be missing from it, but
    // that is a node without a GPU in a job that asked for one, and there is no
    // sensible run to be had from it either way.
    int visible = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess) {
        visible = 0;
        cudaGetLastError();
    }
    const DeviceChoice choice = selectDevice(visible);
    if (visible < 1) {
        reason = "no CUDA device";
        return false;
    }
    p_->devId = choice.device;
    if (!CU(cudaSetDevice(p_->devId)) ||
        !CU(cudaGetDeviceProperties(&p_->prop, p_->devId))) {
        reason = g_cudaError;
        return false;
    }

    // A device whose architecture this binary has no code for fails at the
    // first launch with a message that says nothing useful, so say it here.
    {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, p_->devId);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, p_->devId);
        if (major < 7) {
            std::ostringstream os;
            os << "the CUDA backend needs compute capability 7.0 or newer; "
               << p_->prop.name << " is " << major << "." << minor;
            reason = os.str();
            return false;
        }
    }

    if (!CU(cudaStreamCreate(&p_->stream))) {
        reason = g_cudaError;
        return false;
    }
    p_->spans.resize(64);
    for (Impl::Span &sp : p_->spans) {
        if (!CU(cudaEventCreate(&sp.a)) || !CU(cudaEventCreate(&sp.b))) {
            reason = g_cudaError;
            return false;
        }
    }

    // The line the tracking loop prints. It carries the whole node's mapping,
    // because the failure this guards against -- every rank on device 0 -- is
    // invisible in a name alone.
    {
        std::ostringstream os;
        os << p_->prop.name;
        if (choice.visible > 1 || choice.localSize > 1 || choice.fromEnv) {
            os << " (rank " << choice.localRank << " of " << choice.localSize
               << " on this node -> cuda:" << p_->devId << " of " << choice.visible;
            if (choice.localSize > choice.visible) {
                os << ", " << ((choice.localSize + choice.visible - 1) / choice.visible)
                   << " ranks per device";
            }
            if (choice.fromEnv) { os << ", from G4_CUDA_DEVICE"; }
            os << ")";
        }
        p_->desc = os.str();
    }

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

    if (field->size() > static_cast<size_t>(kMaxHarm)) {
        std::ostringstream os;
        os << field->size() << " field harmonics; the kernels support at most "
           << kMaxHarm;
        reason = os.str();
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
        // The FFT decomposes ngrid into two register stages, which needs a
        // power of two between 64 and 1024.
        int lanes, regs, rows, cols;
        if (!pickFFTShape(f->ngrid, lanes, regs, rows, cols)) {
            std::ostringstream os;
            os << "ngrid = " << f->ngrid
               << " is not supported by the CUDA field solver, which handles "
                  "powers of two from 64 to 1024. Set ngrid = "
               << nearestSupported(f->ngrid)
               << " in &field. Genesis decks traditionally use an odd ngrid so that a "
                  "grid point sits exactly on axis, but nothing in the physics "
                  "requires that.";
            reason = os.str();
            return false;
        }
        p_->lanes = lanes;
        p_->regs = regs;
        p_->rowsPerTG = rows;
        p_->colsPerTG = cols;
        // The push kernel gathers every harmonic at one set of bilinear
        // weights, so all harmonics must share a grid.
        if (f->ngrid != field->at(0)->ngrid || f->dgrid != field->at(0)->dgrid ||
            f->gridmax != field->at(0)->gridmax) {
            reason = "field harmonics do not share the same transverse grid";
            return false;
        }
    }
    if (!buildLaunch(field->at(0)->ngrid, p_->fft)) {
        reason = "no transform instantiated for this grid size";
        return false;
    }
    // The blocking is chosen for 32 KB, which every supported architecture has
    // without opting in, but check rather than assume: a future part with less
    // shared memory would otherwise fail at the first launch.
    {
        const size_t need = static_cast<size_t>(std::max(p_->rowsPerTG, p_->colsPerTG)) *
                            field->at(0)->ngrid * sizeof(float2);
        if (need > p_->prop.sharedMemPerBlock) {
            std::ostringstream os;
            os << "the transform needs " << need / 1024
               << " KB of shared memory per block but " << p_->prop.name
               << " offers " << p_->prop.sharedMemPerBlock / 1024 << " KB";
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
    bool oom = false;
    auto alloc = [&](size_t n) -> void * {
        void *d = nullptr;
        if (cudaMalloc(&d, n) != cudaSuccess) {
            oom = true;
            cudaGetLastError();
            return nullptr;
        }
        p_->bytes += n;
        return d;
    };
    auto allocHost = [&](size_t n) -> float * {
        void *h = nullptr;
        if (cudaMallocHost(&h, n) != cudaSuccess) {
            oom = true;
            cudaGetLastError();
            return nullptr;
        }
        return static_cast<float *>(h);
    };

    p_->dX = static_cast<float *>(alloc(np * sizeof(float)));
    p_->dY = static_cast<float *>(alloc(np * sizeof(float)));
    p_->dPX = static_cast<float *>(alloc(np * sizeof(float)));
    p_->dPY = static_cast<float *>(alloc(np * sizeof(float)));
    p_->dG = static_cast<float *>(alloc(np * sizeof(float)));
    p_->dT = static_cast<float *>(alloc(np * sizeof(float)));
    p_->dCurrent = static_cast<float *>(alloc(static_cast<size_t>(p_->nslice) *
                                              sizeof(float)));

    p_->dField.clear();
    p_->ngrid.clear();
    for (size_t i = 0; i < field->size(); i++) {
        const int ng = field->at(i)->ngrid;
        p_->ngrid.push_back(ng);
        p_->dField.push_back(static_cast<float2 *>(
            alloc(static_cast<size_t>(p_->nslice) * ng * ng * sizeof(float2))));
    }

    // Short-range space charge. The radial arrays of the solve live in shared
    // memory, 64 bytes per grid point across the six complex and four real
    // arrays, so the grid is bounded by what one block can be given. That is a
    // property of the card rather than a constant, so it is read from the
    // device: 48 KB without opting in on everything since Volta, and 100 KB on
    // an L4 or a consumer Blackwell part, 164 KB on an A100.
    p_->scOn = beam->hasShortRangeSC();
    p_->scNgrid = p_->scOn ? beam->scGridSize() : 2;
    {
        const size_t perPoint = 6 * sizeof(float2) + 4 * sizeof(float);
        const size_t fixed = sizeof(float);   // the extra entry of ldig
        size_t cap = static_cast<size_t>(p_->prop.sharedMemPerBlockOptin);
        if (cap < p_->prop.sharedMemPerBlock) { cap = p_->prop.sharedMemPerBlock; }
        const int maxGrid = static_cast<int>((cap - fixed) / perPoint);
        p_->scShmem = perPoint * static_cast<size_t>(p_->scNgrid) + fixed;
        if (p_->scOn && (p_->scNgrid < 3 || p_->scNgrid > maxGrid)) {
            std::ostringstream os;
            os << "ngrid = " << p_->scNgrid << " in &efield, but the GPU space-charge "
                  "solve holds the radial arrays in shared memory and "
               << p_->prop.name << " has room for 3 to " << maxGrid;
            reason = os.str();
            return false;
        }
        if (p_->scOn && p_->scShmem > p_->prop.sharedMemPerBlock) {
            // Above 48 KB a kernel has to ask for the larger allocation.
            if (!CU(cudaFuncSetAttribute(reinterpret_cast<const void *>(kSCSolve),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(p_->scShmem)))) {
                reason = g_cudaError;
                return false;
            }
        }
    }
    {
        // dSCphi and dSCidx are written and read only by the space-charge
        // kernels, which do not launch unless the solver is on, so they can be
        // placeholders. dEZP is different: the push kernel reads it under a
        // runtime flag, and a predicated load into a one-element buffer is one
        // compiler decision away from a fault, so it is allocated in full
        // whether the solver is on or not. It is one float per particle against
        // a field of tens of complex numbers per particle.
        const size_t nsc = p_->scOn ? np : 1;
        p_->dSCcen = static_cast<float *>(
            alloc(static_cast<size_t>(p_->nslice) * 3 * sizeof(float)));
        p_->dSCdr = static_cast<float *>(
            alloc(static_cast<size_t>(p_->nslice) * sizeof(float)));
        p_->dSCout = static_cast<float *>(
            alloc(static_cast<size_t>(p_->nslice) * sizeof(float)));
        p_->dSCphi = static_cast<float *>(alloc(nsc * sizeof(float)));
        p_->dSCidx = static_cast<unsigned *>(alloc(nsc * sizeof(unsigned)));
        p_->dEZP = static_cast<float *>(alloc(np * sizeof(float)));
    }

    // Field solve scratch. One source buffer is enough because the harmonics
    // are propagated one after another.
    const int ng = p_->ngrid[0];
    const size_t nn = static_cast<size_t>(ng) * ng;
    p_->dSrc = static_cast<float2 *>(
        alloc(static_cast<size_t>(p_->nslice) * nn * sizeof(float2)));
    p_->dW = static_cast<float2 *>(alloc(static_cast<size_t>(ng) * sizeof(float2)));
    p_->dEZ = static_cast<float *>(alloc(static_cast<size_t>(p_->nslice) * sizeof(float)));
    p_->dELoss = static_cast<float *>(
        alloc(static_cast<size_t>(p_->nslice) * sizeof(float)));
    // The incoherent kick is one value per beamlet, not per slice, because that
    // is the granularity Incoherent::apply works at.
    p_->nbins = (beam->one4one || beam->nbins < 1) ? 1 : beam->nbins;
    p_->nbeamlet = (p_->npart + p_->nbins - 1) / p_->nbins;
    const size_t nisr = static_cast<size_t>(p_->nslice) * p_->nbeamlet;
    p_->dISR = static_cast<float *>(alloc(nisr * sizeof(float)));
    p_->dExpK.clear();
    p_->delzCached.assign(field->size(), -1.0);
    for (size_t i = 0; i < field->size(); i++) {
        p_->dExpK.push_back(static_cast<float2 *>(alloc(nn * sizeof(float2))));
    }

    p_->dBM = static_cast<float *>(
        alloc(static_cast<size_t>(p_->nslice) * kBeamStride * sizeof(float)));
    p_->dFM = static_cast<float *>(
        alloc(static_cast<size_t>(p_->nslice) * kFieldStride * sizeof(float)));

    p_->dFilt = static_cast<float *>(alloc(nn * sizeof(float)));

    // Pinned mirrors of the small per-step arrays, and the staging buffer for
    // the bulk transfers.
    p_->hEZ = allocHost(static_cast<size_t>(p_->nslice) * sizeof(float));
    p_->hELoss = allocHost(static_cast<size_t>(p_->nslice) * sizeof(float));
    p_->hISR = allocHost(nisr * sizeof(float));
    p_->hSCcen = allocHost(static_cast<size_t>(p_->nslice) * 3 * sizeof(float));
    p_->hSCdr = allocHost(static_cast<size_t>(p_->nslice) * sizeof(float));
    p_->hSCout = allocHost(static_cast<size_t>(p_->nslice) * sizeof(float));
    p_->hBM = allocHost(static_cast<size_t>(p_->nslice) * kBeamStride * sizeof(float));
    p_->hFM = allocHost(static_cast<size_t>(p_->nslice) * kFieldStride * sizeof(float));
    // Staging for the bulk transfers. Sized to hold one whole beam array, or a
    // run of field slices about 32 MB long, whichever is larger. The chunk
    // matters because gpu_validate uploads the entire state at every step: one
    // slice at a time would be a synchronisation per slice, which on a
    // 500-slice deck is 500 of them where sixteen will do.
    p_->stageSlices = static_cast<int>((32u << 20) / (nn * sizeof(float2)));
    if (p_->stageSlices < 1) { p_->stageSlices = 1; }
    if (p_->stageSlices > p_->nslice) { p_->stageSlices = p_->nslice; }
    p_->hStageFloats =
        std::max(np, static_cast<size_t>(p_->stageSlices) * nn * 2);
    p_->hStage = allocHost(p_->hStageFloats * sizeof(float));

    if (oom) {
        std::ostringstream os;
        size_t freeB = 0, totalB = 0;
        cudaMemGetInfo(&freeB, &totalB);
        os << "the resident buffers do not fit: " << p_->bytes / (1024 * 1024)
           << " MB wanted, " << freeB / (1024 * 1024) << " MB free of "
           << totalB / (1024 * 1024) << " MB on " << p_->prop.name
           << ". Spread the run over more MPI ranks, and more cards if there are "
              "any, or reduce ngrid or the particle count";
        reason = os.str();
        return false;
    }

    // Source filter, tabulated exactly as FieldSolverFFT::init does, from the
    // values the solver settled on rather than the ones the deck asked for: an
    // unphysical width or centre turns the filter off there rather than being
    // used. The filter is real, shared by every slice, and depends only on the
    // transverse wavenumber, so it is one array of ngrid^2 floats.
    {
        double xcf = 1, ycf = 1, sigf = 1;
        p_->filterOn = field->at(0)->getSourceFilter(xcf, ycf, sigf);
        for (size_t i = 1; i < field->size(); i++) {
            double a, b, c;
            if (field->at(i)->getSourceFilter(a, b, c) != p_->filterOn) {
                reason = "the source filter is on for some field harmonics and "
                         "off for others, which the GPU field solve cannot do";
                return false;
            }
        }
        std::vector<float> fl(nn, 0.0f);
        if (p_->filterOn) {
            const double shift = -0.5 * (ng - 1);
            for (int iy = 0; iy < ng; iy++) {
                const double y = (iy + shift) / static_cast<double>(ng) / ycf;
                for (int ix = 0; ix < ng; ix++) {
                    const double x = (ix + shift) / static_cast<double>(ng) / xcf;
                    const int ii =
                        ((iy + (ng + 1) / 2) % ng) * ng + ((ix + (ng + 1) / 2) % ng);
                    const double r = (sqrt(x * x + y * y) - 1) / sigf;
                    fl[ii] = static_cast<float>(1.0 / (1.0 + exp(r)));
                }
            }
        }
        if (!CU(cudaMemcpy(p_->dFilt, fl.data(), nn * sizeof(float),
                           cudaMemcpyHostToDevice))) {
            reason = g_cudaError;
            return false;
        }
    }

    {
        std::vector<float2> W(ng);
        for (int m = 0; m < ng; m++) {
            const double a = -2.0 * M_PI * m / ng;
            W[m] = make_float2(static_cast<float>(cos(a)), static_cast<float>(sin(a)));
        }
        if (!CU(cudaMemcpy(p_->dW, W.data(), static_cast<size_t>(ng) * sizeof(float2),
                           cudaMemcpyHostToDevice))) {
            reason = g_cudaError;
            return false;
        }
    }

    if (g_cudaFailed) {
        reason = g_cudaError;
        return false;
    }
    return true;
}

void CudaEngine::fieldStep(Undulator *und, std::vector<Field *> *field, double delz)
{
    cudaSetDevice(p_->devId);
    const int ns = p_->nslice;
    const int istep = und->getStep();
    const float fwd = 1.0f, inv = -1.0f, one = 1.0f;
    cudaStream_t s = p_->stream;

    for (size_t ih = 0; ih < p_->dField.size(); ih++) {
        Field *f = field->at(ih);
        const int ng = p_->ngrid[ih];
        const size_t nn = static_cast<size_t>(ng) * ng;
        const float nrm = 1.0f / static_cast<float>(nn);

        // exp(K2*delz) only changes when the step length does, and it is built
        // on the host, so anything already queued has to be out of the way.
        if (p_->delzCached[ih] != delz) {
            p_->sync();
            std::vector<float2> K(nn);
            const double dk = 4.0 * asin(1.0) / (ng * f->dgrid);
            const double shift = -0.5 * (ng - 1);
            for (int iy = 0; iy < ng; iy++) {
                const double dy = iy + shift;
                for (int ix = 0; ix < ng; ix++) {
                    const double dx = ix + shift;
                    const int ii =
                        ((iy + (ng + 1) / 2) % ng) * ng + ((ix + (ng + 1) / 2) % ng);
                    const std::complex<double> v = std::exp(
                        std::complex<double>(0, -(dx * dx + dy * dy) * dk * dk / 2.0 /
                                                    f->xks) *
                        delz);
                    K[ii] = make_float2(static_cast<float>(v.real()),
                                        static_cast<float>(v.imag()));
                }
            }
            CU(cudaMemcpy(p_->dExpK[ih], K.data(), nn * sizeof(float2),
                          cudaMemcpyHostToDevice));
            p_->delzCached[ih] = delz;
        }

        // Zero the source, then deposit if this harmonic couples. Even
        // harmonics have fc == 0 and are skipped, as on the CPU.
        p_->begin();
        CU(cudaMemsetAsync(p_->dSrc, 0,
                           static_cast<size_t>(ns) * nn * sizeof(float2), s));

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
            P.ngrid = static_cast<unsigned>(ng);
            P.npart = static_cast<unsigned>(p_->npart);
            P.nslice = static_cast<unsigned>(ns);
            P.harm = static_cast<unsigned>(harm);
            P.first = static_cast<unsigned>(f->first);

            const size_t n = p_->nthread();
            kDeposit<<<Impl::blocks(n, 256), 256, 0, s>>>(
                reinterpret_cast<float *>(p_->dSrc), p_->dX, p_->dY, p_->dG, p_->dT,
                p_->dCurrent, P, n);
        }

        // The field's own forward transform, which both paths need.
        p_->fft.rows(s, ns, p_->dField[ih], p_->dW, fwd);
        p_->fft.cols(s, ns, p_->dField[ih], p_->dW, fwd, one);

        if (p_->filterOn) {
            // Filtered: the source is shaped in Fourier space, so it needs its
            // own forward transform and has to be combined there. Six passes
            // rather than four.
            //
            //   field = IFFT(FFT(field)*expK + 2*filter*FFT(src))/ngrid^2
            p_->fft.rows(s, ns, p_->dSrc, p_->dW, fwd);
            p_->fft.colsFilt(s, ns, p_->dSrc, p_->dW, fwd, p_->dFilt);
            p_->fft.rowsMulAdd(s, ns, p_->dField[ih], p_->dW, inv, p_->dExpK[ih],
                               p_->dSrc);
            p_->fft.cols(s, ns, p_->dField[ih], p_->dW, inv, nrm);
        } else {
            // Unfiltered: the source is untouched in Fourier space, and the
            // transform is linear, so it can simply be added after the back
            // transform. Four passes.
            //
            //   field = IFFT(FFT(field)*expK)/ngrid^2 + 2*src
            p_->fft.rowsMul(s, ns, p_->dField[ih], p_->dW, inv, p_->dExpK[ih]);
            p_->fft.colsAdd(s, ns, p_->dField[ih], p_->dW, inv, nrm, p_->dSrc);
        }
    }
    p_->endSpan();
}

bool CudaEngine::beamStep(Beam *beam, Undulator *und, std::vector<Field *> *field,
                          double delz, std::string &reason)
{
    // Everything Beam::track does that is not yet on the GPU has to be
    // inactive, otherwise the answer would silently differ from the CPU.
    double angle, lb, ld, lt, cx, cy;
    und->getChicaneParameters(&angle, &lb, &ld, &lt);
    und->getCorrectorParameters(&cx, &cy);
    if (beam->gpuUnsupportedPhysics(reason)) {
        return false;
    }

    cudaSetDevice(p_->devId);
    cudaStream_t s = p_->stream;
    const int istep = und->getStep();
    const int ns = p_->nslice;
    const size_t nthread = p_->nthread();
    const int nblk = Impl::blocks(nthread, 256);

    // Transverse optics, transcribed from TrackBeam::track.
    double aw, dax, day, ku, kx, ky, qf, dqx, dqy;
    und->getUndulatorParameters(&aw, &dax, &day, &ku, &kx, &ky);
    und->getQuadrupoleParameters(&qf, &dqx, &dqy);
    const double gamma0 = und->getGammaRef();
    const double betpar0 = sqrt(1 - (1 + aw * aw) / gamma0 / gamma0);
    const double qquad = qf * gamma0;
    const double qnatx = kx * aw * aw / gamma0 / betpar0;
    const double qnaty = ky * aw * aw / gamma0 / betpar0;

    TrkPar T;
    const double qx = qquad + qnatx;
    const double qy = -qquad + qnaty;
    double xoff = qquad * dqx + qnatx * dax;
    double yoff = -qquad * dqy + qnaty * day;
    T.mx = (qx == 0) ? 0 : (qx > 0 ? 1 : 2);
    T.my = (qy == 0) ? 0 : (qy > 0 ? 1 : 2);
    if (qx != 0) { xoff /= qx; }
    if (qy != 0) { yoff /= qy; }
    T.aw = static_cast<float>(aw);
    T.qx = static_cast<float>(qx);
    T.qy = static_cast<float>(qy);
    T.xoff = static_cast<float>(xoff);
    T.yoff = static_cast<float>(yoff);
    T.gref = static_cast<float>(p_->gref);
    T.delz = static_cast<float>(0.5 * delz);

    // The chicane map, built by the same code the CPU path uses so that the two
    // cannot drift apart. It rides the opening half step, which is where
    // TrackBeam::track calls applyChicane.
    const unsigned chicMap = (angle != 0) ? 1u : 0u;
    T.doMap = 0;
    for (int i = 0; i < 8; i++) { T.cm[i] = 0; }
    if (angle != 0) {
        double m[4][4];
        TrackBeam::chicaneMatrix(angle, lb, ld, lt, m);
        T.cm[0] = static_cast<float>(m[0][0]);
        T.cm[1] = static_cast<float>(m[0][1]);
        T.cm[2] = static_cast<float>(m[1][0]);
        T.cm[3] = static_cast<float>(m[1][1]);
        T.cm[4] = static_cast<float>(m[2][2]);
        T.cm[5] = static_cast<float>(m[2][3]);
        T.cm[6] = static_cast<float>(m[3][2]);
        T.cm[7] = static_cast<float>(m[3][3]);
    }

    // Longitudinal push. Note that the undulator parameters used here are the
    // raw lattice values, not the ones zeroed outside an undulator that
    // TrackBeam uses -- this mirrors BeamSolver::advance.
    PushPar P;
    P.delz = static_cast<float>(delz);
    P.aw = static_cast<float>(und->getaw());
    P.gref = static_cast<float>(p_->gref);
    P.autophase = static_cast<float>(und->autophase());
    P.ax = static_cast<float>(und->ax[istep]);
    P.ay = static_cast<float>(und->ay[istep]);
    P.kx = static_cast<float>(und->kx[istep]);
    P.ky = static_cast<float>(und->ky[istep]);
    P.gradx = static_cast<float>(und->gradx[istep]);
    P.grady = static_cast<float>(und->grady[istep]);
    P.gridmax = static_cast<float>(field->at(0)->gridmax);
    P.dgrid = static_cast<float>(field->at(0)->dgrid);
    P.ngrid = static_cast<unsigned>(p_->ngrid[0]);
    P.npart = static_cast<unsigned>(p_->npart);
    P.nslice = static_cast<unsigned>(ns);
    P.nfld = static_cast<unsigned>(field->size());

    double xks = 1;
    for (size_t i = 0; i < field->size(); i++) {
        Field *f = field->at(i);
        const int harm = f->getHarm();
        xks = f->xks / static_cast<double>(harm);
        P.rtmp[i] = static_cast<float>(und->fc(harm) / f->xks);
        P.rharm[i] = static_cast<float>(harm);
        P.first[i] = static_cast<unsigned>(f->first);
        P.F[i] = p_->dField[i];
    }
    for (size_t i = field->size(); i < static_cast<size_t>(kMaxHarm); i++) {
        P.rtmp[i] = 0;
        P.rharm[i] = 0;
        P.first[i] = 0;
        P.F[i] = p_->dField[0];
    }
    P.xks = static_cast<float>(xks);
    double xku = und->getku();
    if (xku == 0) {
        // In a drift a particle at the reference energy stays in phase.
        xku = xks * 0.5 / gamma0 / gamma0;
    }
    P.xku = static_cast<float>(xku);

    // The pinned arrays below are written by the host and read by copies queued
    // in this step, so anything queued by the previous step -- and still only
    // queued, since nothing but sync() waits -- has to be out of the way first.
    // This is the one drain a step always performs.
    p_->sync();

    // Everything the host has to compute for this step is done first and the
    // copies are queued afterwards, rather than interleaved. The device is idle
    // until the first thing is queued, and the incoherent draws and the wake's
    // MPI_Allgather are the two slowest things on the host in a step, so
    // queueing across them would put that host time inside the interval the
    // busy figure measures and report a device that is waiting as one that is
    // working.

    // Long-range space charge, in units of the electron rest mass. Zero unless
    // the space-charge solver is switched on.
    for (int is = 0; is < ns; is++) {
        p_->hEZ[is] = (is < static_cast<int>(beam->longESC.size()))
                          ? static_cast<float>(-beam->longESC[is] / eev)
                          : 0.0f;
    }

    // Incoherent synchrotron radiation. One value per beamlet, drawn on the
    // host from the generator Genesis already uses and in the order the CPU
    // path consumes them, so that a GPU run reproduces a CPU run rather than
    // merely agreeing with it in distribution. Beam::track applies this between
    // the longitudinal push and the wakefield, which is where the launch below
    // sits.
    std::vector<double> isrKick;
    const bool haveISR = beam->computeIncoherentKick(und, delz, isrKick);
    const size_t nisr = static_cast<size_t>(ns) * p_->nbeamlet;
    if (haveISR) {
        for (size_t k = 0; k < nisr; k++) {
            p_->hISR[k] = (k < isrKick.size()) ? static_cast<float>(isrKick[k]) : 0.0f;
        }
    }

    // Wakefields. The loss profile is one number per slice and is built on the
    // host, which is where the current of every slice is reachable through an
    // MPI gather; the GPU only has to add it to the particles. Beam::track
    // applies this after the longitudinal push, so the launch below sits
    // between the push and the second transverse half step.
    const bool haveWake = beam->computeWakeLoss(und);
    if (haveWake) {
        for (int is = 0; is < ns; is++) {
            p_->hELoss[is] = (is < static_cast<int>(beam->eloss.size()))
                                 ? static_cast<float>(beam->eloss[is] * delz / eev)
                                 : 0.0f;
        }
    }

    // From here on the host only queues.
    p_->toDevice(p_->dEZ, p_->hEZ, static_cast<size_t>(ns) * sizeof(float));
    if (haveISR) { p_->toDevice(p_->dISR, p_->hISR, nisr * sizeof(float)); }
    if (haveWake) {
        p_->toDevice(p_->dELoss, p_->hELoss, static_cast<size_t>(ns) * sizeof(float));
    }

    auto encTrack = [&](double kickx, double kicky, bool map) {
        T.cpx = static_cast<float>(kickx);
        T.cpy = static_cast<float>(kicky);
        T.doMap = map ? chicMap : 0u;
        p_->begin();
        kTrackBeam<<<nblk, 256, 0, s>>>(p_->dX, p_->dY, p_->dPX, p_->dPY, p_->dG, T,
                                        nthread);
    };

    encTrack(0, 0, true);   // first half step, carrying any chicane

    // Short-range space charge, which BeamSolver::advance computes slice by
    // slice just before the push and which therefore has to see the particles
    // as the opening half step left them.
    //
    // The one part that cannot stay on the device is the radial grid. rmax
    // grows to hold the widest slice seen so far, sequentially over the slices
    // and persistently over the run, so slice k is solved on a grid that
    // already accounts for the slices before it. The analysis kernel therefore
    // reduces each slice to a centroid and a radius, the host replays that
    // growth, and the resulting spacing comes back. It costs one round trip per
    // step and three floats per slice, rather than the particles.
    unsigned useSR = 0;
    if (p_->scOn) {
        p_->begin();
        kSCAnalyse<<<ns, kRedThreads, 0, s>>>(p_->dX, p_->dY, p_->dSCcen,
                                              static_cast<unsigned>(p_->npart));
        p_->toHost(p_->hSCcen, p_->dSCcen, static_cast<size_t>(ns) * 3 * sizeof(float));
        p_->sync();

        std::vector<double> rbound(ns);
        for (int is = 0; is < ns; is++) { rbound[is] = p_->hSCcen[3 * is + 2]; }

        // Both scalars are taken the way BeamSolver::advance takes them.
        // gammaz2 uses the lattice aw, not the one TrackBeam zeroes outside an
        // undulator, and the wavenumber is the reference one from &setup, which
        // is what EFieldSolver::init was given.
        SCPlan plan;
        const double awlat = und->getaw();
        const double gz2 = gamma0 * gamma0 / (1 + awlat * awlat);
        if (beam->planShortRangeSC(rbound, gz2, plan)) {
            for (int is = 0; is < ns; is++) {
                p_->hSCdr[is] = static_cast<float>(plan.dr[is]);
            }
            p_->toDevice(p_->dSCdr, p_->hSCdr, static_cast<size_t>(ns) * sizeof(float));

            SCPar S;
            S.coef = static_cast<float>(plan.coef);
            // vacimp/eev/ks, the rest of econst being per slice. ks is the
            // reference wavenumber, as EFieldSolver::init formed it.
            const double ksref = 4.0 * asin(1.0) / beam->reflength;
            S.econstScale = static_cast<float>(vacimp / eev / ksref);
            S.npart = static_cast<unsigned>(p_->npart);
            S.ngrid = static_cast<unsigned>(plan.ngrid);
            S.nz = static_cast<unsigned>(plan.nz);
            S.nphi = static_cast<unsigned>(plan.nphi);

            p_->begin();
            kSCPrepare<<<nblk, 256, 0, s>>>(p_->dX, p_->dY, p_->dSCcen, p_->dSCdr,
                                            p_->dSCphi, p_->dSCidx,
                                            static_cast<unsigned>(p_->npart),
                                            static_cast<unsigned>(plan.ngrid), nthread);
            kSCSolve<<<ns, kRedThreads, p_->scShmem, s>>>(
                p_->dT, p_->dSCphi, p_->dSCidx, p_->dSCdr, p_->dCurrent, p_->dEZP,
                p_->dSCout, S);
            useSR = 1;
        }
    }
    P.useSR = useSR;

    p_->begin();
    kPushBeam<<<nblk, 256, 0, s>>>(p_->dG, p_->dT, p_->dX, p_->dY, p_->dPX, p_->dPY,
                                   p_->dEZ, p_->dEZP, P, nthread);

    // Incoherent radiation first, then the wake, which is the order Beam::track
    // applies them in.
    if (haveISR) {
        kApplyISR<<<nblk, 256, 0, s>>>(p_->dG, p_->dISR,
                                       static_cast<unsigned>(p_->npart),
                                       static_cast<unsigned>(p_->nbins), nthread);
    }
    if (haveWake) {
        kApplyELoss<<<nblk, 256, 0, s>>>(p_->dG, p_->dELoss,
                                         static_cast<unsigned>(p_->npart), nthread);
    }

    // The R56 phase shift, between the collective kick and the closing half
    // step, as in Beam::track.
    if (angle != 0) {
        const double gamma0lat = und->getGammaRef();
        double r56 = (4 * lb / sin(angle) * (1 - angle / tan(angle)) +
                      2 * ld * tan(angle) / cos(angle)) *
                     angle;
        r56 = r56 * 4 * asin(1.0) / beam->reflength / gamma0lat;
        kApplyR56<<<nblk, 256, 0, s>>>(p_->dT, p_->dG, static_cast<float>(r56),
                                       static_cast<float>(p_->gref - gamma0lat),
                                       nthread);
    }

    // TrackBeam::track applies the corrector on the closing half step only,
    // which is why the kick is attached here and not to the opening one.
    encTrack(cx * gamma0, cy * gamma0, false);   // second half step

    // The space-charge solve also produces the SSCfield diagnostic, one number
    // per slice, which the CPU path writes as it goes. It has to come back to
    // the host before the diagnostics of this step are assembled, so this is
    // the one place the engine drains itself in the middle of a run. Only decks
    // with short-range space charge pay it, and they are paying far more for
    // the solve itself.
    if (useSR != 0) {
        p_->toHost(p_->hSCout, p_->dSCout, static_cast<size_t>(ns) * sizeof(float));
        p_->sync();
        for (int is = 0; is < ns; is++) { beam->setSCField(is, p_->hSCout[is]); }
    }
    p_->endSpan();
    return true;
}

bool CudaEngine::beamMoments(int nharm, bool wantAux, BeamSliceMoments &out) const
{
    if (nharm < 1 || nharm > kMaxBunchHarm) { return false; }

    cudaSetDevice(p_->devId);
    BMPar P;
    P.gref = static_cast<float>(p_->gref);
    P.npart = static_cast<unsigned>(p_->npart);
    P.nharm = static_cast<unsigned>(nharm);
    P.doAux = wantAux ? 1u : 0u;

    const int ns = p_->nslice;
    p_->begin();
    kBeamMoments<<<ns, kRedThreads, 0, p_->stream>>>(p_->dX, p_->dY, p_->dPX, p_->dPY,
                                                     p_->dG, p_->dT, p_->dBM, P);
    p_->toHost(p_->hBM, p_->dBM,
               static_cast<size_t>(ns) * kBeamStride * sizeof(float));
    p_->sync();

    out.nslice = ns;
    out.nharm = nharm;
    out.hasAux = wantAux;
    auto fill = [ns](std::vector<double> &v) { v.assign(ns, 0.0); };
    fill(out.x1); fill(out.x2); fill(out.y1); fill(out.y2);
    fill(out.px1); fill(out.px2); fill(out.py1); fill(out.py2);
    fill(out.g1); fill(out.g2); fill(out.xpx); fill(out.ypy);
    out.bre.assign(static_cast<size_t>(ns) * nharm, 0.0);
    out.bim.assign(static_cast<size_t>(ns) * nharm, 0.0);
    if (wantAux) {
        fill(out.xmin); fill(out.xmax); fill(out.ymin); fill(out.ymax);
        fill(out.pxmin); fill(out.pxmax); fill(out.pymin); fill(out.pymax);
        fill(out.gmin); fill(out.gmax);
    }

    // Undo the centring in double. x2 = <(x-<x>)^2> + <x>^2 reproduces the raw
    // moment to double precision, while the GPU never had to subtract two
    // nearly equal single precision numbers.
    const float *r = p_->hBM;
    for (int is = 0; is < ns; is++) {
        const float *s = r + static_cast<size_t>(is) * kBeamStride;
        const double x1 = s[0], y1 = s[1], px1 = s[2], py1 = s[3];
        const double g1 = p_->gref + s[4];
        out.x1[is] = x1;   out.x2[is] = s[5] + x1 * x1;
        out.y1[is] = y1;   out.y2[is] = s[6] + y1 * y1;
        out.px1[is] = px1; out.px2[is] = s[7] + px1 * px1;
        out.py1[is] = py1; out.py2[is] = s[8] + py1 * py1;
        out.g1[is] = g1;   out.g2[is] = s[9] + g1 * g1;
        out.xpx[is] = s[10] + x1 * px1;
        out.ypy[is] = s[11] + y1 * py1;
        for (int h = 0; h < nharm; h++) {
            out.bre[static_cast<size_t>(is) * nharm + h] = s[12 + 2 * h];
            out.bim[static_cast<size_t>(is) * nharm + h] = s[13 + 2 * h];
        }
        if (wantAux) {
            out.xmin[is] = s[28]; out.xmax[is] = s[29];
            out.ymin[is] = s[30]; out.ymax[is] = s[31];
            out.pxmin[is] = s[32]; out.pxmax[is] = s[33];
            out.pymin[is] = s[34]; out.pymax[is] = s[35];
            out.gmin[is] = p_->gref + s[36];
            out.gmax[is] = p_->gref + s[37];
        }
    }
    return true;
}

bool CudaEngine::fieldMoments(int ih, bool wantFar, FieldSliceMoments &out) const
{
    if (ih < 0 || ih >= static_cast<int>(p_->dField.size())) { return false; }

    cudaSetDevice(p_->devId);
    const int ns = p_->nslice;
    const int ng = p_->ngrid[ih];
    cudaStream_t s = p_->stream;

    FMPar P;
    P.ngrid = static_cast<unsigned>(ng);
    P.isfar = 0;
    P.shift = static_cast<float>(-0.5 * (ng - 1));
    P.wscale = 1.0f;

    const float fwd = -1.0f, one = 1.0f;

    p_->begin();
    kFieldMoments<<<ns, kRedThreads, 0, s>>>(p_->dField[ih], p_->dFM, P);

    if (wantFar) {
        // The near field has to survive, so the transform is written elsewhere:
        // the row pass reads the field and writes dSrc, which is only live
        // inside fieldStep and so is free to borrow here. Copying the slice
        // first and transforming in place would cost a read and a write of
        // every point for nothing.
        p_->fft.rowsOut(s, ns, p_->dSrc, p_->dW, fwd, p_->dField[ih]);
        p_->fft.cols(s, ns, p_->dSrc, p_->dW, fwd, one);

        FMPar F = P;
        F.isfar = 1;
        F.wscale = 1.0f / static_cast<float>(ng);
        kFieldMoments<<<ns, kRedThreads, 0, s>>>(p_->dSrc, p_->dFM, F);
    }
    p_->toHost(p_->hFM, p_->dFM,
               static_cast<size_t>(ns) * kFieldStride * sizeof(float));
    p_->sync();

    out.nslice = ns;
    out.hasFar = wantFar;
    auto fill = [ns](std::vector<double> &v) { v.assign(ns, 0.0); };
    fill(out.power); fill(out.x1); fill(out.x2); fill(out.y1); fill(out.y2);
    fill(out.ffre); fill(out.ffim); fill(out.midre); fill(out.midim);
    if (wantFar) {
        fill(out.fpower); fill(out.fx1); fill(out.fx2); fill(out.fy1); fill(out.fy2);
    }

    const float *r = p_->hFM;
    auto uncentre = [](double c, double s1, double w) {
        return (w > 0.0) ? c + s1 * s1 / w : 0.0;
    };
    for (int is = 0; is < ns; is++) {
        const float *v = r + static_cast<size_t>(is) * kFieldStride;
        out.power[is] = v[0];
        out.x1[is] = v[1];  out.y1[is] = v[2];
        out.x2[is] = uncentre(v[3], v[1], v[0]);
        out.y2[is] = uncentre(v[4], v[2], v[0]);
        out.ffre[is] = v[5]; out.ffim[is] = v[6];
        if (wantFar) {
            out.fpower[is] = v[7];
            out.fx1[is] = v[8];  out.fy1[is] = v[9];
            out.fx2[is] = uncentre(v[10], v[8], v[7]);
            out.fy2[is] = uncentre(v[11], v[9], v[7]);
        }
        // The on-axis cell, which the near-field pass copied out for us.
        out.midre[is] = v[12];
        out.midim[is] = v[13];
    }
    return true;
}

// ---------------------------------------------------------------------------
// host transfers
// ---------------------------------------------------------------------------
//
// Everything from here on touches the device from the host, so each of these
// first drains whatever is queued. See Impl::sync().

// The six particle coordinates, in the order the arrays are held. The switch on
// this index is what keeps the conversion loop over the host's array-of-structs
// tight: one pass per coordinate, writing one contiguous device array.
enum BeamArray { kAX = 0, kAY, kAPX, kAPY, kAG, kATH };

void CudaEngine::uploadBeam(Beam *beam)
{
    cudaSetDevice(p_->devId);
    Impl::NoBusy quiet(p_);
    const int ns = p_->nslice, np = p_->npart;
    const size_t n = static_cast<size_t>(ns) * np;
    const double gref = p_->gref;
    float *st = p_->hStage;

    float *dst[6] = {p_->dX, p_->dY, p_->dPX, p_->dPY, p_->dG, p_->dT};
    for (int a = 0; a < 6; a++) {
        for (int is = 0; is < ns; is++) {
            const size_t o = static_cast<size_t>(is) * np;
            const Particle *src = beam->beam[is].data();
            switch (a) {
            case kAX:  for (int i = 0; i < np; i++) { st[o + i] = static_cast<float>(src[i].x); } break;
            case kAY:  for (int i = 0; i < np; i++) { st[o + i] = static_cast<float>(src[i].y); } break;
            case kAPX: for (int i = 0; i < np; i++) { st[o + i] = static_cast<float>(src[i].px); } break;
            case kAPY: for (int i = 0; i < np; i++) { st[o + i] = static_cast<float>(src[i].py); } break;
            case kAG:  for (int i = 0; i < np; i++) { st[o + i] = static_cast<float>(src[i].gamma - gref); } break;
            default:   for (int i = 0; i < np; i++) { st[o + i] = static_cast<float>(src[i].theta); } break;
            }
        }
        p_->toDevice(dst[a], st, n * sizeof(float));
        p_->sync();   // the staging buffer is reused by the next coordinate
    }

    for (int is = 0; is < ns; is++) {
        st[is] = static_cast<float>(beam->current[is]);
    }
    p_->toDevice(p_->dCurrent, st, static_cast<size_t>(ns) * sizeof(float));
    p_->sync();
}

// Field slices in runs of stageSlices, converted into the pinned buffer and
// sent in one copy. One slice at a time would work but costs a synchronisation
// per slice, and gpu_validate does this at every step.
void CudaEngine::upload(Beam *beam, std::vector<Field *> *field)
{
    this->uploadBeam(beam);

    cudaSetDevice(p_->devId);
    Impl::NoBusy quiet(p_);
    for (size_t i = 0; i < p_->dField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        float2 *st = reinterpret_cast<float2 *>(p_->hStage);
        for (int is0 = 0; is0 < p_->nslice; is0 += p_->stageSlices) {
            const int nb = std::min(p_->stageSlices, p_->nslice - is0);
            for (int b = 0; b < nb; b++) {
                const std::complex<double> *src = field->at(i)->field[is0 + b].data();
                float2 *d = st + static_cast<size_t>(b) * nn;
                for (size_t k = 0; k < nn; k++) {
                    d[k] = make_float2(static_cast<float>(src[k].real()),
                                       static_cast<float>(src[k].imag()));
                }
            }
            p_->toDevice(p_->dField[i] + static_cast<size_t>(is0) * nn, st,
                         static_cast<size_t>(nb) * nn * sizeof(float2));
            p_->sync();
        }
    }
}

void CudaEngine::downloadBeam(Beam *beam)
{
    cudaSetDevice(p_->devId);
    Impl::NoBusy quiet(p_);
    const int ns = p_->nslice, np = p_->npart;
    const size_t n = static_cast<size_t>(ns) * np;
    const double gref = p_->gref;
    float *st = p_->hStage;

    const float *src[6] = {p_->dX, p_->dY, p_->dPX, p_->dPY, p_->dG, p_->dT};
    for (int a = 0; a < 6; a++) {
        p_->toHost(st, src[a], n * sizeof(float));
        p_->sync();
        for (int is = 0; is < ns; is++) {
            const size_t o = static_cast<size_t>(is) * np;
            Particle *dst = beam->beam[is].data();
            switch (a) {
            case kAX:  for (int i = 0; i < np; i++) { dst[i].x = st[o + i]; } break;
            case kAY:  for (int i = 0; i < np; i++) { dst[i].y = st[o + i]; } break;
            case kAPX: for (int i = 0; i < np; i++) { dst[i].px = st[o + i]; } break;
            case kAPY: for (int i = 0; i < np; i++) { dst[i].py = st[o + i]; } break;
            case kAG:  for (int i = 0; i < np; i++) { dst[i].gamma = gref + static_cast<double>(st[o + i]); } break;
            default:   for (int i = 0; i < np; i++) { dst[i].theta = st[o + i]; } break;
            }
        }
    }
}

void CudaEngine::downloadField(std::vector<Field *> *field)
{
    cudaSetDevice(p_->devId);
    Impl::NoBusy quiet(p_);
    for (size_t i = 0; i < p_->dField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        float2 *st = reinterpret_cast<float2 *>(p_->hStage);
        for (int is0 = 0; is0 < p_->nslice; is0 += p_->stageSlices) {
            const int nb = std::min(p_->stageSlices, p_->nslice - is0);
            p_->toHost(st, p_->dField[i] + static_cast<size_t>(is0) * nn,
                       static_cast<size_t>(nb) * nn * sizeof(float2));
            p_->sync();
            for (int b = 0; b < nb; b++) {
                std::complex<double> *dst = field->at(i)->field[is0 + b].data();
                const float2 *s = st + static_cast<size_t>(b) * nn;
                for (size_t k = 0; k < nn; k++) {
                    dst[k] = std::complex<double>(s[k].x, s[k].y);
                }
            }
        }
    }
}

void CudaEngine::download(Beam *beam, std::vector<Field *> *field)
{
    this->downloadBeam(beam);
    this->downloadField(field);
}

// Slippage moves exactly one slice per slip event, so the whole field does not
// have to make the trip. These two keep that one slice consistent while the
// rest of the grid stays resident. On a discrete card each is a PCIe transfer
// of ngrid^2 complex values -- 512 KB at ngrid = 256 -- and a synchronisation,
// which is why the tracking loop only calls them when the slippage actually
// moves a slice rather than once per step.
void CudaEngine::downloadFieldSlice(int ifld, int islice, Field *field)
{
    if (ifld < 0 || static_cast<size_t>(ifld) >= p_->dField.size()) { return; }
    if (islice < 0 || islice >= p_->nslice) { return; }
    cudaSetDevice(p_->devId);
    p_->sync();
    const size_t nn = static_cast<size_t>(p_->ngrid[ifld]) * p_->ngrid[ifld];
    float2 *st = reinterpret_cast<float2 *>(p_->hStage);
    p_->toHost(st, p_->dField[ifld] + static_cast<size_t>(islice) * nn,
               nn * sizeof(float2));
    p_->sync();
    std::complex<double> *dst = field->field[islice].data();
    for (size_t k = 0; k < nn; k++) {
        dst[k] = std::complex<double>(st[k].x, st[k].y);
    }
}

void CudaEngine::uploadFieldSlice(int ifld, int islice, const Field *field)
{
    if (ifld < 0 || static_cast<size_t>(ifld) >= p_->dField.size()) { return; }
    if (islice < 0 || islice >= p_->nslice) { return; }
    cudaSetDevice(p_->devId);
    p_->sync();
    const size_t nn = static_cast<size_t>(p_->ngrid[ifld]) * p_->ngrid[ifld];
    float2 *st = reinterpret_cast<float2 *>(p_->hStage);
    const std::complex<double> *src = field->field[islice].data();
    for (size_t k = 0; k < nn; k++) {
        st[k] = make_float2(static_cast<float>(src[k].real()),
                            static_cast<float>(src[k].imag()));
    }
    p_->toDevice(p_->dField[ifld] + static_cast<size_t>(islice) * nn, st,
                 nn * sizeof(float2));
    p_->sync();
}

CudaEngine::SyncError CudaEngine::compare(Beam *beam,
                                          std::vector<Field *> *field) const
{
    cudaSetDevice(p_->devId);
    Impl::NoBusy quiet(p_);
    SyncError e;
    const int ns = p_->nslice, np = p_->npart;
    const size_t n = static_cast<size_t>(ns) * np;
    const double gref = p_->gref;
    float *st = p_->hStage;

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
        if (scale[k] == 0) { scale[k] = 1; }
    }

    const float *src[6] = {p_->dX, p_->dY, p_->dPX, p_->dPY, p_->dG, p_->dT};
    for (int a = 0; a < 6; a++) {
        p_->toHost(st, src[a], n * sizeof(float));
        p_->sync();
        for (int is = 0; is < ns; is++) {
            const size_t o = static_cast<size_t>(is) * np;
            for (int ip = 0; ip < np; ip++) {
                const Particle &h = beam->beam[is][ip];
                double hv = 0;
                switch (a) {
                case 0: hv = h.x; break;
                case 1: hv = h.y; break;
                case 2: hv = h.px; break;
                case 3: hv = h.py; break;
                case 4: hv = h.gamma - gref; break;
                default: hv = h.theta; break;
                }
                e.beam = std::max(e.beam, std::abs(hv - st[o + ip]) / scale[a]);
            }
        }
    }

    for (size_t i = 0; i < p_->dField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        float2 *fst = reinterpret_cast<float2 *>(st);
        double fmax = 0;
        for (int is = 0; is < ns; is++) {
            const std::complex<double> *h = field->at(i)->field[is].data();
            for (size_t k = 0; k < nn; k++) { fmax = std::max(fmax, std::abs(h[k])); }
        }
        if (fmax == 0) { fmax = 1; }
        for (int is0 = 0; is0 < ns; is0 += p_->stageSlices) {
            const int nb = std::min(p_->stageSlices, ns - is0);
            p_->toHost(fst, p_->dField[i] + static_cast<size_t>(is0) * nn,
                       static_cast<size_t>(nb) * nn * sizeof(float2));
            p_->sync();
            for (int b = 0; b < nb; b++) {
                const std::complex<double> *h = field->at(i)->field[is0 + b].data();
                const float2 *s = fst + static_cast<size_t>(b) * nn;
                for (size_t k = 0; k < nn; k++) {
                    e.field = std::max(
                        e.field,
                        std::abs(h[k] - std::complex<double>(s[k].x, s[k].y)) / fmax);
                }
            }
        }
    }
    return e;
}
