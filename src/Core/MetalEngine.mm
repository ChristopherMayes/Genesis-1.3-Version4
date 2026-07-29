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

// The transform is a four-step Cooley-Tukey decomposition N = REGS * LANES:
// LANES threads cooperate on one N-point FFT, each holding REGS complex values
// in registers, with a single threadgroup exchange between the two stages.
// That is 2.2x faster than a radix-2 shuffle ladder and also more accurate,
// because it is two stages rather than log2(N).
//
// The shape is chosen per grid size in init() and injected as preprocessor
// macros when the library is compiled, so each build of the shader is
// specialised to one ngrid. At ngrid = 256 this reproduces the original
// 16 x 16 radix-16 kernel exactly.
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

constant uint N = NG;

inline float2 cmul(float2 a, float2 b){ return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }

// Twiddle stride: the table holds W_N, so W_P^m lives at index (NG/P)*m.
#define WSTRIDE(P) ((uint)(NG)/(uint)(P))

inline void dft4(thread float2* a, float s){
    float2 t0=a[0]+a[2], t1=a[0]-a[2], t2=a[1]+a[3], d=a[1]-a[3];
    float2 t3 = float2(s*d.y, -s*d.x);
    a[0]=t0+t2; a[1]=t1+t3; a[2]=t0-t2; a[3]=t1-t3;
}
// The register DFTs below all leave their output permuted: for a P-point
// transform written as P = P1 x P2, slot P2*k1 + k2 holds frequency k1 + P1*k2.
// The caller folds that permutation into its store index, which avoids a
// P-element temporary and the dynamic indexing that comes with it.
//
//   dft8   8 = 2 x 4   slot j holds frequency (j>>2) + 2*(j&3)
//   dft16  16 = 4 x 4  slot j holds frequency (j>>2) + 4*(j&3)
//   dft32  32 = 4 x 8  slot j holds frequency (j>>3) + 4*(j&7)

inline void dft8(thread float2* a, const device float2* W, float s){
    for (uint n2=0;n2<4;n2++){
        float2 u=a[n2], v=a[4+n2];
        float2 w = W[(WSTRIDE(8)*n2) & (N-1u)]; w.y *= s;
        a[n2]   = u+v;
        a[4+n2] = cmul(u-v, w);
    }
    float2 t[4];
    for (uint k1=0;k1<2;k1++){
        for (uint n2=0;n2<4;n2++) t[n2]=a[4*k1+n2];
        dft4(t,s);
        for (uint k2=0;k2<4;k2++) a[4*k1+k2]=t[k2];
    }
}
// Natural-order 8-point transform, needed as the inner stage of dft32.
inline void dft8n(thread float2* a, const device float2* W, float s){
    dft8(a, W, s);
    float2 t[8];
    for (uint j=0;j<8;j++) t[j]=a[j];
    for (uint j=0;j<8;j++) a[(j>>2) + 2u*(j&3u)] = t[j];
}
inline void dft16(thread float2* a, const device float2* W, float s){
    float2 t[4];
    for (uint n2=0;n2<4;n2++){
        for (uint j=0;j<4;j++) t[j]=a[4*j+n2];
        dft4(t,s);
        for (uint k1=0;k1<4;k1++){
            float2 w = W[(WSTRIDE(16)*n2*k1) & (N-1u)]; w.y *= s;
            a[4*k1+n2] = cmul(t[k1], w);
        }
    }
    for (uint k1=0;k1<4;k1++){
        for (uint n2=0;n2<4;n2++) t[n2]=a[4*k1+n2];
        dft4(t,s);
        for (uint k2=0;k2<4;k2++) a[4*k1+k2]=t[k2];
    }
}
inline void dft32(thread float2* a, const device float2* W, float s){
    float2 t[8];
    for (uint n2=0;n2<8;n2++){
        for (uint j=0;j<4;j++) t[j]=a[8*j+n2];
        dft4(t,s);
        for (uint k1=0;k1<4;k1++){
            float2 w = W[(WSTRIDE(32)*n2*k1) & (N-1u)]; w.y *= s;
            a[8*k1+n2] = cmul(t[k1], w);
        }
    }
    for (uint k1=0;k1<4;k1++){
        for (uint n2=0;n2<8;n2++) t[n2]=a[8*k1+n2];
        dft8n(t, W, s);
        for (uint k2=0;k2<8;k2++) a[8*k1+k2]=t[k2];
    }
}

// Select the two stages. REGS is the first stage, done once per thread; LANES
// is the second, done CHUNK = REGS/LANES times per thread after the exchange.
#if REGS == 8
#  define DFT_REGS(a)  dft8(a, W, sgn)
#  define PERM_REGS(j) (((j)>>2) + 2u*((j)&3u))
#elif REGS == 16
#  define DFT_REGS(a)  dft16(a, W, sgn)
#  define PERM_REGS(j) (((j)>>2) + 4u*((j)&3u))
#elif REGS == 32
#  define DFT_REGS(a)  dft32(a, W, sgn)
#  define PERM_REGS(j) (((j)>>3) + 4u*((j)&7u))
#endif

#if LANES == 8
#  define DFT_LANES(a)  dft8(a, W, sgn)
#  define PERM_LANES(j) (((j)>>2) + 2u*((j)&3u))
#elif LANES == 16
#  define DFT_LANES(a)  dft16(a, W, sgn)
#  define PERM_LANES(j) (((j)>>2) + 4u*((j)&3u))
#elif LANES == 32
#  define DFT_LANES(a)  dft32(a, W, sgn)
#  define PERM_LANES(j) (((j)>>3) + 4u*((j)&7u))
#endif

// N-point FFT across LANES threads. On entry a[n1] = x[LANES*n1 + lane].
// On exit slot cc*LANES + j holds X[(lane + cc*LANES) + REGS*PERM_LANES(j)].
// The exchange index is XOR-swizzled to avoid bank conflicts without padding,
// which is what lets several transforms share one threadgroup allocation.
inline void fftN(thread float2* a, threadgroup float2* s,
                 const device float2* W, uint lane, float sgn){
    DFT_REGS(a);
    for (uint j=0;j<REGS;j++){
        uint k1 = PERM_REGS(j);
        float2 w = W[(lane*k1) & (N-1u)]; w.y *= sgn;
        s[k1*LANES + (lane ^ (k1 & (LANES-1u)))] = cmul(a[j], w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cc=0;cc<CHUNK;cc++)
        for (uint n2=0;n2<LANES;n2++)
            a[cc*LANES+n2] = s[(lane + cc*LANES)*LANES + (n2 ^ lane)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cc=0;cc<CHUNK;cc++) DFT_LANES(a + cc*LANES);
}
// Frequency held by output slot cc*LANES + j in the thread with this lane.
#define OUTK(cc,j) ((lane + (cc)*LANES) + REGS*PERM_LANES(j))

kernel void fft_rows(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                     constant float& sgn [[buffer(2)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[RF_ROWS*NG];
    uint lane = t % LANES, r = t / LANES;
    device float2* p = d + (ulong)tg.y*((ulong)N*N) + (ulong)(tg.x*RF_ROWS + r)*N;
    float2 a[REGS];
    for (uint n1=0;n1<REGS;n1++) a[n1] = p[LANES*n1 + lane];
    fftN(a, sh + r*NG, W, lane, sgn);
    for (uint cc=0;cc<CHUNK;cc++)
        for (uint j=0;j<LANES;j++) p[OUTK(cc,j)] = a[cc*LANES+j];
}
kernel void fft_rows_mul(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                         constant float& sgn [[buffer(2)]], const device float2* expK [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[RF_ROWS*NG];
    uint lane = t % LANES, r = t / LANES;
    ulong row = (ulong)(tg.x*RF_ROWS + r);
    device float2* p = d + (ulong)tg.y*((ulong)N*N) + row*N;
    const device float2* k = expK + row*N;
    float2 a[REGS];
    for (uint n1=0;n1<REGS;n1++) a[n1] = cmul(p[LANES*n1 + lane], k[LANES*n1 + lane]);
    fftN(a, sh + r*NG, W, lane, sgn);
    for (uint cc=0;cc<CHUNK;cc++)
        for (uint j=0;j<LANES;j++) p[OUTK(cc,j)] = a[cc*LANES+j];
}
kernel void fft_cols(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                     constant float& sgn [[buffer(2)]], constant float& scale [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[CC_COLS*NG];
    uint c = t % CC_COLS, lane = t / CC_COLS;
    device float2* b = d + (ulong)tg.y*((ulong)N*N) + (ulong)tg.x*CC_COLS + c;
    float2 a[REGS];
    for (uint n1=0;n1<REGS;n1++) a[n1] = b[(ulong)(LANES*n1+lane)*N];
    fftN(a, sh + c*NG, W, lane, sgn);
    for (uint cc=0;cc<CHUNK;cc++)
        for (uint j=0;j<LANES;j++) b[(ulong)OUTK(cc,j)*N] = a[cc*LANES+j]*scale;
}
kernel void fft_cols_add(device float2* d [[buffer(0)]], const device float2* W [[buffer(1)]],
                         constant float& sgn [[buffer(2)]], constant float& scale [[buffer(3)]],
                         const device float2* src [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]){
    threadgroup float2 sh[CC_COLS*NG];
    uint c = t % CC_COLS, lane = t / CC_COLS;
    ulong off = (ulong)tg.y*((ulong)N*N) + (ulong)tg.x*CC_COLS + c;
    device float2* b = d + off; const device float2* sb = src + off;
    float2 a[REGS];
    for (uint n1=0;n1<REGS;n1++) a[n1] = b[(ulong)(LANES*n1+lane)*N];
    fftN(a, sh + c*NG, W, lane, sgn);
    for (uint cc=0;cc<CHUNK;cc++)
        for (uint j=0;j<LANES;j++){
            ulong q = (ulong)OUTK(cc,j)*N;
            b[q] = a[cc*LANES+j]*scale + 2.0f*sb[q];
        }
}

// ---------------- diagnostic reductions ----------------
// One threadgroup of 256 threads per slice. The partial sums are folded first
// across each SIMD group and then across the eight groups, which keeps the
// threadgroup traffic to eight floats per quantity.

constant uint NSIMD = 8;

inline float tg_sum(float v, threadgroup float* sh, uint t){
    v = simd_sum(v);
    if ((t & 31u) == 0u) { sh[t >> 5]  = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0;
    for (uint i = 0; i < NSIMD; i++) { r += sh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
inline float tg_min(float v, threadgroup float* sh, uint t){
    v = simd_min(v);
    if ((t & 31u) == 0u) { sh[t >> 5] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = sh[0];
    for (uint i = 1; i < NSIMD; i++) { r = min(r, sh[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
inline float tg_max(float v, threadgroup float* sh, uint t){
    v = simd_max(v);
    if ((t & 31u) == 0u) { sh[t >> 5] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = sh[0];
    for (uint i = 1; i < NSIMD; i++) { r = max(r, sh[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

constant uint MAXBH = 8;    // largest bunchharm handled on the GPU
constant uint BSTRIDE = 32; // floats of output per slice
constant uint FSTRIDE = 16;

struct BMPar { float gref; uint npart, nharm, doAux; };

kernel void beam_moments(const device float* X  [[buffer(0)]],
                         const device float* Y  [[buffer(1)]],
                         const device float* PX [[buffer(2)]],
                         const device float* PY [[buffer(3)]],
                         const device float* G  [[buffer(4)]],
                         const device float* TH [[buffer(5)]],
                         device float* out      [[buffer(6)]],
                         constant BMPar& P      [[buffer(7)]],
                         uint slice [[threadgroup_position_in_grid]],
                         uint t     [[thread_index_in_threadgroup]],
                         uint nt    [[threads_per_threadgroup]]){
    threadgroup float sh[NSIMD];
    const uint np = P.npart;
    const ulong o = (ulong)slice * np;

    // first pass: means, and the bunching factor which needs no centring
    float sx=0, sy=0, spx=0, spy=0, sg=0;
    float2 b[MAXBH];
    for (uint h=0; h<MAXBH; h++) { b[h] = float2(0.0f, 0.0f); }
    for (uint i=t; i<np; i+=nt){
        sx += X[o+i]; sy += Y[o+i]; spx += PX[o+i]; spy += PY[o+i]; sg += G[o+i];
        float th = TH[o+i];
        for (uint h=0; h<P.nharm; h++){
            float a = float(h+1) * th;
            b[h] += float2(cos(a), sin(a));
        }
    }
    sx = tg_sum(sx, sh, t); sy = tg_sum(sy, sh, t);
    spx = tg_sum(spx, sh, t); spy = tg_sum(spy, sh, t); sg = tg_sum(sg, sh, t);
    for (uint h=0; h<P.nharm; h++){
        b[h].x = tg_sum(b[h].x, sh, t);
        b[h].y = tg_sum(b[h].y, sh, t);
    }

    const float n = 1.0f / float(np);
    const float mx = sx*n, my = sy*n, mpx = spx*n, mpy = spy*n, mg = sg*n;

    // second pass: centred second moments
    float cx=0, cy=0, cpx=0, cpy=0, cg=0, cxpx=0, cypy=0;
    for (uint i=t; i<np; i+=nt){
        float dx = X[o+i]-mx, dy = Y[o+i]-my;
        float dpx = PX[o+i]-mpx, dpy = PY[o+i]-mpy, dg = G[o+i]-mg;
        cx += dx*dx; cy += dy*dy; cpx += dpx*dpx; cpy += dpy*dpy; cg += dg*dg;
        cxpx += dx*dpx; cypy += dy*dpy;
    }
    cx = tg_sum(cx, sh, t); cy = tg_sum(cy, sh, t);
    cpx = tg_sum(cpx, sh, t); cpy = tg_sum(cpy, sh, t); cg = tg_sum(cg, sh, t);
    cxpx = tg_sum(cxpx, sh, t); cypy = tg_sum(cypy, sh, t);

    device float* r = out + (ulong)slice * BSTRIDE;
    if (t == 0){
        r[0]=mx; r[1]=my; r[2]=mpx; r[3]=mpy; r[4]=mg;
        r[5]=cx*n; r[6]=cy*n; r[7]=cpx*n; r[8]=cpy*n; r[9]=cg*n;
        r[10]=cxpx*n; r[11]=cypy*n;
        for (uint h=0; h<P.nharm; h++){ r[12+2*h]=b[h].x*n; r[13+2*h]=b[h].y*n; }
    }

    if (P.doAux == 0u) { return; }
    float xmn=1e30, xmx=-1e30, ymn=1e30, ymx=-1e30;
    float pxmn=1e30, pxmx=-1e30, pymn=1e30, pymx=-1e30, gmn=1e30, gmx=-1e30;
    for (uint i=t; i<np; i+=nt){
        float vx=X[o+i], vy=Y[o+i], vpx=PX[o+i], vpy=PY[o+i], vg=G[o+i];
        xmn=min(xmn,vx); xmx=max(xmx,vx); ymn=min(ymn,vy); ymx=max(ymx,vy);
        pxmn=min(pxmn,vpx); pxmx=max(pxmx,vpx); pymn=min(pymn,vpy); pymx=max(pymx,vpy);
        gmn=min(gmn,vg); gmx=max(gmx,vg);
    }
    xmn=tg_min(xmn,sh,t); xmx=tg_max(xmx,sh,t); ymn=tg_min(ymn,sh,t); ymx=tg_max(ymx,sh,t);
    pxmn=tg_min(pxmn,sh,t); pxmx=tg_max(pxmx,sh,t);
    pymn=tg_min(pymn,sh,t); pymx=tg_max(pymx,sh,t);
    gmn=tg_min(gmn,sh,t); gmx=tg_max(gmx,sh,t);
    if (t == 0){
        r[20]=xmn; r[21]=xmx; r[22]=ymn; r[23]=ymx;
        r[24]=pxmn; r[25]=pxmx; r[26]=pymn; r[27]=pymx; r[28]=gmn; r[29]=gmx;
    }
}

struct FMPar { uint ngrid, isfar; float shift, wscale; };

// Intensity-weighted transverse moments of one slice. With isfar != 0 the input
// is the transform of the slice and the cell index is FFT-shifted, matching the
// far-field branch of DiagField::getValues.
//
// wscale exists because the transform is unnormalised: |FFT|^2 is up to ngrid^4
// times the cell intensity, and dx^2*|FFT|^2 then overflows FP32 outright at
// ngrid = 256. Every quantity derived from these sums is a ratio, so scaling
// them all by a constant is free.
kernel void field_moments(const device float2* F [[buffer(0)]],
                          device float* out      [[buffer(1)]],
                          constant FMPar& P      [[buffer(2)]],
                          uint slice [[threadgroup_position_in_grid]],
                          uint t     [[thread_index_in_threadgroup]],
                          uint nt    [[threads_per_threadgroup]]){
    threadgroup float sh[NSIMD];
    const uint ng = P.ngrid, nn = ng*ng, hng = (ng+1u)/2u;
    const device float2* f = F + (ulong)slice * nn;

    float p=0, sx=0, sy=0, fr=0, fi=0;
    for (uint i=t; i<nn; i+=nt){
        uint ix = i % ng, iy = i / ng;
        uint j = (P.isfar != 0u) ? (((iy+hng)%ng)*ng + ((ix+hng)%ng)) : i;
        float2 c = f[j] * P.wscale;
        float w = c.x*c.x + c.y*c.y;
        float dx = float(ix) + P.shift, dy = float(iy) + P.shift;
        p += w; sx += dx*w; sy += dy*w; fr += c.x; fi += c.y;
    }
    p = tg_sum(p, sh, t); sx = tg_sum(sx, sh, t); sy = tg_sum(sy, sh, t);
    fr = tg_sum(fr, sh, t); fi = tg_sum(fi, sh, t);

    const float xc = (p > 0.0f) ? sx/p : 0.0f;
    const float yc = (p > 0.0f) ? sy/p : 0.0f;

    float cx=0, cy=0;
    for (uint i=t; i<nn; i+=nt){
        uint ix = i % ng, iy = i / ng;
        uint j = (P.isfar != 0u) ? (((iy+hng)%ng)*ng + ((ix+hng)%ng)) : i;
        float2 c = f[j] * P.wscale;
        float w = c.x*c.x + c.y*c.y;
        float dx = float(ix) + P.shift - xc, dy = float(iy) + P.shift - yc;
        cx += dx*dx*w; cy += dy*dy*w;
    }
    cx = tg_sum(cx, sh, t); cy = tg_sum(cy, sh, t);

    if (t == 0){
        device float* r = out + (ulong)slice * FSTRIDE + (P.isfar != 0u ? 7u : 0u);
        r[0]=p; r[1]=sx; r[2]=sy; r[3]=cx; r[4]=cy;
        if (P.isfar == 0u){ r[5]=fr; r[6]=fi; }
    }
}

kernel void copy_field(device float2* dst [[buffer(0)]],
                       const device float2* src [[buffer(1)]],
                       uint i [[thread_position_in_grid]]){
    dst[i] = src[i];
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

// ---------------- transverse tracking ----------------

struct TrkPar {
    float delz, aw, qx, qy, xoff, yoff, gref;
    uint  mx, my;          // 0 = drift, 1 = focusing, 2 = defocusing
};

kernel void track_beam(device float* X  [[buffer(0)]], device float* Y  [[buffer(1)]],
                       device float* PX [[buffer(2)]], device float* PY [[buffer(3)]],
                       const device float* G [[buffer(4)]],
                       constant TrkPar& P [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]){
    float gam = P.gref + G[gid];
    float px = PX[gid], py = PY[gid];
    // gamma*beta_z. Written as gamma*sqrt(1-r) rather than
    // sqrt(gamma^2-1-aw^2-p^2): at gamma = 11357 the FP32 quantum of gamma^2 is
    // 15, so the O(1) terms would be lost completely in the subtraction.
    float r  = (1.0f + P.aw*P.aw + px*px + py*py)/(gam*gam);
    float gz = gam*sqrt(max(0.0f, 1.0f - r));

    float x = X[gid], y = Y[gid];
    if (P.mx == 0u) {
        x += px*P.delz/gz;
    } else {
        float foc = sqrt(fabs(P.qx)/gz), omg = foc*P.delz;
        float a1, a2, a3;
        if (P.mx == 1u) { a1 = cos(omg);  a2 = sin(omg)/foc;  a3 = -a2*foc*foc; }
        else            { a1 = cosh(omg); a2 = sinh(omg)/foc; a3 =  a2*foc*foc; }
        float xt = x - P.xoff;
        x  = a1*xt + a2*px/gz + P.xoff;
        px = a3*xt*gz + a1*px;
    }
    if (P.my == 0u) {
        y += py*P.delz/gz;
    } else {
        float foc = sqrt(fabs(P.qy)/gz), omg = foc*P.delz;
        float a1, a2, a3;
        if (P.my == 1u) { a1 = cos(omg);  a2 = sin(omg)/foc;  a3 = -a2*foc*foc; }
        else            { a1 = cosh(omg); a2 = sinh(omg)/foc; a3 =  a2*foc*foc; }
        float yt = y - P.yoff;
        y  = a1*yt + a2*py/gz + P.yoff;
        py = a3*yt*gz + a1*py;
    }
    X[gid] = x; PX[gid] = px;
    Y[gid] = y; PY[gid] = py;
}

// ---------------- longitudinal push ----------------

constant uint MAXH = 4;

struct PushPar {
    float delz, aw, xks, xku, gref, autophase;
    float ax, ay, kx, ky, gradx, grady;
    float gridmax, dgrid;
    uint  ngrid, npart, nslice, nfld;
    float rtmp[MAXH];      // und->fc(harm) / field->xks
    float rharm[MAXH];
    uint  first[MAXH];
    uint  onGrid[MAXH];    // filled by the host; 1 if this harmonic is bound
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
struct Acc { float gg, pp; };

inline Acc ode(float dg, float th, float btpar, float ez,
               thread const float2* rpart, constant PushPar& P, Acc k){
    float2 ctmp = float2(0.0f);
    for (uint i = 0; i < P.nfld; i++) {
        float a = P.rharm[i]*th;
        float2 e = float2(cos(a), -sin(a));
        ctmp += float2(rpart[i].x*e.x - rpart[i].y*e.y,
                       rpart[i].x*e.y + rpart[i].y*e.x);
    }
    float tgam   = P.gref + dg;
    float btper0 = btpar - (2.0f/P.xks)*ctmp.x;
    float u      = btper0/(tgam*tgam);
    float invb   = 1.0f + 0.5f*u + 0.375f*u*u;          // 1/sqrt(1-u)
    k.pp += -P.xks*0.5f*u*(1.0f + 0.75f*u) + P.xku;     // dtheta/dz
    k.gg += ctmp.y*invb/tgam - ez;                      // dgamma/dz
    return k;
}

kernel void push_beam(device float* G  [[buffer(0)]], device float* TH [[buffer(1)]],
                      const device float* X [[buffer(2)]], const device float* Y [[buffer(3)]],
                      const device float* PX [[buffer(4)]], const device float* PY [[buffer(5)]],
                      const device float* EZ [[buffer(6)]],
                      constant PushPar& P [[buffer(7)]],
                      const device float2* F0 [[buffer(8)]],
                      const device float2* F1 [[buffer(9)]],
                      const device float2* F2 [[buffer(10)]],
                      const device float2* F3 [[buffer(11)]],
                      uint gid [[thread_position_in_grid]]){
    uint is = gid / P.npart;

    float x = X[gid], y = Y[gid], px = PX[gid], py = PY[gid];
    float dx = x - P.ax, dy = y - P.ay;
    float awloc = 1.0f + 0.5f*(P.kx*dx*dx + P.ky*dy*dy) + P.gradx*dx + P.grady*dy;
    float btpar = 1.0f + px*px + py*py + P.aw*P.aw*awloc*awloc;
    float ez = EZ[is];

    // Bilinear gather of each harmonic at the particle position.
    float2 rpart[MAXH];
    for (uint i = 0; i < MAXH; i++) rpart[i] = float2(0.0f);

    bool on = (x > -P.gridmax && x < P.gridmax && y > -P.gridmax && y < P.gridmax);
    if (on) {
        float wx = (x + P.gridmax)/P.dgrid, wy = (y + P.gridmax)/P.dgrid;
        float fx = floor(wx), fy = floor(wy);
        wx = 1.0f + fx - wx;
        wy = 1.0f + fy - wy;
        uint c = uint(fx) + uint(fy)*P.ngrid;
        for (uint i = 0; i < P.nfld; i++) {
            const device float2* F = (i==0u)?F0:((i==1u)?F1:((i==2u)?F2:F3));
            uint b = ((is + P.first[i]) % P.nslice)*(P.ngrid*P.ngrid) + c;
            float2 cp = F[b           ]*(wx*wy)
                      + F[b+1u        ]*((1.0f-wx)*wy)
                      + F[b+P.ngrid   ]*(wx*(1.0f-wy))
                      + F[b+P.ngrid+1u]*((1.0f-wx)*(1.0f-wy));
            // rtmp*awloc*conj(cpart)
            float s = P.rtmp[i]*awloc;
            rpart[i] = float2(s*cp.x, -s*cp.y);
        }
    }

    float dg = G[gid], th = TH[gid] + P.autophase;

    // Classic RK4, transcribed from BeamSolver::RungeKutta.
    Acc k2 = ode(dg, th, btpar, ez, rpart, P, Acc{0.0f, 0.0f});
    float stpz = 0.5f*P.delz;
    dg += stpz*k2.gg;  th += stpz*k2.pp;
    Acc k3 = k2;
    k2 = ode(dg, th, btpar, ez, rpart, P, Acc{0.0f, 0.0f});
    dg += stpz*(k2.gg - k3.gg);  th += stpz*(k2.pp - k3.pp);
    k3.gg /= 6.0f; k3.pp /= 6.0f;
    k2.gg *= -0.5f; k2.pp *= -0.5f;
    k2 = ode(dg, th, btpar, ez, rpart, P, k2);
    stpz = P.delz;
    dg += stpz*k2.gg;  th += stpz*k2.pp;
    k3.gg -= k2.gg; k3.pp -= k2.pp;
    k2.gg *= 2.0f;  k2.pp *= 2.0f;
    k2 = ode(dg, th, btpar, ez, rpart, P, k2);
    dg += stpz*(k3.gg + k2.gg/6.0f);
    th += stpz*(k3.pp + k2.pp/6.0f);

    G[gid] = dg; TH[gid] = th;
}

// Wakefield energy loss. The wake is a single number per slice, because it is
// driven by the slice current rather than by individual particles, so every
// particle in a slice takes the same kick. The profile is built on the host,
// where the current of every slice is available through an MPI gather; all that
// is left here is the addition. G holds gamma as an offset from a reference
// energy, and the kick is a difference, so it applies unchanged to the offset.
kernel void apply_eloss(device float* G [[buffer(0)]],
                        const device float* DG [[buffer(1)]],
                        constant uint& npart [[buffer(2)]],
                        uint gid [[thread_position_in_grid]])
{
    G[gid] += DG[gid/npart];
}
)MSL";

// Mirrors the MSL structs above.
struct DepPar {
    float gridmax, dgrid, gref, scl;
    float ax, ay, kx, ky, gradx, grady;
    uint32_t ngrid, npart, nslice, harm, first;
};

struct TrkPar {
    float delz, aw, qx, qy, xoff, yoff, gref;
    uint32_t mx, my;
};

enum { kMaxHarm = 4 };

struct PushPar {
    float delz, aw, xks, xku, gref, autophase;
    float ax, ay, kx, ky, gradx, grady;
    float gridmax, dgrid;
    uint32_t ngrid, npart, nslice, nfld;
    float rtmp[kMaxHarm];
    float rharm[kMaxHarm];
    uint32_t first[kMaxHarm];
    uint32_t onGrid[kMaxHarm];
};

enum { kMaxBunchHarm = 8, kBeamStride = 32, kFieldStride = 16 };

struct BMPar {
    float gref;
    uint32_t npart, nharm, doAux;
};

struct FMPar {
    uint32_t ngrid, isfar;
    float shift, wscale;
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

    // FFT shape for this grid, chosen by pickFFTShape() and baked into the
    // shader as preprocessor macros. lanes*regs == ngrid.
    int lanes {0}, regs {0}, rowsPerTG {0}, colsPerTG {0};

    // Field solve scratch. The source is reused across harmonics because they
    // are propagated one at a time. expK depends on the harmonic (through xks)
    // and on the step length, so it is cached per harmonic and rebuilt when
    // delz changes.
    id<MTLBuffer> bSrc {nil};
    id<MTLBuffer> bW {nil};
    id<MTLBuffer> bEZ {nil};
    id<MTLBuffer> bELoss {nil};
    std::vector<id<MTLBuffer> > bExpK;
    std::vector<double> delzCached;

    // Diagnostic reduction output, a fixed number of floats per slice.
    id<MTLBuffer> bBM {nil};
    id<MTLBuffer> bFM {nil};

    id<MTLComputePipelineState> pZero, pDep, pRow, pRowM, pCol, pColA, pTrk, pPush;
    id<MTLComputePipelineState> pBM, pFM, pCopy, pELoss;

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

// The FFT is a four-step decomposition ngrid = regs * lanes. lanes threads
// cooperate on one transform, each holding regs complex values in registers and
// exchanging once through threadgroup memory.
//
// regs is either lanes or 2*lanes; in the latter case each thread performs two
// of the second-stage transforms after the exchange. rowsPerTG and colsPerTG
// are how many transforms share one threadgroup, bounded by the 32 KB of
// threadgroup memory (8 bytes per point) and by 1024 threads.
//
// Returns false if the size is not supported.
static bool pickFFTShape(int ng, int &lanes, int &regs, int &rows, int &cols)
{
    switch (ng) {
    case   64: lanes =  8; regs =  8; rows = 16; cols = 16; return true;
    case  128: lanes =  8; regs = 16; rows = 16; cols = 16; return true;
    case  256: lanes = 16; regs = 16; rows =  8; cols = 16; return true;
    case  512: lanes = 16; regs = 32; rows =  4; cols =  8; return true;
    case 1024: lanes = 32; regs = 32; rows =  2; cols =  4; return true;
    default:   return false;
    }
}

// Nearest supported size, for the error message. Ties go to the larger grid,
// because dropping resolution silently is the worse surprise.
static int nearestSupported(int ng)
{
    static const int sizes[] = {64, 128, 256, 512, 1024};
    int best = sizes[0];
    double bd = 1e300;
    for (int i = 0; i < 5; i++) {
        const double d = std::fabs(std::log(static_cast<double>(sizes[i])) -
                                   std::log(static_cast<double>(ng > 0 ? ng : 1)));
        if (d <= bd) { bd = d; best = sizes[i]; }
    }
    return best;
}

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
            os << "ngrid = " << f->ngrid << " is not supported by the Metal field solver, "
                  "which handles powers of two from 64 to 1024. Set ngrid = "
               << nearestSupported(f->ngrid)
               << " in &field. Genesis decks traditionally use an odd ngrid so that a grid "
                  "point sits exactly on axis, but nothing in the physics requires that.";
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
    p_->bEZ = alloc(static_cast<size_t>(p_->nslice) * sizeof(float));
    p_->bELoss = alloc(static_cast<size_t>(p_->nslice) * sizeof(float));
    p_->bExpK.clear();
    p_->delzCached.assign(field->size(), -1.0);
    for (size_t i = 0; i < field->size(); i++) {
        p_->bExpK.push_back(alloc(nn * 2 * sizeof(float)));
    }
    if (p_->bSrc == nil || p_->bW == nil) {
        reason = "field solver buffer allocation failed";
        return false;
    }

    p_->bBM = alloc(static_cast<size_t>(p_->nslice) * kBeamStride * sizeof(float));
    p_->bFM = alloc(static_cast<size_t>(p_->nslice) * kFieldStride * sizeof(float));
    if (p_->bBM == nil || p_->bFM == nil) {
        reason = "diagnostic buffer allocation failed";
        return false;
    }

    std::complex<float> *W = (std::complex<float> *)[p_->bW contents];
    for (int m = 0; m < ng; m++) {
        const double a = -2.0 * M_PI * m / ng;
        W[m] = std::complex<float>(static_cast<float>(cos(a)),
                                   static_cast<float>(sin(a)));
    }

    NSError *err = nil;
    // The FFT shape has to be a compile-time constant, because it sizes both
    // the register arrays and the threadgroup allocation. The library is built
    // from source at startup anyway, so it is simply specialised to this grid.
    MTLCompileOptions *copt = [[MTLCompileOptions alloc] init];
    copt.preprocessorMacros = @{
        @"NG"      : @(ng),
        @"LANES"   : @(p_->lanes),
        @"REGS"    : @(p_->regs),
        @"CHUNK"   : @(p_->regs / p_->lanes),
        @"RF_ROWS" : @(p_->rowsPerTG),
        @"CC_COLS" : @(p_->colsPerTG),
    };
    id<MTLLibrary> lib = [p_->dev newLibraryWithSource:kMSL options:copt error:&err];
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
    p_->pRow = pso(@"fft_rows");
    p_->pRowM = pso(@"fft_rows_mul");
    p_->pCol = pso(@"fft_cols");
    p_->pColA = pso(@"fft_cols_add");
    p_->pTrk = pso(@"track_beam");
    p_->pPush = pso(@"push_beam");
    p_->pBM = pso(@"beam_moments");
    p_->pFM = pso(@"field_moments");
    p_->pCopy = pso(@"copy_field");
    p_->pELoss = pso(@"apply_eloss");
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
            const MTLSize rowTG = MTLSizeMake(ng / p_->rowsPerTG, ns, 1);
            const MTLSize rowT = MTLSizeMake(p_->rowsPerTG * p_->lanes, 1, 1);
            const MTLSize colTG = MTLSizeMake(ng / p_->colsPerTG, ns, 1);
            const MTLSize colT = MTLSizeMake(p_->colsPerTG * p_->lanes, 1, 1);

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

bool MetalEngine::beamStep(Beam *beam, Undulator *und,
                           std::vector<Field *> *field, double delz,
                           std::string &reason)
{
    // Everything Beam::track does that is not yet on the GPU has to be
    // inactive, otherwise the answer would silently differ from the CPU.
    double angle, lb, ld, lt, cx, cy;
    und->getChicaneParameters(&angle, &lb, &ld, &lt);
    und->getCorrectorParameters(&cx, &cy);
    if (angle != 0) {
        reason = "chicane";
        return false;
    }
    if (cx != 0 || cy != 0) {
        reason = "corrector";
        return false;
    }
    if (beam->gpuUnsupportedPhysics(reason)) {
        return false;
    }

    @autoreleasepool {
        const int istep = und->getStep();
        const int ns = p_->nslice;
        const size_t nthread = static_cast<size_t>(ns) * p_->npart;
        const MTLSize grid = MTLSizeMake(nthread, 1, 1);
        const MTLSize tg = MTLSizeMake(256, 1, 1);

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
        if (qx != 0) {
            xoff /= qx;
        }
        if (qy != 0) {
            yoff /= qy;
        }
        T.aw = static_cast<float>(aw);
        T.qx = static_cast<float>(qx);
        T.qy = static_cast<float>(qy);
        T.xoff = static_cast<float>(xoff);
        T.yoff = static_cast<float>(yoff);
        T.gref = static_cast<float>(p_->gref);
        T.delz = static_cast<float>(0.5 * delz);

        // Longitudinal push. Note that the undulator parameters used here are
        // the raw lattice values, not the ones zeroed outside an undulator that
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
        P.ngrid = static_cast<uint32_t>(p_->ngrid[0]);
        P.npart = static_cast<uint32_t>(p_->npart);
        P.nslice = static_cast<uint32_t>(ns);
        P.nfld = static_cast<uint32_t>(field->size());

        double xks = 1;
        for (size_t i = 0; i < field->size(); i++) {
            Field *f = field->at(i);
            const int harm = f->getHarm();
            xks = f->xks / static_cast<double>(harm);
            P.rtmp[i] = static_cast<float>(und->fc(harm) / f->xks);
            P.rharm[i] = static_cast<float>(harm);
            P.first[i] = static_cast<uint32_t>(f->first);
            P.onGrid[i] = 1;
        }
        for (size_t i = field->size(); i < kMaxHarm; i++) {
            P.rtmp[i] = 0;
            P.rharm[i] = 0;
            P.first[i] = 0;
            P.onGrid[i] = 0;
        }
        P.xks = static_cast<float>(xks);
        double xku = und->getku();
        if (xku == 0) {
            // In a drift a particle at the reference energy stays in phase.
            xku = xks * 0.5 / gamma0 / gamma0;
        }
        P.xku = static_cast<float>(xku);

        // Long-range space charge, in units of the electron rest mass. Zero
        // unless the space-charge solver is switched on.
        float *ez = (float *)[p_->bEZ contents];
        for (int is = 0; is < ns; is++) {
            ez[is] = (is < static_cast<int>(beam->longESC.size()))
                         ? static_cast<float>(-beam->longESC[is] / 511000.0)
                         : 0.0f;
        }

        // Wakefields. The loss profile is one number per slice and is built on
        // the host, which is where the current of every slice is reachable
        // through an MPI gather; the GPU only has to add it to the particles.
        // Beam::track applies this after the longitudinal push, so the dispatch
        // below sits between the push and the second transverse half step.
        const bool haveWake = beam->computeWakeLoss(und);
        if (haveWake) {
            float *el = (float *)[p_->bELoss contents];
            for (int is = 0; is < ns; is++) {
                el[is] = (is < static_cast<int>(beam->eloss.size()))
                             ? static_cast<float>(beam->eloss[is] * delz / 511000.0)
                             : 0.0f;
            }
        }

        id<MTLCommandBuffer> cb = [p_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

        auto encTrack = [&]() {
            [e setComputePipelineState:p_->pTrk];
            [e setBuffer:p_->bX offset:0 atIndex:0];
            [e setBuffer:p_->bY offset:0 atIndex:1];
            [e setBuffer:p_->bPX offset:0 atIndex:2];
            [e setBuffer:p_->bPY offset:0 atIndex:3];
            [e setBuffer:p_->bG offset:0 atIndex:4];
            [e setBytes:&T length:sizeof(T) atIndex:5];
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
        };

        encTrack();   // first half step

        [e setComputePipelineState:p_->pPush];
        [e setBuffer:p_->bG offset:0 atIndex:0];
        [e setBuffer:p_->bT offset:0 atIndex:1];
        [e setBuffer:p_->bX offset:0 atIndex:2];
        [e setBuffer:p_->bY offset:0 atIndex:3];
        [e setBuffer:p_->bPX offset:0 atIndex:4];
        [e setBuffer:p_->bPY offset:0 atIndex:5];
        [e setBuffer:p_->bEZ offset:0 atIndex:6];
        [e setBytes:&P length:sizeof(P) atIndex:7];
        for (int i = 0; i < kMaxHarm; i++) {
            const size_t k = (i < static_cast<int>(p_->bField.size()))
                                 ? static_cast<size_t>(i)
                                 : 0;
            [e setBuffer:p_->bField[k] offset:0 atIndex:8 + i];
        }
        [e dispatchThreads:grid threadsPerThreadgroup:tg];

        if (haveWake) {
            const uint32_t np = static_cast<uint32_t>(p_->npart);
            [e setComputePipelineState:p_->pELoss];
            [e setBuffer:p_->bG offset:0 atIndex:0];
            [e setBuffer:p_->bELoss offset:0 atIndex:1];
            [e setBytes:&np length:sizeof(np) atIndex:2];
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
        }

        encTrack();   // second half step

        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return true;
}

bool MetalEngine::beamMoments(int nharm, bool wantAux, BeamSliceMoments &out) const
{
    if (nharm < 1 || nharm > kMaxBunchHarm) { return false; }

    @autoreleasepool {
        BMPar P;
        P.gref = static_cast<float>(p_->gref);
        P.npart = static_cast<uint32_t>(p_->npart);
        P.nharm = static_cast<uint32_t>(nharm);
        P.doAux = wantAux ? 1u : 0u;

        id<MTLCommandBuffer> cb = [p_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p_->pBM];
        [e setBuffer:p_->bX offset:0 atIndex:0];
        [e setBuffer:p_->bY offset:0 atIndex:1];
        [e setBuffer:p_->bPX offset:0 atIndex:2];
        [e setBuffer:p_->bPY offset:0 atIndex:3];
        [e setBuffer:p_->bG offset:0 atIndex:4];
        [e setBuffer:p_->bT offset:0 atIndex:5];
        [e setBuffer:p_->bBM offset:0 atIndex:6];
        [e setBytes:&P length:sizeof(P) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(p_->nslice, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    const int ns = p_->nslice;
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
    const float *r = (const float *)[p_->bBM contents];
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
            out.xmin[is] = s[20]; out.xmax[is] = s[21];
            out.ymin[is] = s[22]; out.ymax[is] = s[23];
            out.pxmin[is] = s[24]; out.pxmax[is] = s[25];
            out.pymin[is] = s[26]; out.pymax[is] = s[27];
            out.gmin[is] = p_->gref + s[28];
            out.gmax[is] = p_->gref + s[29];
        }
    }
    return true;
}

bool MetalEngine::fieldMoments(int ih, bool wantFar, FieldSliceMoments &out) const
{
    if (ih < 0 || ih >= static_cast<int>(p_->bField.size())) { return false; }

    const int ns = p_->nslice;
    const int ng = p_->ngrid[ih];
    const size_t nn = static_cast<size_t>(ng) * ng;

    @autoreleasepool {
        FMPar P;
        P.ngrid = static_cast<uint32_t>(ng);
        P.isfar = 0;
        P.shift = static_cast<float>(-0.5 * (ng - 1));
        P.wscale = 1.0f;

        const float fwd = -1.0f, one = 1.0f;
        const MTLSize rowTG = MTLSizeMake(ng / p_->rowsPerTG, ns, 1);
        const MTLSize rowT = MTLSizeMake(p_->rowsPerTG * p_->lanes, 1, 1);
        const MTLSize colTG = MTLSizeMake(ng / p_->colsPerTG, ns, 1);
        const MTLSize colT = MTLSizeMake(p_->colsPerTG * p_->lanes, 1, 1);
        const MTLSize slices = MTLSizeMake(ns, 1, 1);
        const MTLSize tg = MTLSizeMake(256, 1, 1);

        id<MTLCommandBuffer> cb = [p_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

        [e setComputePipelineState:p_->pFM];
        [e setBuffer:p_->bField[ih] offset:0 atIndex:0];
        [e setBuffer:p_->bFM offset:0 atIndex:1];
        [e setBytes:&P length:sizeof(P) atIndex:2];
        [e dispatchThreadgroups:slices threadsPerThreadgroup:tg];

        if (wantFar) {
            // The near field has to survive, so transform a copy. bSrc is only
            // live inside fieldStep, so it is free to borrow here.
            [e setComputePipelineState:p_->pCopy];
            [e setBuffer:p_->bSrc offset:0 atIndex:0];
            [e setBuffer:p_->bField[ih] offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(nn * ns, 1, 1) threadsPerThreadgroup:tg];

            [e setComputePipelineState:p_->pRow];
            [e setBuffer:p_->bSrc offset:0 atIndex:0];
            [e setBuffer:p_->bW offset:0 atIndex:1];
            [e setBytes:&fwd length:4 atIndex:2];
            [e dispatchThreadgroups:rowTG threadsPerThreadgroup:rowT];

            [e setComputePipelineState:p_->pCol];
            [e setBuffer:p_->bSrc offset:0 atIndex:0];
            [e setBuffer:p_->bW offset:0 atIndex:1];
            [e setBytes:&fwd length:4 atIndex:2];
            [e setBytes:&one length:4 atIndex:3];
            [e dispatchThreadgroups:colTG threadsPerThreadgroup:colT];

            FMPar F = P;
            F.isfar = 1;
            F.wscale = 1.0f / static_cast<float>(ng);
            [e setComputePipelineState:p_->pFM];
            [e setBuffer:p_->bSrc offset:0 atIndex:0];
            [e setBuffer:p_->bFM offset:0 atIndex:1];
            [e setBytes:&F length:sizeof(F) atIndex:2];
            [e dispatchThreadgroups:slices threadsPerThreadgroup:tg];
        }

        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    out.nslice = ns;
    out.hasFar = wantFar;
    auto fill = [ns](std::vector<double> &v) { v.assign(ns, 0.0); };
    fill(out.power); fill(out.x1); fill(out.x2); fill(out.y1); fill(out.y2);
    fill(out.ffre); fill(out.ffim); fill(out.midre); fill(out.midim);
    if (wantFar) {
        fill(out.fpower); fill(out.fx1); fill(out.fx2); fill(out.fy1); fill(out.fy2);
    }

    const float *r = (const float *)[p_->bFM contents];
    const std::complex<float> *fld =
        (const std::complex<float> *)[p_->bField[ih] contents];
    const size_t mid = (nn - 1) / 2;
    auto uncentre = [](double c, double s1, double w) {
        return (w > 0.0) ? c + s1 * s1 / w : 0.0;
    };
    for (int is = 0; is < ns; is++) {
        const float *s = r + static_cast<size_t>(is) * kFieldStride;
        out.power[is] = s[0];
        out.x1[is] = s[1];  out.y1[is] = s[2];
        out.x2[is] = uncentre(s[3], s[1], s[0]);
        out.y2[is] = uncentre(s[4], s[2], s[0]);
        out.ffre[is] = s[5]; out.ffim[is] = s[6];
        if (wantFar) {
            out.fpower[is] = s[7];
            out.fx1[is] = s[8];  out.fy1[is] = s[9];
            out.fx2[is] = uncentre(s[10], s[8], s[7]);
            out.fy2[is] = uncentre(s[11], s[9], s[7]);
        }
        const std::complex<float> c = fld[static_cast<size_t>(is) * nn + mid];
        out.midre[is] = c.real();
        out.midim[is] = c.imag();
    }
    return true;
}

void MetalEngine::upload(Beam *beam, std::vector<Field *> *field)
{
    this->uploadBeam(beam);

    for (size_t i = 0; i < p_->bField.size(); i++) {
        const size_t nn = static_cast<size_t>(p_->ngrid[i]) * p_->ngrid[i];
        std::complex<float> *dst = (std::complex<float> *)[p_->bField[i] contents];
        for (int is = 0; is < p_->nslice; is++) {
            const std::complex<double> *src = field->at(i)->field[is].data();
            std::complex<float> *d = dst + static_cast<size_t>(is) * nn;
            for (size_t k = 0; k < nn; k++) {
                d[k] = std::complex<float>(src[k]);
            }
        }
    }
}

void MetalEngine::uploadBeam(Beam *beam)
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

// Slippage moves exactly one slice per slip event, so the whole field does not
// have to make the trip. These two keep that one slice consistent while the
// rest of the grid stays resident.
void MetalEngine::downloadFieldSlice(int ifld, int islice, Field *field)
{
    if (ifld < 0 || static_cast<size_t>(ifld) >= p_->bField.size()) { return; }
    if (islice < 0 || islice >= p_->nslice) { return; }
    const size_t nn = static_cast<size_t>(p_->ngrid[ifld]) * p_->ngrid[ifld];
    const std::complex<float> *s =
        (const std::complex<float> *)[p_->bField[ifld] contents] +
        static_cast<size_t>(islice) * nn;
    std::complex<double> *dst = field->field[islice].data();
    for (size_t k = 0; k < nn; k++) {
        dst[k] = std::complex<double>(s[k]);
    }
}

void MetalEngine::uploadFieldSlice(int ifld, int islice, const Field *field)
{
    if (ifld < 0 || static_cast<size_t>(ifld) >= p_->bField.size()) { return; }
    if (islice < 0 || islice >= p_->nslice) { return; }
    const size_t nn = static_cast<size_t>(p_->ngrid[ifld]) * p_->ngrid[ifld];
    std::complex<float> *d =
        (std::complex<float> *)[p_->bField[ifld] contents] +
        static_cast<size_t>(islice) * nn;
    const std::complex<double> *src = field->field[islice].data();
    for (size_t k = 0; k < nn; k++) {
        d[k] = std::complex<float>(src[k]);
    }
}

void MetalEngine::downloadBeam(Beam *beam)
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
