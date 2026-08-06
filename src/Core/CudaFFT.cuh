#ifndef __GENESIS_CUDAFFT__
#define __GENESIS_CUDAFFT__

// The transverse transform used by the CUDA field solve, and the complex
// helpers it needs.
//
// This lives in a header rather than inside CudaEngine.cu so that
// tools/fftcheck.cu tests the transform Genesis actually runs rather than a
// copy of it that can drift. Nothing here knows about Genesis: it is a batched
// two-dimensional complex FFT, with the propagator multiply, the source
// addition and the source filter fused into the passes that would otherwise
// have to read the same memory twice.

#include <cuda_runtime.h>

#include <cmath>

// ---------------------------------------------------------------------------
// device-side complex helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
__device__ __forceinline__ float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
__device__ __forceinline__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 cdiv(float2 a, float2 b)
{
    const float d = b.x * b.x + b.y * b.y;
    return make_float2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

// ---------------------------------------------------------------------------
// the transform
// ---------------------------------------------------------------------------
//
// A four-step Cooley-Tukey decomposition N = REGS * LANES: LANES threads
// cooperate on one N-point transform, each holding REGS complex values in
// registers, with a single exchange through shared memory between the two
// stages. Two stages rather than log2(N) is both faster and more accurate than
// a radix-2 ladder.
//
// The register DFTs leave their output permuted: for a P-point transform
// written as P = P1 x P2, slot P2*k1 + k2 holds frequency k1 + P1*k2. The
// caller folds that permutation into its store index, which avoids a P-element
// temporary and the dynamic indexing that would come with it.
//
// The twiddle table holds W_N, so W_P^m lives at index (N/P)*m.

__device__ __forceinline__ void dft4(float2 *a, float s)
{
    const float2 t0 = a[0] + a[2], t1 = a[0] - a[2], t2 = a[1] + a[3],
                 d = a[1] - a[3];
    const float2 t3 = make_float2(s * d.y, -s * d.x);
    a[0] = t0 + t2;
    a[1] = t1 + t3;
    a[2] = t0 - t2;
    a[3] = t1 - t3;
}

template <int N>
__device__ __forceinline__ void dft8(float2 *a, const float2 *W, float s)
{
#pragma unroll
    for (int n2 = 0; n2 < 4; n2++) {
        const float2 u = a[n2], v = a[4 + n2];
        float2 w = W[((N / 8) * n2) & (N - 1)];
        w.y *= s;
        a[n2] = u + v;
        a[4 + n2] = cmul(u - v, w);
    }
    float2 t[4];
#pragma unroll
    for (int k1 = 0; k1 < 2; k1++) {
#pragma unroll
        for (int n2 = 0; n2 < 4; n2++) { t[n2] = a[4 * k1 + n2]; }
        dft4(t, s);
#pragma unroll
        for (int k2 = 0; k2 < 4; k2++) { a[4 * k1 + k2] = t[k2]; }
    }
}
// Natural-order 8-point transform, needed as the inner stage of dft32.
template <int N>
__device__ __forceinline__ void dft8n(float2 *a, const float2 *W, float s)
{
    dft8<N>(a, W, s);
    float2 t[8];
#pragma unroll
    for (int j = 0; j < 8; j++) { t[j] = a[j]; }
#pragma unroll
    for (int j = 0; j < 8; j++) { a[(j >> 2) + 2 * (j & 3)] = t[j]; }
}
template <int N>
__device__ __forceinline__ void dft16(float2 *a, const float2 *W, float s)
{
    float2 t[4];
#pragma unroll
    for (int n2 = 0; n2 < 4; n2++) {
#pragma unroll
        for (int j = 0; j < 4; j++) { t[j] = a[4 * j + n2]; }
        dft4(t, s);
#pragma unroll
        for (int k1 = 0; k1 < 4; k1++) {
            float2 w = W[((N / 16) * n2 * k1) & (N - 1)];
            w.y *= s;
            a[4 * k1 + n2] = cmul(t[k1], w);
        }
    }
#pragma unroll
    for (int k1 = 0; k1 < 4; k1++) {
#pragma unroll
        for (int n2 = 0; n2 < 4; n2++) { t[n2] = a[4 * k1 + n2]; }
        dft4(t, s);
#pragma unroll
        for (int k2 = 0; k2 < 4; k2++) { a[4 * k1 + k2] = t[k2]; }
    }
}
template <int N>
__device__ __forceinline__ void dft32(float2 *a, const float2 *W, float s)
{
    float2 t[8];
#pragma unroll
    for (int n2 = 0; n2 < 8; n2++) {
#pragma unroll
        for (int j = 0; j < 4; j++) { t[j] = a[8 * j + n2]; }
        dft4(t, s);
#pragma unroll
        for (int k1 = 0; k1 < 4; k1++) {
            float2 w = W[((N / 32) * n2 * k1) & (N - 1)];
            w.y *= s;
            a[8 * k1 + n2] = cmul(t[k1], w);
        }
    }
#pragma unroll
    for (int k1 = 0; k1 < 4; k1++) {
#pragma unroll
        for (int n2 = 0; n2 < 8; n2++) { t[n2] = a[8 * k1 + n2]; }
        dft8n<N>(t, W, s);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) { a[8 * k1 + k2] = t[k2]; }
    }
}

template <int N, int P>
__device__ __forceinline__ void dftP(float2 *a, const float2 *W, float s)
{
    if constexpr (P == 8) {
        dft8<N>(a, W, s);
    } else if constexpr (P == 16) {
        dft16<N>(a, W, s);
    } else {
        dft32<N>(a, W, s);
    }
}
template <int P>
__device__ __forceinline__ unsigned permP(unsigned j)
{
    if constexpr (P == 8) {
        return (j >> 2) + 2u * (j & 3u);
    } else if constexpr (P == 16) {
        return (j >> 2) + 4u * (j & 3u);
    } else {
        return (j >> 3) + 4u * (j & 7u);
    }
}

// N-point FFT across LANES threads. On entry a[n1] = x[LANES*n1 + lane]. On
// exit slot cc*LANES + j holds X[(lane + cc*LANES) + REGS*permP<LANES>(j)].
// The exchange index is XOR-swizzled to avoid bank conflicts without padding,
// which is what lets several transforms share one shared-memory allocation.
template <int N, int LANES, int REGS>
__device__ __forceinline__ void fftN(float2 *a, float2 *s, const float2 *W,
                                     unsigned lane, float sgn)
{
    constexpr int CHUNK = REGS / LANES;
    dftP<N, REGS>(a, W, sgn);
#pragma unroll
    for (int j = 0; j < REGS; j++) {
        const unsigned k1 = permP<REGS>(j);
        float2 w = W[(lane * k1) & (N - 1u)];
        w.y *= sgn;
        s[k1 * LANES + (lane ^ (k1 & (LANES - 1u)))] = cmul(a[j], w);
    }
    __syncthreads();
#pragma unroll
    for (int cc = 0; cc < CHUNK; cc++) {
#pragma unroll
        for (int n2 = 0; n2 < LANES; n2++) {
            a[cc * LANES + n2] = s[(lane + cc * LANES) * LANES + (n2 ^ lane)];
        }
    }
    __syncthreads();
#pragma unroll
    for (int cc = 0; cc < CHUNK; cc++) { dftP<N, LANES>(a + cc * LANES, W, sgn); }
}

// Frequency held by output slot cc*LANES + j in the thread with this lane.
template <int LANES, int REGS>
__device__ __forceinline__ unsigned outk(unsigned lane, int cc, int j)
{
    return (lane + cc * LANES) + REGS * permP<LANES>(static_cast<unsigned>(j));
}

// ---- row passes -----------------------------------------------------------
//
// One block holds ROWS transforms; blockIdx.x selects the group of rows and
// blockIdx.y the slice. Consecutive threads carry consecutive lanes, so the
// loads and stores of a row are contiguous.

template <int N, int LANES, int REGS, int ROWS>
__global__ void kFftRows(float2 *__restrict__ d, const float2 *__restrict__ W,
                         float sgn)
{
    __shared__ float2 sh[ROWS * N];
    const unsigned t = threadIdx.x, lane = t % LANES, r = t / LANES;
    float2 *p = d + static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) +
                static_cast<size_t>(blockIdx.x * ROWS + r) * N;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) { a[n1] = p[LANES * n1 + lane]; }
    fftN<N, LANES, REGS>(a, sh + r * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            p[outk<LANES, REGS>(lane, cc, j)] = a[cc * LANES + j];
        }
    }
}

// Out-of-place row pass: reads one buffer, writes another. The far-field
// diagnostic needs the transform of a slice while the slice itself has to
// survive, and doing the first pass out of place is free, where copying the
// whole field first costs a read and a write of every point.
template <int N, int LANES, int REGS, int ROWS>
__global__ void kFftRowsOut(float2 *__restrict__ d, const float2 *__restrict__ W,
                            float sgn, const float2 *__restrict__ s0)
{
    __shared__ float2 sh[ROWS * N];
    const unsigned t = threadIdx.x, lane = t % LANES, r = t / LANES;
    const size_t off = static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) +
                       static_cast<size_t>(blockIdx.x * ROWS + r) * N;
    float2 *p = d + off;
    const float2 *q = s0 + off;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) { a[n1] = q[LANES * n1 + lane]; }
    fftN<N, LANES, REGS>(a, sh + r * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            p[outk<LANES, REGS>(lane, cc, j)] = a[cc * LANES + j];
        }
    }
}

// Inverse row pass with the propagator folded into the load.
template <int N, int LANES, int REGS, int ROWS>
__global__ void kFftRowsMul(float2 *__restrict__ d, const float2 *__restrict__ W,
                            float sgn, const float2 *__restrict__ expK)
{
    __shared__ float2 sh[ROWS * N];
    const unsigned t = threadIdx.x, lane = t % LANES, r = t / LANES;
    const size_t row = static_cast<size_t>(blockIdx.x * ROWS + r);
    float2 *p = d + static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) + row * N;
    const float2 *k = expK + row * N;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) {
        a[n1] = cmul(p[LANES * n1 + lane], k[LANES * n1 + lane]);
    }
    fftN<N, LANES, REGS>(a, sh + r * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            p[outk<LANES, REGS>(lane, cc, j)] = a[cc * LANES + j];
        }
    }
}

// Inverse row pass which also adds the transformed source. Used only when the
// source filter is on, where the source has been through its own forward
// transform and so has to be combined in Fourier space rather than after the
// back transform.
template <int N, int LANES, int REGS, int ROWS>
__global__ void kFftRowsMulAdd(float2 *__restrict__ d, const float2 *__restrict__ W,
                               float sgn, const float2 *__restrict__ expK,
                               const float2 *__restrict__ sf)
{
    __shared__ float2 sh[ROWS * N];
    const unsigned t = threadIdx.x, lane = t % LANES, r = t / LANES;
    const size_t row = static_cast<size_t>(blockIdx.x * ROWS + r);
    const size_t off =
        static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) + row * N;
    float2 *p = d + off;
    const float2 *k = expK + row * N;
    const float2 *s = sf + off;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) {
        const int j = LANES * n1 + lane;
        a[n1] = cmul(p[j], k[j]) + 2.0f * s[j];
    }
    fftN<N, LANES, REGS>(a, sh + r * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            p[outk<LANES, REGS>(lane, cc, j)] = a[cc * LANES + j];
        }
    }
}

// ---- column passes --------------------------------------------------------
//
// Consecutive threads carry consecutive columns, so a column pass reads and
// writes contiguously across the block even though each transform is strided.

template <int N, int LANES, int REGS, int COLS>
__global__ void kFftCols(float2 *__restrict__ d, const float2 *__restrict__ W,
                         float sgn, float scale)
{
    __shared__ float2 sh[COLS * N];
    const unsigned t = threadIdx.x, c = t % COLS, lane = t / COLS;
    float2 *b = d + static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) +
                static_cast<size_t>(blockIdx.x) * COLS + c;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) {
        a[n1] = b[static_cast<size_t>(LANES * n1 + lane) * N];
    }
    fftN<N, LANES, REGS>(a, sh + c * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            b[static_cast<size_t>(outk<LANES, REGS>(lane, cc, j)) * N] =
                a[cc * LANES + j] * scale;
        }
    }
}

template <int N, int LANES, int REGS, int COLS>
__global__ void kFftColsAdd(float2 *__restrict__ d, const float2 *__restrict__ W,
                            float sgn, float scale, const float2 *__restrict__ src)
{
    __shared__ float2 sh[COLS * N];
    const unsigned t = threadIdx.x, c = t % COLS, lane = t / COLS;
    const size_t off = static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) +
                       static_cast<size_t>(blockIdx.x) * COLS + c;
    float2 *b = d + off;
    const float2 *sb = src + off;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) {
        a[n1] = b[static_cast<size_t>(LANES * n1 + lane) * N];
    }
    fftN<N, LANES, REGS>(a, sh + c * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            const size_t q = static_cast<size_t>(outk<LANES, REGS>(lane, cc, j)) * N;
            b[q] = a[cc * LANES + j] * scale + 2.0f * sb[q];
        }
    }
}

// Forward column pass which applies the source filter as it writes. The filter
// is a real function of the transverse wavenumber and does not depend on the
// slice, so it is indexed by position within the slice.
template <int N, int LANES, int REGS, int COLS>
__global__ void kFftColsFilt(float2 *__restrict__ d, const float2 *__restrict__ W,
                             float sgn, const float *__restrict__ filt)
{
    __shared__ float2 sh[COLS * N];
    const unsigned t = threadIdx.x, c = t % COLS, lane = t / COLS;
    const size_t col = static_cast<size_t>(blockIdx.x) * COLS + c;
    float2 *b = d + static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(N) * N) + col;
    float2 a[REGS];
#pragma unroll
    for (int n1 = 0; n1 < REGS; n1++) {
        a[n1] = b[static_cast<size_t>(LANES * n1 + lane) * N];
    }
    fftN<N, LANES, REGS>(a, sh + c * N, W, lane, sgn);
#pragma unroll
    for (int cc = 0; cc < REGS / LANES; cc++) {
#pragma unroll
        for (int j = 0; j < LANES; j++) {
            const size_t q = static_cast<size_t>(outk<LANES, REGS>(lane, cc, j)) * N;
            b[q] = a[cc * LANES + j] * filt[col + q];
        }
    }
}

// ---------------------------------------------------------------------------
// launch table
// ---------------------------------------------------------------------------
//
// The transform shape has to be a compile-time constant, because it sizes both
// the register arrays and the shared allocation. init() picks the shape for the
// grid in the deck and fills this table with the matching instantiations, so
// the dispatch happens once per run rather than once per pass.

struct FFTLaunch {
    void (*rows)(cudaStream_t, int ns, float2 *, const float2 *, float);
    void (*rowsOut)(cudaStream_t, int ns, float2 *, const float2 *, float,
                    const float2 *);
    void (*rowsMul)(cudaStream_t, int ns, float2 *, const float2 *, float,
                    const float2 *);
    void (*rowsMulAdd)(cudaStream_t, int ns, float2 *, const float2 *, float,
                       const float2 *, const float2 *);
    void (*cols)(cudaStream_t, int ns, float2 *, const float2 *, float, float);
    void (*colsAdd)(cudaStream_t, int ns, float2 *, const float2 *, float, float,
                    const float2 *);
    void (*colsFilt)(cudaStream_t, int ns, float2 *, const float2 *, float,
                     const float *);
};

template <int N, int LANES, int REGS, int ROWS, int COLS>
FFTLaunch makeLaunch()
{
    FFTLaunch L;
    L.rows = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn) {
        kFftRows<N, LANES, REGS, ROWS>
            <<<dim3(N / ROWS, ns), ROWS * LANES, 0, s>>>(d, W, sgn);
    };
    L.rowsOut = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn,
                   const float2 *s0) {
        kFftRowsOut<N, LANES, REGS, ROWS>
            <<<dim3(N / ROWS, ns), ROWS * LANES, 0, s>>>(d, W, sgn, s0);
    };
    L.rowsMul = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn,
                   const float2 *k) {
        kFftRowsMul<N, LANES, REGS, ROWS>
            <<<dim3(N / ROWS, ns), ROWS * LANES, 0, s>>>(d, W, sgn, k);
    };
    L.rowsMulAdd = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn,
                      const float2 *k, const float2 *sf) {
        kFftRowsMulAdd<N, LANES, REGS, ROWS>
            <<<dim3(N / ROWS, ns), ROWS * LANES, 0, s>>>(d, W, sgn, k, sf);
    };
    L.cols = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn,
                float scale) {
        kFftCols<N, LANES, REGS, COLS>
            <<<dim3(N / COLS, ns), COLS * LANES, 0, s>>>(d, W, sgn, scale);
    };
    L.colsAdd = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn,
                   float scale, const float2 *src) {
        kFftColsAdd<N, LANES, REGS, COLS>
            <<<dim3(N / COLS, ns), COLS * LANES, 0, s>>>(d, W, sgn, scale, src);
    };
    L.colsFilt = [](cudaStream_t s, int ns, float2 *d, const float2 *W, float sgn,
                    const float *f) {
        kFftColsFilt<N, LANES, REGS, COLS>
            <<<dim3(N / COLS, ns), COLS * LANES, 0, s>>>(d, W, sgn, f);
    };
    return L;
}

// lanes and regs fix the arithmetic and are the same as the Metal backend's, so
// the two produce identical numbers. rows and cols are pure blocking and are
// chosen for this hardware instead: 32 KB of shared memory per block, which
// every target supports without opting in, and at least 128 threads.
inline bool pickFFTShape(int ng, int &lanes, int &regs, int &rows, int &cols)
{
    switch (ng) {
    case 64:   lanes =  8; regs =  8; rows = 32; cols = 32; return true;
    case 128:  lanes =  8; regs = 16; rows = 32; cols = 32; return true;
    case 256:  lanes = 16; regs = 16; rows = 16; cols = 16; return true;
    case 512:  lanes = 16; regs = 32; rows =  8; cols =  8; return true;
    case 1024: lanes = 32; regs = 32; rows =  4; cols =  4; return true;
    default:   return false;
    }
}

inline bool buildLaunch(int ng, FFTLaunch &L)
{
    switch (ng) {
    case 64:   L = makeLaunch<64, 8, 8, 32, 32>();     return true;
    case 128:  L = makeLaunch<128, 8, 16, 32, 32>();   return true;
    case 256:  L = makeLaunch<256, 16, 16, 16, 16>();  return true;
    case 512:  L = makeLaunch<512, 16, 32, 8, 8>();    return true;
    case 1024: L = makeLaunch<1024, 32, 32, 4, 4>();   return true;
    default:   return false;
    }
}

// Nearest supported size, for the error message. Ties go to the larger grid,
// because dropping resolution silently is the worse surprise.
inline int nearestSupported(int ng)
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

#endif
