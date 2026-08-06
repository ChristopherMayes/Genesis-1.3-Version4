// Standalone check of the FFT kernels used by the CUDA backend.
//
// It includes src/Core/CudaFFT.cuh, which is the same header CudaEngine.cu
// compiles, so it always tests the transform Genesis actually runs rather than
// a copy of it. Three things are checked against a direct double-precision DFT:
// a row pass, a column pass, and the complete four-pass solve run as an
// identity.
//
// This is a diagnostic rather than a regression test and is not part of the
// build. It is what to reach for when the sweep says the field is wrong but not
// where.
//
// build: nvcc -std=c++17 -O2 -arch=native -Isrc/Core tools/fftcheck.cu -o fftcheck
// run:   ./fftcheck            # every shape the backend supports
//        ./fftcheck 256        # just one

#include "CudaFFT.cuh"

#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call)                                                               \
    do {                                                                       \
        const cudaError_t e_ = (call);                                         \
        if (e_ != cudaSuccess) {                                               \
            std::printf("%s failed: %s\n", #call, cudaGetErrorString(e_));     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

std::vector<std::complex<float> > randomField(size_t n)
{
    std::vector<std::complex<float> > v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = std::complex<float>(static_cast<float>(drand48()) - 0.5f,
                                   static_cast<float>(drand48()) - 0.5f);
    }
    return v;
}

// Direct DFT of one line, in double precision, as the reference.
std::vector<std::complex<double> > dftRef(const std::vector<std::complex<double> > &in)
{
    const int N = static_cast<int>(in.size());
    std::vector<std::complex<double> > out(N);
    for (int k = 0; k < N; k++) {
        std::complex<double> s(0, 0);
        for (int n = 0; n < N; n++) {
            const double a = -2.0 * M_PI * n * k / N;
            s += in[n] * std::complex<double>(cos(a), sin(a));
        }
        out[k] = s;
    }
    return out;
}

bool report(const char *what, int N, int lanes, int regs, double worst,
            double scale, double tol)
{
    const bool ok = worst < tol * scale;
    std::printf("N=%-5d %2dx%-2d %-6s max abs err %.3e  (scale %.3e)  -> %s\n", N,
                lanes, regs, what, worst, scale, ok ? "ok" : "FAIL");
    return ok;
}

bool checkShape(int N)
{
    int lanes = 0, regs = 0, rows = 0, cols = 0;
    if (!pickFFTShape(N, lanes, regs, rows, cols)) {
        std::printf("N=%d is not a supported grid size\n", N);
        return false;
    }
    FFTLaunch L;
    if (!buildLaunch(N, L)) {
        std::printf("N=%d has no instantiated transform\n", N);
        return false;
    }

    const size_t nn = static_cast<size_t>(N) * N;
    float2 *dD = nullptr, *dW = nullptr, *dK = nullptr, *dS = nullptr;
    CK(cudaMalloc(&dD, nn * sizeof(float2)));
    CK(cudaMalloc(&dW, static_cast<size_t>(N) * sizeof(float2)));
    CK(cudaMalloc(&dK, nn * sizeof(float2)));
    CK(cudaMalloc(&dS, nn * sizeof(float2)));

    std::vector<float2> W(N);
    for (int m = 0; m < N; m++) {
        const double a = -2.0 * M_PI * m / N;
        W[m] = make_float2(static_cast<float>(cos(a)), static_cast<float>(sin(a)));
    }
    CK(cudaMemcpy(dW, W.data(), static_cast<size_t>(N) * sizeof(float2),
                  cudaMemcpyHostToDevice));

    bool ok = true;
    const float fwd = 1.0f, inv = -1.0f, one = 1.0f;
    const float nrm = 1.0f / static_cast<float>(nn);

    // ---- rows ----
    {
        std::vector<std::complex<float> > h = randomField(nn);
        std::vector<std::complex<double> > in(N);
        for (int i = 0; i < N; i++) { in[i] = std::complex<double>(h[1 * N + i]); }
        CK(cudaMemcpy(dD, h.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));
        L.rows(nullptr, 1, dD, dW, fwd);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), dD, nn * sizeof(float2), cudaMemcpyDeviceToHost));

        const std::vector<std::complex<double> > ref = dftRef(in);
        double worst = 0, scale = 0;
        for (int k = 0; k < N; k++) {
            worst = std::max(worst, std::abs(std::complex<double>(h[1 * N + k]) - ref[k]));
            scale = std::max(scale, std::abs(ref[k]));
        }
        ok &= report("rows:", N, lanes, regs, worst, scale, 1e-3);
    }

    // ---- columns ----
    {
        std::vector<std::complex<float> > h = randomField(nn);
        std::vector<std::complex<double> > in(N);
        for (int i = 0; i < N; i++) {
            in[i] = std::complex<double>(h[static_cast<size_t>(i) * N + 1]);
        }
        CK(cudaMemcpy(dD, h.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));
        L.cols(nullptr, 1, dD, dW, fwd, one);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), dD, nn * sizeof(float2), cudaMemcpyDeviceToHost));

        const std::vector<std::complex<double> > ref = dftRef(in);
        double worst = 0, scale = 0;
        for (int k = 0; k < N; k++) {
            worst = std::max(
                worst,
                std::abs(std::complex<double>(h[static_cast<size_t>(k) * N + 1]) - ref[k]));
            scale = std::max(scale, std::abs(ref[k]));
        }
        ok &= report("cols:", N, lanes, regs, worst, scale, 1e-3);
    }

    // ---- the complete solve, as an identity ----
    //
    // With expK = 1 and src = 0 the four passes the field solve dispatches --
    // rows(fwd), cols(fwd), rowsMul(inv), colsAdd(inv) -- have to reproduce the
    // input. This is exactly what Genesis runs every step.
    {
        std::vector<float2> k(nn, make_float2(1.0f, 0.0f));
        std::vector<float2> s(nn, make_float2(0.0f, 0.0f));
        CK(cudaMemcpy(dK, k.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dS, s.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));

        const std::vector<std::complex<float> > before = randomField(nn);
        std::vector<std::complex<float> > after(nn);
        CK(cudaMemcpy(dD, before.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));

        L.rows(nullptr, 1, dD, dW, fwd);
        L.cols(nullptr, 1, dD, dW, fwd, one);
        L.rowsMul(nullptr, 1, dD, dW, inv, dK);
        L.colsAdd(nullptr, 1, dD, dW, inv, nrm, dS);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(after.data(), dD, nn * sizeof(float2), cudaMemcpyDeviceToHost));

        double worst = 0, scale = 0;
        for (size_t i = 0; i < nn; i++) {
            worst = std::max(worst, static_cast<double>(std::abs(after[i] - before[i])));
            scale = std::max(scale, static_cast<double>(std::abs(before[i])));
        }
        ok &= report("solve:", N, lanes, regs, worst, scale, 1e-4);
    }

    // ---- the out-of-place row pass the far-field diagnostic uses ----
    {
        std::vector<std::complex<float> > h = randomField(nn);
        std::vector<std::complex<float> > out(nn, std::complex<float>(0, 0));
        std::vector<std::complex<double> > in(N);
        for (int i = 0; i < N; i++) { in[i] = std::complex<double>(h[1 * N + i]); }
        CK(cudaMemcpy(dK, h.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dD, out.data(), nn * sizeof(float2), cudaMemcpyHostToDevice));
        L.rowsOut(nullptr, 1, dD, dW, fwd, dK);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), dD, nn * sizeof(float2), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h.data(), dK, nn * sizeof(float2), cudaMemcpyDeviceToHost));

        const std::vector<std::complex<double> > ref = dftRef(in);
        double worst = 0, scale = 0;
        for (int j = 0; j < N; j++) {
            worst = std::max(worst, std::abs(std::complex<double>(out[1 * N + j]) - ref[j]));
            scale = std::max(scale, std::abs(ref[j]));
            // and the source must not have been touched
            worst = std::max(worst, std::abs(std::complex<double>(h[1 * N + j]) - in[j]));
        }
        ok &= report("out:", N, lanes, regs, worst, scale, 1e-3);
    }

    cudaFree(dD);
    cudaFree(dW);
    cudaFree(dK);
    cudaFree(dS);
    return ok;
}

}  // namespace

int main(int argc, char **argv)
{
    srand48(12345);
    bool ok = true;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) { ok &= checkShape(atoi(argv[i])); }
    } else {
        static const int all[] = {64, 128, 256, 512, 1024};
        for (int n : all) { ok &= checkShape(n); }
    }
    std::printf("%s\n", ok ? "all shapes ok" : "FAILURES above");
    return ok ? 0 : 1;
}
