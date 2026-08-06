// Standalone check of the FFT kernels in src/Core/MetalEngine.mm.
// Extracts the MSL string from the source file so that it always tests the
// shader that Genesis actually runs, compiles it for a given shape, and
// compares one row against a direct DFT.
//
// build: clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//          -O2 fftcheck.mm -o fftcheck
// run:   ./fftcheck <ngrid> <lanes> <regs> <rowsPerTG> <colsPerTG>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string loadMSL(const char *path)
{
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string all = ss.str();
    const std::string open = "@R\"MSL(";
    size_t a = all.find(open);
    size_t b = all.find(")MSL\"", a);
    if (a == std::string::npos || b == std::string::npos) {
        fprintf(stderr, "could not find the MSL string\n");
        exit(2);
    }
    return all.substr(a + open.size(), b - a - open.size());
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s ngrid lanes regs rows cols\n", argv[0]);
        return 2;
    }
    const int N = atoi(argv[1]);
    const int lanes = atoi(argv[2]);
    const int regs = atoi(argv[3]);
    const int rows = atoi(argv[4]);
    const int cols = atoi(argv[5]);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        std::string src = loadMSL("src/Core/MetalEngine.mm");
        MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
        opt.preprocessorMacros = @{
            @"NG" : @(N), @"LANES" : @(lanes), @"REGS" : @(regs),
            @"CHUNK" : @(regs / lanes), @"RF_ROWS" : @(rows), @"CC_COLS" : @(cols),
        };
        NSError *err = nil;
        id<MTLLibrary> lib =
            [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                              options:opt
                                error:&err];
        if (lib == nil) {
            printf("compile failed: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLComputePipelineState> pRow =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fft_rows"]
                                               error:&err];
        id<MTLComputePipelineState> pCol =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fft_cols"]
                                               error:&err];
        if (pRow == nil || pCol == nil) {
            printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        const size_t nn = (size_t)N * N;
        id<MTLBuffer> bD = [dev newBufferWithLength:nn * 2 * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bW = [dev newBufferWithLength:N * 2 * sizeof(float)
                                            options:MTLResourceStorageModeShared];

        std::complex<float> *W = (std::complex<float> *)[bW contents];
        for (int m = 0; m < N; m++) {
            const double a = -2.0 * M_PI * m / N;
            W[m] = std::complex<float>((float)cos(a), (float)sin(a));
        }

        std::complex<float> *D = (std::complex<float> *)[bD contents];
        srand(12345);
        std::vector<std::complex<double> > in0(N), in0c(N);
        for (size_t i = 0; i < nn; i++) {
            D[i] = std::complex<float>((float)drand48() - 0.5f, (float)drand48() - 0.5f);
        }
        // keep copies of row 1 and column 1 for the reference transforms
        for (int i = 0; i < N; i++) {
            in0[i] = std::complex<double>(D[1 * N + i]);
            in0c[i] = std::complex<double>(D[(size_t)i * N + 1]);
        }

        float sgn = 1.0f, scale = 1.0f;   // +1 is the forward transform
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pRow];
        [e setBuffer:bD offset:0 atIndex:0];
        [e setBuffer:bW offset:0 atIndex:1];
        [e setBytes:&sgn length:4 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(N / rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(rows * lanes, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        double worst = 0, scaleRef = 0;
        for (int k = 0; k < N; k++) {
            std::complex<double> ref(0, 0);
            for (int n = 0; n < N; n++) {
                const double a = -2.0 * M_PI * n * k / N;
                ref += in0[n] * std::complex<double>(cos(a), sin(a));
            }
            const std::complex<double> got(D[1 * N + k]);
            worst = std::max(worst, std::abs(got - ref));
            scaleRef = std::max(scaleRef, std::abs(ref));
        }
        printf("N=%-5d %2dx%-2d rows: max abs err %.3e  (scale %.3e)  -> %s\n", N,
               lanes, regs, worst, scaleRef,
               (worst < 1e-3 * scaleRef) ? "ok" : "FAIL");

        // columns, on a fresh buffer
        for (size_t i = 0; i < nn; i++) {
            D[i] = std::complex<float>((float)drand48() - 0.5f, (float)drand48() - 0.5f);
        }
        for (int i = 0; i < N; i++) in0c[i] = std::complex<double>(D[(size_t)i * N + 1]);

        cb = [q commandBuffer];
        e = [cb computeCommandEncoder];
        [e setComputePipelineState:pCol];
        [e setBuffer:bD offset:0 atIndex:0];
        [e setBuffer:bW offset:0 atIndex:1];
        [e setBytes:&sgn length:4 atIndex:2];
        [e setBytes:&scale length:4 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(N / cols, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(cols * lanes, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        worst = 0;
        scaleRef = 0;
        for (int k = 0; k < N; k++) {
            std::complex<double> ref(0, 0);
            for (int n = 0; n < N; n++) {
                const double a = -2.0 * M_PI * n * k / N;
                ref += in0c[n] * std::complex<double>(cos(a), sin(a));
            }
            const std::complex<double> got(D[(size_t)k * N + 1]);
            worst = std::max(worst, std::abs(got - ref));
            scaleRef = std::max(scaleRef, std::abs(ref));
        }
        printf("N=%-5d %2dx%-2d cols: max abs err %.3e  (scale %.3e)  -> %s\n", N,
               lanes, regs, worst, scaleRef,
               (worst < 1e-3 * scaleRef) ? "ok" : "FAIL");

        // Full four-pass solve as an identity: with expK = 1 and src = 0 the
        // round trip rows(fwd), cols(fwd), rows_mul(inv), cols_add(inv) must
        // reproduce the input. This is what Genesis actually dispatches.
        id<MTLComputePipelineState> pRowM =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fft_rows_mul"]
                                               error:&err];
        id<MTLComputePipelineState> pColA =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fft_cols_add"]
                                               error:&err];
        id<MTLBuffer> bK = [dev newBufferWithLength:nn * 2 * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bS = [dev newBufferWithLength:nn * 2 * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        std::complex<float> *K = (std::complex<float> *)[bK contents];
        std::complex<float> *S = (std::complex<float> *)[bS contents];
        for (size_t i = 0; i < nn; i++) {
            K[i] = std::complex<float>(1.0f, 0.0f);
            S[i] = std::complex<float>(0.0f, 0.0f);
        }
        std::vector<std::complex<float> > before(nn);
        for (size_t i = 0; i < nn; i++) {
            D[i] = std::complex<float>((float)drand48() - 0.5f, (float)drand48() - 0.5f);
            before[i] = D[i];
        }

        const float inv = -1.0f, nrm = 1.0f / (float)nn, one = 1.0f;
        cb = [q commandBuffer];
        e = [cb computeCommandEncoder];
        const MTLSize rowTG = MTLSizeMake(N / rows, 1, 1), rowT = MTLSizeMake(rows * lanes, 1, 1);
        const MTLSize colTG = MTLSizeMake(N / cols, 1, 1), colT = MTLSizeMake(cols * lanes, 1, 1);

        [e setComputePipelineState:pRow];
        [e setBuffer:bD offset:0 atIndex:0];
        [e setBuffer:bW offset:0 atIndex:1];
        [e setBytes:&sgn length:4 atIndex:2];
        [e dispatchThreadgroups:rowTG threadsPerThreadgroup:rowT];

        [e setComputePipelineState:pCol];
        [e setBuffer:bD offset:0 atIndex:0];
        [e setBuffer:bW offset:0 atIndex:1];
        [e setBytes:&sgn length:4 atIndex:2];
        [e setBytes:&one length:4 atIndex:3];
        [e dispatchThreadgroups:colTG threadsPerThreadgroup:colT];

        [e setComputePipelineState:pRowM];
        [e setBuffer:bD offset:0 atIndex:0];
        [e setBuffer:bW offset:0 atIndex:1];
        [e setBytes:&inv length:4 atIndex:2];
        [e setBuffer:bK offset:0 atIndex:3];
        [e dispatchThreadgroups:rowTG threadsPerThreadgroup:rowT];

        [e setComputePipelineState:pColA];
        [e setBuffer:bD offset:0 atIndex:0];
        [e setBuffer:bW offset:0 atIndex:1];
        [e setBytes:&inv length:4 atIndex:2];
        [e setBytes:&nrm length:4 atIndex:3];
        [e setBuffer:bS offset:0 atIndex:4];
        [e dispatchThreadgroups:colTG threadsPerThreadgroup:colT];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        worst = 0;
        scaleRef = 0;
        for (size_t i = 0; i < nn; i++) {
            worst = std::max(worst, (double)std::abs(D[i] - before[i]));
            scaleRef = std::max(scaleRef, (double)std::abs(before[i]));
        }
        printf("N=%-5d %2dx%-2d solve: max abs err %.3e  (scale %.3e)  -> %s\n", N,
               lanes, regs, worst, scaleRef,
               (worst < 1e-4 * scaleRef) ? "ok" : "FAIL");
    }
    return 0;
}
