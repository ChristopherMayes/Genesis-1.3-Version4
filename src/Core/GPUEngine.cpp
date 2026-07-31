// Backend selection for the GPU tracking loop.
//
// This file is always compiled, so that Gencore can be written without
// preprocessor conditionals: with no backend built in, create() simply reports
// why there is nothing to run on and the tracking loop takes the CPU path it
// always took. Adding a backend means adding one branch here.
//
// Metal and CUDA are mutually exclusive in practice -- one is macOS only and
// the other is not available there -- but nothing below assumes that, and a
// build with both compiled in would prefer CUDA only if Metal found no device.

#include "GPUEngine.h"

#ifdef G4_METAL
  #include "MetalEngine.h"
#endif
#ifdef G4_CUDA
  #include "CudaEngine.h"
#endif

#include <algorithm>

std::string GPUEngine::backend()
{
#if defined(G4_METAL) && defined(G4_CUDA)
    return "Metal+CUDA";
#elif defined(G4_METAL)
    return "Metal";
#elif defined(G4_CUDA)
    return "CUDA";
#else
    return std::string();
#endif
}

int GPUEngine::maxBunchHarm()
{
    int h = 0;
#ifdef G4_METAL
    h = std::max(h, MetalEngine::maxBunchHarm());
#endif
#ifdef G4_CUDA
    h = std::max(h, CudaEngine::maxBunchHarm());
#endif
    return h;
}

GPUEngine *GPUEngine::create(std::string &reason)
{
    reason.clear();
#ifdef G4_METAL
    if (MetalEngine::available()) {
        return new MetalEngine();
    }
    reason = "no Metal device with unified memory";
#endif
#ifdef G4_CUDA
    {
        std::string why;
        if (CudaEngine::available(why)) {
            reason.clear();
            return new CudaEngine();
        }
        reason = why;
    }
#endif
#if !defined(G4_METAL) && !defined(G4_CUDA)
    reason = "this binary was built without the GPU backend. Reconfigure with "
             "-DENABLE_CUDA=ON for an NVIDIA card or -DENABLE_METAL=ON on an "
             "Apple Silicon Mac";
#endif
    return nullptr;
}
