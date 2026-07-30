// Backend selection for the GPU tracking loop.
//
// This file is always compiled, so that Gencore can be written without
// preprocessor conditionals: with no backend built in, create() simply reports
// why there is nothing to run on and the tracking loop takes the CPU path it
// always took. Adding a backend means adding one branch here.

#include "GPUEngine.h"

#ifdef G4_METAL
  #include "MetalEngine.h"
#endif

std::string GPUEngine::backend()
{
#ifdef G4_METAL
    return "Metal";
#else
    return std::string();
#endif
}

int GPUEngine::maxBunchHarm()
{
#ifdef G4_METAL
    return MetalEngine::maxBunchHarm();
#else
    return 0;
#endif
}

GPUEngine *GPUEngine::create(std::string &reason)
{
    reason.clear();
#ifdef G4_METAL
    if (!MetalEngine::available()) {
        reason = "no Metal device with unified memory";
        return nullptr;
    }
    return new MetalEngine();
#else
    reason = "this binary was built without the GPU backend. Reconfigure with "
             "-DENABLE_METAL=ON";
    return nullptr;
#endif
}
