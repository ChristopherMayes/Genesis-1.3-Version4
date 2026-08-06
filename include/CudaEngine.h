#ifndef __GENESIS_CUDAENGINE__
#define __GENESIS_CUDAENGINE__

#include <string>
#include <vector>

#include "GPUEngine.h"

class Beam;
class Field;
class Undulator;

// NVIDIA (CUDA) implementation of GPUEngine. See GPUEngine.h for what the
// interface requires of any backend and why; this header only records what is
// specific to CUDA and to a discrete card.
//
// FP32, for the same reasons the Metal backend gives and for one more. On a
// consumer card -- the RTX 5080 this was written on, or an L4 -- double
// precision runs at a sixty-fourth of the single-precision rate and is not
// worth having; on an A100 it runs at half, so a double-precision variant would
// be worth writing, but it would be a second engine rather than a flag on this
// one, because every kernel below is written against the FP32 reformulations.
// gamma is therefore carried as an offset from a reference energy, exactly as
// in the Metal backend, and gammaRef() reports it.
//
// The transform shapes and the arithmetic are identical to the Metal backend's,
// so the two produce the same numbers to the last bit of the rounding, and the
// accuracy figures in manual/GPU.md apply unchanged. What differs is everything
// around the arithmetic:
//
// 1. The memory is not unified. Every buffer is a device allocation with a
//    pinned host staging area beside it, and the round trips that Metal gets
//    for free -- the space-charge analysis, the wake profile, the incoherent
//    draws, the slippage slice -- are explicit copies on the engine's stream.
//    They are listed in manual/GPU.md and none of them is large; each is,
//    however, a synchronisation point, which is why the step still encodes
//    everything it can before draining.
//
// 2. The kernels are specialised at build time rather than at run time. The
//    grid size and the transform shape are template parameters and every
//    supported combination is instantiated by the compiler, so there is no
//    startup cost and no runtime compilation.
//
// 3. The device is chosen from the rank's position within its node, not from
//    its rank in the job, so that a multi-node job spreads over the cards it
//    was given instead of piling every rank onto device 0. See selectDevice()
//    in the implementation.
//
// Only the FFT field solver was converted, so a GPU run has to set
// fft_fieldsolver = true; ADI stays FP64 on the host.
class CudaEngine : public GPUEngine {
  public:
    CudaEngine();
    ~CudaEngine() override;
    CudaEngine(const CudaEngine &) = delete;
    CudaEngine &operator=(const CudaEngine &) = delete;

    // True if the CUDA runtime can see at least one device of an architecture
    // this binary was compiled for. Safe to call on any machine, including one
    // with no driver installed at all.
    static bool available(std::string &reason);

    // Highest bunching harmonic the reduction kernel produces.
    static int maxBunchHarm();

    std::string deviceName() const override;

    bool init(Beam *beam, std::vector<Field *> *field,
              std::string &reason) override;

    void upload(Beam *beam, std::vector<Field *> *field) override;
    void uploadBeam(Beam *beam) override;
    void download(Beam *beam, std::vector<Field *> *field) override;
    void downloadField(std::vector<Field *> *field) override;
    void downloadBeam(Beam *beam) override;

    void downloadFieldSlice(int ifld, int islice, Field *field) override;
    void uploadFieldSlice(int ifld, int islice, const Field *field) override;

    void fieldStep(Undulator *und, std::vector<Field *> *field,
                   double delz) override;
    bool beamStep(Beam *beam, Undulator *und, std::vector<Field *> *field,
                  double delz, std::string &reason) override;

    bool beamMoments(int nharm, bool wantAux,
                     BeamSliceMoments &out) const override;
    bool fieldMoments(int ih, bool wantFar,
                      FieldSliceMoments &out) const override;

    SyncError compare(Beam *beam,
                      std::vector<Field *> *field) const override;

    double gammaRef() const override;
    size_t bytesResident() const override;
    double deviceSeconds() const override;

  private:
    struct Impl;
    Impl *p_;
};

#endif
