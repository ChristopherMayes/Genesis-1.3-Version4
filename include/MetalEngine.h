#ifndef __GENESIS_METALENGINE__
#define __GENESIS_METALENGINE__

#include <string>
#include <vector>

#include "GPUEngine.h"

class Beam;
class Field;
class Undulator;

// Apple Silicon (Metal) implementation of GPUEngine. See GPUEngine.h for what
// the interface requires of any backend and why; this header only records what
// is specific to Metal.
//
// FP32 is not a choice: Apple GPUs have no FP64 and Metal has no 'double' type.
// gamma is therefore stored as an offset from a reference energy, because at
// gamma0 = 11357 the FP32 quantum of absolute gamma is 1.35e-3, larger than the
// per-step energy change (3.0e-4 at saturation, 5.5e-7 at seed).
//
// Only the FFT field solver was converted, so a GPU run has to set
// fft_fieldsolver = true; ADI stays FP64 on the host.
class MetalEngine : public GPUEngine {
  public:
    MetalEngine();
    ~MetalEngine() override;
    MetalEngine(const MetalEngine &) = delete;
    MetalEngine &operator=(const MetalEngine &) = delete;

    // True if a Metal device with unified memory is present. Safe to call on
    // any machine. A discrete GPU would need staging copies, which is exactly
    // the cost the resident design exists to avoid.
    static bool available();

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
