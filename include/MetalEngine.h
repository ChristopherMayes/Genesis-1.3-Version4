#ifndef __GENESIS_METALENGINE__
#define __GENESIS_METALENGINE__

#include <string>
#include <vector>

#include "SliceMoments.h"

class Beam;
class Field;
class Undulator;

// GPU backend for Apple Silicon (Metal, FP32).
//
// The engine holds the beam and the field in GPU memory so that they can stay
// resident across integration steps. That residency is the entire point of the
// class: marshalling the host arrays in and out every step was measured at
// 38 ms per step (504 slices, ngrid=256, 8192 particles/slice) against ~11 ms
// of GPU compute, so a solver plugged in behind FieldSolver::advance, which
// would have to copy every call, cannot pay off.
//
// The beam and the field are coupled in both directions every step -- the
// source deposition reads the particles and the Runge-Kutta push gathers the
// field -- so neither can be moved to the GPU without the other.
//
// FP32 is not a choice: Apple GPUs have no FP64 and Metal has no 'double'.
// gamma is therefore stored as an offset from a reference energy, because at
// gamma0 = 11357 the FP32 quantum of absolute gamma is 1.35e-3, larger than the
// per-step energy change (3.0e-4 at saturation, 5.5e-7 at seed).
class MetalEngine {
  public:
    MetalEngine();
    ~MetalEngine();
    MetalEngine(const MetalEngine &) = delete;
    MetalEngine &operator=(const MetalEngine &) = delete;

    // True if a Metal device with unified memory is present. Safe to call on
    // any machine; on a build without the Metal backend it returns false.
    static bool available();
    static std::string deviceName();

    // Highest bunching harmonic the diagnostic reduction can produce. Above it
    // the reduction has no answer, and because the host arrays are stale while
    // the beam is resident there is nothing to fall back on, so &track has to
    // refuse the deck rather than report bunching computed from old particles.
    static int maxBunchHarm();

    // Allocates the resident buffers for this run. Returns false and fills
    // 'reason' if the configuration is not supported, in which case the caller
    // must keep using the CPU path.
    bool init(Beam *beam, std::vector<Field *> *field, std::string &reason);

    // Full state transfers. These are for setup, teardown and field dumps --
    // not for the inner loop.
    void upload(Beam *beam, std::vector<Field *> *field);
    void uploadBeam(Beam *beam);
    void download(Beam *beam, std::vector<Field *> *field);
    void downloadField(std::vector<Field *> *field);
    void downloadBeam(Beam *beam);

    // Single-slice transfers. Slippage moves exactly one slice per slip event,
    // so it does not need the whole field to come back to the host.
    void downloadFieldSlice(int ifld, int islice, Field *field);
    void uploadFieldSlice(int ifld, int islice, const Field *field);

    // Field solve for one integration step: the equivalent of calling
    // Field::track on every harmonic, but for all slices at once and entirely
    // on the resident buffers. Reads the resident beam, writes the resident
    // field. Blocks until the GPU is done.
    void fieldStep(Undulator *und, std::vector<Field *> *field, double delz);

    // Beam step: transverse half step, Runge-Kutta longitudinal push, second
    // transverse half step. Returns false without touching the GPU state if the
    // step needs something that is not ported yet, in which case the caller
    // must fall back to the CPU. No lattice element does that at present.
    bool beamStep(Beam *beam, Undulator *und, std::vector<Field *> *field,
                  double delz, std::string &reason);

    // Per-slice diagnostic reductions, computed from the resident copies so
    // that no particle or grid array has to cross to the host.
    bool beamMoments(int nharm, bool wantAux, BeamSliceMoments &out) const;
    bool fieldMoments(int ih, bool wantFar, FieldSliceMoments &out) const;

    // Largest relative difference between the host arrays and the resident GPU
    // copy, without modifying either. Checks the marshalling in isolation; both
    // numbers should sit at the FP32 rounding level (~1e-7).
    struct SyncError {
        double field {0};
        double beam {0};
    };
    SyncError compare(Beam *beam, std::vector<Field *> *field) const;

    // Reference energy that the stored gamma offsets are measured from.
    double gammaRef() const;
    // Resident footprint in bytes.
    size_t bytesResident() const;

  private:
    struct Impl;
    Impl *p_;
};

#endif
