#ifndef __GENESIS_GPUENGINE__
#define __GENESIS_GPUENGINE__

#include <string>
#include <vector>

#include "SliceMoments.h"

class Beam;
class Field;
class Undulator;

// Backend-neutral interface to a GPU implementation of the tracking loop.
//
// Gencore talks to this class and to nothing device specific, so a second
// backend is a new subclass plus one line in GPUEngine::create(), with no
// preprocessor conditionals anywhere in the tracking loop. The only
// implementation today is MetalEngine, for Apple Silicon.
//
// Three properties of the design are not Metal specific and any further
// backend has to respect them.
//
// 1. The beam and the field stay resident for the whole of a &track block.
//    Marshalling the host arrays in and out every step was measured at 38 ms
//    per step (504 slices, ngrid=256, 8192 particles/slice) against ~11 ms of
//    GPU compute, so a solver plugged in behind FieldSolver::advance, which
//    would have to copy on every call, cannot pay off. The beam and the field
//    are coupled in both directions every step -- the source deposition reads
//    the particles and the Runge-Kutta push gathers the field -- so neither can
//    be moved to the GPU without the other.
//
// 2. The per-slice diagnostics are reduced on the device. They are 76% of the
//    runtime once the tracking is fast, and they read exactly the arrays that
//    are already there.
//
// 3. Anything the backend cannot reproduce is refused by name, never worked
//    around. A run that completes and writes a plausible output file having
//    quietly done something other than what the deck asked for is the failure
//    mode this interface exists to prevent, which is why beamStep() reports a
//    reason rather than doing its best.
//
// The Metal backend is single precision because Apple GPUs have no FP64 at all.
// That is not part of this interface: a backend with double precision would
// implement the same calls. What is part of the interface is that gammaRef()
// exists, because a single-precision backend cannot store absolute gamma (at
// gamma0 = 11357 the FP32 quantum is 1.35e-3, larger than the per-step energy
// change) and has to carry an offset from a reference energy instead.
class GPUEngine {
  public:
    virtual ~GPUEngine() = default;

    // ---- backend selection, all safe to call on any machine ----

    // Name of the backend compiled into this binary ("Metal"), or an empty
    // string if it was built without one.
    static std::string backend();

    // The engine for this machine, or nullptr with 'reason' filled in. Does not
    // allocate anything; init() does that once the run is set up.
    static GPUEngine *create(std::string &reason);

    // Highest bunching harmonic the diagnostic reduction can produce, or 0 if
    // no backend is compiled in. Above it the reduction has no answer, and
    // because the host arrays are stale while the beam is resident there is
    // nothing to fall back on, so &track has to refuse the deck rather than
    // report bunching computed from old particles.
    static int maxBunchHarm();

    // Device this engine is running on, for the log line.
    virtual std::string deviceName() const = 0;

    // ---- setup ----

    // Allocates the resident buffers for this run. Returns false and fills
    // 'reason' if the configuration is not supported.
    virtual bool init(Beam *beam, std::vector<Field *> *field,
                      std::string &reason) = 0;

    // ---- full state transfers, for setup, teardown and field dumps ----

    virtual void upload(Beam *beam, std::vector<Field *> *field) = 0;
    virtual void uploadBeam(Beam *beam) = 0;
    virtual void download(Beam *beam, std::vector<Field *> *field) = 0;
    virtual void downloadField(std::vector<Field *> *field) = 0;
    virtual void downloadBeam(Beam *beam) = 0;

    // Single-slice transfers. Slippage moves exactly one slice per slip event,
    // so it does not need the whole field to come back to the host.
    virtual void downloadFieldSlice(int ifld, int islice, Field *field) = 0;
    virtual void uploadFieldSlice(int ifld, int islice, const Field *field) = 0;

    // ---- one integration step ----

    // Field solve: the equivalent of calling Field::track on every harmonic,
    // but for all slices at once and entirely on the resident buffers.
    virtual void fieldStep(Undulator *und, std::vector<Field *> *field,
                           double delz) = 0;

    // Beam step: transverse half step, longitudinal push, collective kicks,
    // second transverse half step. Returns false without touching the resident
    // state if the step needs something that is not ported, in which case the
    // caller must fall back to the CPU. No lattice element does that at
    // present; the path is kept for the next one that is added.
    virtual bool beamStep(Beam *beam, Undulator *und,
                          std::vector<Field *> *field, double delz,
                          std::string &reason) = 0;

    // ---- diagnostics ----

    // Per-slice reductions, computed from the resident copies so that no
    // particle or grid array has to cross to the host.
    virtual bool beamMoments(int nharm, bool wantAux,
                             BeamSliceMoments &out) const = 0;
    virtual bool fieldMoments(int ih, bool wantFar,
                              FieldSliceMoments &out) const = 0;

    // ---- validation and reporting ----

    // Largest relative difference between the host arrays and the resident
    // copy, without modifying either. Checks the marshalling in isolation; on a
    // single-precision backend both numbers should sit at the FP32 rounding
    // level (~1e-7).
    struct SyncError {
        double field {0};
        double beam {0};
    };
    virtual SyncError compare(Beam *beam,
                              std::vector<Field *> *field) const = 0;

    // Reference energy that the stored gamma values are measured from.
    virtual double gammaRef() const = 0;
    // Resident footprint in bytes.
    virtual size_t bytesResident() const = 0;

    // Seconds the device itself spent executing since init(). Compared against
    // the wall clock of the tracking loop this separates a saturated device
    // from a host that is not keeping it fed, which is the question that
    // decides whether more MPI ranks against one GPU will help. Returns 0 if
    // the backend cannot measure it.
    virtual double deviceSeconds() const { return 0; }
};

#endif
