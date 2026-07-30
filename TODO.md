# TODO

## 1. The four upstream pull requests

All are open **as drafts** against `svenreiche/Genesis-1.3-Version4`, ready for you to edit and
then mark ready for review. Take #272 out of draft first — #273's body references it, because
the three-FFT branch it touches is unreachable without the filter fix. #274 and #275 are
independent of both and of each other, and can go at any time.

| PR | branch | commit | title |
|---|---|---|---|
| [#272](https://github.com/svenreiche/Genesis-1.3-Version4/pull/272) | `fix/source-filter-noop` | `4fe13fe` | Fix `source_filter` being silently ignored in `FieldSolverFFT` |
| [#273](https://github.com/svenreiche/Genesis-1.3-Version4/pull/273) | `perf/fft-two-pass` | `e101b30` | `FieldSolverFFT`: drop the source-term FFT when the source filter is off |
| [#274](https://github.com/svenreiche/Genesis-1.3-Version4/pull/274) | `perf/user-diagnostic-guard` | `b815d7a` | Skip the user diagnostic's per-particle work when its output is disabled |
| [#275](https://github.com/svenreiche/Genesis-1.3-Version4/pull/275) | `fix/lattice-seed-key` | `b997669` | Fix an uncaught exception when `&lattice` sets a seed |

The bodies as submitted are `PR-1-source-filter-fix.md` and `PR-2-fft-two-pass.md`. Edit them on
GitHub, not here — these files are only the record of what was sent.

#275 was built and tested in a worktree at `/tmp/g4seed` (branch off `master`, own `build/`),
which is still there if Sven asks for changes. The same fix is on `gpu/metal-engine` as
`4ef0989`, with identical content, so the branches will merge cleanly.

### Possible fifth report: `transient` in `&wake` appears to do nothing

`Collective::update` computes `icut`, the catch-up cut for the transient wake, in step 3, but
the loop in step 4 starts at `i = 0` and never reads it. Only the commented-out older loop
below it used `icut`. Measured on a 500-slice deck: `transient = true` costs 2.6x the runtime,
because the convolution is redone every step, and produces a result identical to
`transient = false` to 1.6e-16 in `Beam/energy` and exactly zero in `Beam/wakefield`.
This needs confirming against Sven's intent before reporting, since fixing it would change
results for anyone currently setting the flag.

| other branch | commit | state |
|---|---|---|
| `wip/fp32-precision` | `8f456e6` | local only, parked |
| `tmp/merge-check` | `56781b4` | local scratch, safe to delete |

## 2. GPU work (Apple Silicon / Metal, FP32)

Prototypes are preserved in `~/Code/genesis4-gpu-proto/`.

Work happens on branch `gpu/metal-engine`. Build with
`cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DENABLE_METAL=ON`
and switch on per `&track` block with `gpu = true` (plus `gpu_validate = true` to run the CPU
path alongside and report the difference).

- [x] Metal FFT field solver — radix-16, **14.4 µs/slice-step, 365 GB/s (91% of peak)**
- [x] Engine scaffolding, upload/download round trip, GPU field solve, GPU beam step
- [x] **Backend-neutral seam** — `GPUEngine` is the interface the tracking loop talks to and
      `MetalEngine` implements it. `Gencore.cpp` has no `#ifdef` and nothing device specific
      left in it, so a CUDA or HIP backend is a subclass plus one branch in
      `GPUEngine::create()`. `GPUEngine.h` records the constraints such a backend inherits.
- [x] Namelist control (`gpu`, `gpu_validate`) and hard errors for unported physics
- [x] **Validation matrix** — `examples/metal-gpu/sweep.py`, about 52 cases over undulator
      geometry, transport, lattice errors, grid size, harmonics, beam parameters, collective
      effects, time dependence and every refusal. It found four defects; see below.
- [x] **GPU diagnostic reductions** — the diagnostics were 76% of the GPU runtime
- [x] **Full residency** — no per-step marshalling; the host only sees the arrays at dumps,
      at the slippage boundary slice, and at the end of `&track`

**Measured (500 slices, zstop=10, ngrid=256, 1 harmonic, output_step=1, M3 Max, 12
performance cores, 40-core GPU). End-to-end wall clock:**

| config | wall | vs 1 CPU core | vs 12 CPU cores |
|---|---|---|---|
| CPU 1 rank | 309.1 s | 1× | — |
| CPU 12 ranks | 35.2 s | 8.8× | 1× |
| GPU 1 rank | **5.5 s** | **56×** | **6.4×** |
| GPU 12 ranks (one GPU) | **4.4 s** | **70×** | **8.0×** |

About 1 s of every one of those is setup and output, on both paths, so on the tracking loop
alone the GPU is 4.1 s against about 34 s for twelve ranks, i.e. **8×**.

Extra MPI ranks against the single GPU are worth **nothing**: the tracking loop is 4.1 s at
one rank and 4.1 s at twelve, and the device is already 92% busy at one rank.

**Every GPU timing before this was overstated by about 4×.** Genesis' `Total Wall Clock
Time` was `clock()`, i.e. processor time, which equals the wall clock for a CPU run that
never waits and does not for a GPU run that does: waiting costs no CPU and the shader
compile runs in another process. `GenMain.cpp` now uses `std::chrono::steady_clock`, and a
GPU `&track` block reports the loop's wall clock and the device's own busy time beside it.
**A speedup measured with a timer that does not count waiting is not a speedup.**

Agreement with a rank-matched CPU reference: `Field/power` 1.6e-04, `Field/xsize` 4.8e-05,
`Beam/bunching` 2.7e-04, `Beam/energyspread` 3.7e-06 — the FP32 level throughout.

### What the validation sweep found

Four defects, three of them in the GPU path and one upstream:

1. `&lattice` with a `seed` key aborts. See the fourth report above.
2. `gpu_validate` silently discarded any step the GPU refused. The comparison measured the
   step the GPU had not taken, and then the download overwrote the correct CPU result with
   the un-advanced GPU state, so a chicane's R56 or a corrector's kick was lost without a
   word. Caught as a beam error of 0.96 on the `chicane` case.
3. `source_filter`, `fft_fieldsolver = false` and `bunchharm > 8` were accepted and ignored.
   Each is now a hard error naming the keyword.
4. The `ngrid = 1024` per-step field difference of 8.5e-03 is source granularity, not the
   grid. Holding the grid and raising `npart` gives 8.5e-03, 3.2e-03, 1.2e-03 at 8192, 32768
   and 131072, so the rule is to scale `npart` with `ngrid`.

Two lessons about the harness itself are worth keeping:

- `&time` has to come before `&field` and `&beam`. Put later, Genesis reports nothing and
  runs steady state. The tell was that the time-dependent errors were byte-identical to the
  steady-state ones; **identical numbers across supposedly different configurations mean the
  configuration did not take effect.**
- The end-to-end tier cannot use a fixed tolerance, and the obvious control does not work.
  Scaling the seed power by one part in 1e7 moves the answer by one part in 1e6, because a
  uniform scale is nearly an eigenmode of the amplifier. Raising `npart` sixteenfold does not
  move the end-to-end difference either, so it is not granularity. What does work is the step
  tier's own measurement: the largest single-step difference times the number of steps is the
  most that round-off alone can produce, and every case stays under it. `run_corrector` is the
  closest at 1.3x the bound.
- **A case that only fails on one machine is still a real failure.** On the M3 Max
  `run_two_track_blocks` came out at 2.2x the bound, where the M1 Max had it at 1.3x. It was
  not the machine and not a regression — the same commit failed the same way — but a
  pre-existing upstream defect that the GPU makes unavoidable: the on-axis near-field
  diagnostic samples cell `(ngrid*ngrid-1)/2`, which is the axis only for an odd `ngrid`. The
  GPU only takes powers of two, so every GPU run was reporting a cell at the edge of the grid,
  five orders of magnitude down, where the FP32 difference is naturally percent-level. Fixed
  in `Diagnostic.cpp`, `Field.cpp` and the backend at once, and the case drops to 2.4e-03.

### Still to do

- [x] Generalise `ngrid` beyond 256. The shader is now specialised per grid size through
      preprocessor macros and supports every power of two from 64 to 1024; anything else is a
      hard error naming the nearest supported size.
- [x] Wakefields (`&wake`), including the resistive wall. A wake is driven by the slice
      current rather than by individual particles, so the loss is one number per slice. The
      host keeps the `MPI_Allgather` and the convolution, exactly as on the CPU path, and the
      GPU adds the per-slice kick. Residency is preserved because slice currents do not change
      during a run. Costs under 3% with `transient = false`. GPU vs CPU at 4 ranks:
      `Beam/energy` 4.0e-10, `Beam/bunching` 3.5e-04, `Field/power` 1.7e-04, against a wake
      effect of 4.0e-05, 8.5e-03 and 2.8e-03 respectively.
- [ ] Port `Incoherent` and short-range space charge (hard error today)
- [x] **Port the corrector kick.** `MetalEngine::beamStep` used to refuse any step with a
      non-zero `cx` or `cy`, and `orbiterror = true` is implemented as a per-step corrector,
      so a deck with undulator orbit errors put 187 of 196 steps back on the CPU and got
      almost no benefit from the GPU. The kick is a constant addition to the momenta, so it
      is now two more fields in `TrkPar` applied at the head of `track_beam`, with no extra
      dispatch. It has to be added before gamma*beta_z is formed and only on the closing half
      step, which is where `TrackBeam::track` calls `applyCorrector`. The `orbit_error` case
      now stays on the GPU for all 196 steps at a per-step field difference of 4.6e-04, and
      the sweep treats any unexpected fallback as a failure so that a future regression here
      cannot hide behind a correct answer. On a 400-slice orbit-error deck at 8 ranks the run
      goes from 25.2 s to 9.4 s against 46.2 s on the CPU, so the speedup over the CPU rises
      from 1.8x to 4.9x. A refused step costs more than a CPU step, because the particles and
      the field have to make the round trip either side of it.
- [x] **Port the chicane.** Two pieces: the 4x4 transfer map on the opening half step and the
      R56 phase shear between the collective kick and the closing one. The matrix is per-step
      scalars rather than per-particle work, so it is built on the host by
      `TrackBeam::chicaneMatrix`, which is the CPU path's own construction made static and
      called from both. Duplicating it would have been the obvious way for the two paths to
      drift apart later. The map needs its own gamma*beta_z, which leaves out `aw` because a
      chicane sits outside the undulator. No lattice element falls back to the CPU any more;
      the machinery is kept for the next unported one. New end-to-end cases `run_chicane` and
      `run_corrector` cover residency, which the step tier cannot see because it
      resynchronises either side of every step.
- [ ] `one4one` is refused outright (the GPU wants a rectangular particle array)
- [x] Host-side diagnostic assembly. It was ~5 s of the 7.4 s single-rank time. Most of it was
      not `storeValue` at all but `DiagBeamUser::getValues`, which evaluated a `sin`/`cos` per
      particle per slice per step for an output that is disabled by default and then discarded
      the result. Guarding that, plus hoisting the map lookups out of the slice loops in
      `DiagBeam` and `DiagField`, took `output_step = 1` from 7.33 s to 1.57 s. The same code
      runs in a CPU-only build, where a profile puts `DiagBeamUser` at 2.0% and all of
      `Diagnostic::calc` at 17.9% of the run — worth having, but below the run-to-run noise of
      a 370 s job. Should go upstream on its own branch off master.
- [x] **The diagnostics are the remaining 38% of the tracking loop** (`output_step = 100`
      takes the loop from 4.1 s to 2.6 s). They are already on the device and already skipped
      on steps that produce no output, so what is left is to move less memory. The far-field
      branch used to copy each slice and transform the copy in place; the row pass now reads
      the field and writes the scratch buffer directly, which removed a read and a write of
      every grid point and was worth 8% of the whole loop. Going further means either keeping
      the transform as magnitudes, which costs another buffer a third the size of the field, or
      raw instead of centred second moments, which loses precision exactly when the beam is off
      axis. Neither trade looked worth making.
- [x] Measured the source deposition at 6% of the loop by disabling it, which settles the
      question of the tiled deposition: it cannot repay a threadgroup accumulation and the
      `ngrid % 32` constraint that comes with it, whatever a microbenchmark of the kernel alone
      says. The atomics version stays.
- [ ] `fieldStep`, `beamStep` and each `fieldMoments` call commit their own command buffer and
      wait. At 92% device busy this costs little on a 500-slice deck, but on a small one it is
      most of the time (`validate.in` runs at 16% busy). Merging a step into one command buffer
      would help those, and needs double-buffering of the small host-written arrays.

### Constraints that must not be forgotten

- Apple GPUs have **no FP64**. Metal has no `double`. FP32 is mandatory, not a choice.
- **`gamma` must be stored as an offset from a reference energy.** At γ₀ = 11357 the FP32
  quantum of absolute γ is 1.35e-3, larger than the per-step energy change (3.0e-4 at
  saturation, 5.5e-7 at seed). Storing absolute γ in FP32 gives 65%–276% error.
- Only the FFT field solver was converted to FP32. **ADI remains FP64-only**, so any FP32 or
  GPU run must set `fft_fieldsolver = true` in `&track`. Easy footgun.
- Metal has **no threadgroup `atomic_float`** — `atomic_float` exists in the device address
  space only. Threadgroup float accumulation needs `atomic_uint` + a CAS loop on the bit
  pattern (cheap in local memory).
- The tiled deposition requires `ngrid % 32 == 0`, one more reason to prefer 256 over 255.

## 3. Housekeeping / possible upstream reports

Two of these are now fixed on `gpu/metal-engine` and are worth their own branches off
`master`, since neither has anything to do with the GPU:

- **`Total Wall Clock Time` was `clock()`**, i.e. processor time, not wall clock. Identical
  for a CPU run that never waits; four times too small for one that waits on a device, and it
  also ignores every rank but the zeroth. `GenMain.cpp` now uses `std::chrono::steady_clock`.
- **The on-axis near-field diagnostic samples the wrong cell for an even `ngrid`.**
  `(ngrid*ngrid-1)/2` is the axis only when `ngrid` is odd; for an even grid it is the last
  column of the row below the middle, i.e. the edge of the grid, where the field is orders of
  magnitude smaller. `intensity-nearfield` and `phase-nearfield` are then reporting a corner of
  the box. `(ngrid/2)*ngrid + ngrid/2` is identical for odd `ngrid` and right for even, and is
  now used in `Diagnostic.cpp`, `Field.cpp` and the backend. Nobody noticed because the
  traditional Genesis convention is an odd `ngrid`; the GPU only takes powers of two.

- `export FI_PROVIDER=tcp` is required on macOS with conda MPICH — the default libfabric
  provider busy-polls and costs **5.9× at 8 ranks** (96.0 s → 16.3 s). Worth a manual note.
- `ngrid = 255 = 3 × 5 × 17` is a pathological FFT length: 59.7 s vs 38.3 s at 256 on the same
  case. Worth a manual note or a warning at startup.
- `CMakeLists.txt` forces `/opt/local/bin/h5pcc` as the C++ compiler if that file exists,
  silently overriding `CMAKE_CXX_COMPILER`. Surprising on any machine with MacPorts.
- If FFTW is not found, `FieldSolverFFT` silently produces wrong results rather than failing
  the build — the `fftw_execute` calls are inside `#ifdef FFTW` but the surrounding copy loops
  are not.
