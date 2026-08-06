# TODO

## 1. Upstream pull requests

### Merged

All eight are in `svenreiche/Genesis-1.3-Version4` master as of `5b34ee6`, and our fork's
`master` is synced to it. Nothing is outstanding upstream.

| PR | branch | title |
|---|---|---|
| [#272](https://github.com/svenreiche/Genesis-1.3-Version4/pull/272) | `fix/source-filter-noop` | Fix `source_filter` being silently ignored in `FieldSolverFFT` |
| [#273](https://github.com/svenreiche/Genesis-1.3-Version4/pull/273) | `perf/fft-two-pass` | `FieldSolverFFT`: drop the source-term FFT when the source filter is off |
| [#274](https://github.com/svenreiche/Genesis-1.3-Version4/pull/274) | `perf/user-diagnostic-guard` | Skip the user diagnostic's per-particle work when its output is disabled |
| [#275](https://github.com/svenreiche/Genesis-1.3-Version4/pull/275) | `fix/lattice-seed-key` | Fix an uncaught exception when `&lattice` sets a seed |
| [#276](https://github.com/svenreiche/Genesis-1.3-Version4/pull/276) | `fix/wallclock-timer` | Report the wall clock time rather than the processor time at the end of a run |
| [#277](https://github.com/svenreiche/Genesis-1.3-Version4/pull/277) | `fix/onaxis-cell-even-ngrid` | Take the on-axis field diagnostic from the axis when `ngrid` is even |
| [#278](https://github.com/svenreiche/Genesis-1.3-Version4/pull/278) | `fix/electron-rest-energy` | Take the electron rest energy from one constant, and from CODATA |
| [#279](https://github.com/svenreiche/Genesis-1.3-Version4/pull/279) | `fix/wake-steady-state` | Take the wake slice separation from a slice that exists |

Six of the eight were found by building the GPU validation matrix, or by having to read code
closely enough while porting it to notice. That is the argument for a differential test
against an existing implementation: it finds the existing implementation's bugs as readily as
the new one's.

### What the two rebases did

`gpu/metal-engine` has now been rebased twice, onto `fc046e3` and onto `5b34ee6`, and both
were verified content-preserving by diffing the rebased tree against the tree before it. Worth
continuing to do, because the check is cheap and the failure mode is silent.

The second rebase had four conflicts, all of them the same shape — upstream and the branch
having edited the same line for different reasons:

- `Diagnostic.cpp`: upstream now carries the on-axis index, the branch adds the
  precomputed-moments branch to the same statement. Kept both.
- `Collective.cpp`: upstream's `apply` assigns `beam->eloss`, the branch had already factored
  that into `computeLoss`. Dropped the duplicate.
- `Collective.h`: trailing whitespace. Took upstream's bytes so the file shows no diff.
- `Wake.cpp`: the same fix with a longer comment on the branch. Took upstream's, leaving the
  file identical to master. The diagnosis the shorter comment omits is recorded in section 2
  below rather than lost.

Two commits were left holding only the backend's half of a change whose shared half is now
upstream — the Metal uses of the `511000` literal, and the backend's copy of the on-axis
index. Both were reworded, since a commit titled after the upstream fix but containing two
lines of Metal is worse than no message at all.

**Measuring a "before" build needs a worktree, not a checkout.** `git checkout master` with a
change staged carries it across, so the first attempt at the `eev` measurement built the new
constant into both binaries and reported every dataset bit-identical. What caught it was
searching each binary for the IEEE-754 patterns of the two values; worth repeating whenever a
before-and-after claim comes out as exactly zero.

### Not worth sending

The map-lookup hoisting in `DiagBeam::getValues` and `DiagField::getValues` (part of
`959012c`) is a real speedup on the GPU path, where the reductions are gone and only the
bookkeeping is left, but not on the CPU path. Measured: the whole diagnostic block is 22% of a
12-rank CPU run of the SASE example (33.4 s at `output_step = 1` against 26.2 s at 100), and a
sampling profile puts that time in the `ngrid^2` reduction loops and in FFTW, not in the map
lookups. Porting it upstream would be churn for something below the noise.

### Still to report: `transient` in `&wake` appears to do nothing

Still true on the merged master (`Collective.cpp`, step 4 of `Collective::update`).

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

## 2. GPU work (Metal and CUDA, FP32)

Prototypes are preserved in `~/Code/genesis4-gpu-proto/`.

Work happens on branch `gpu/metal-engine`. Build with
`cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DENABLE_CUDA=ON`
on an NVIDIA machine, or the `-DENABLE_METAL=ON` / `build-metal` equivalent on a Mac, and
switch on per `&track` block with `gpu = true` (plus `gpu_validate = true` to run the CPU
path alongside and report the difference). The CUDA environment is `genesis4-cuda`, which
adds `cuda-nvcc cuda-cudart-dev cuda-version=13.2` to the usual four dependencies.

- [x] Metal FFT field solver — radix-16, **14.4 µs/slice-step, 365 GB/s (91% of peak)**
- [x] Engine scaffolding, upload/download round trip, GPU field solve, GPU beam step
- [x] **Backend-neutral seam** — `GPUEngine` is the interface the tracking loop talks to and
      `MetalEngine` implements it. `Gencore.cpp` has no `#ifdef` and nothing device specific
      left in it, so a CUDA or HIP backend is a subclass plus one branch in
      `GPUEngine::create()`. `GPUEngine.h` records the constraints such a backend inherits.
- [x] Namelist control (`gpu`, `gpu_validate`) and hard errors for unported physics
- [x] **Validation matrix** — `examples/gpu/sweep.py`, about 52 cases over undulator
      geometry, transport, lattice errors, grid size, harmonics, beam parameters, collective
      effects, time dependence and every refusal. It found four defects; see below.
- [x] **GPU diagnostic reductions** — the diagnostics were 76% of the GPU runtime
- [x] **Full residency** — no per-step marshalling; the host only sees the arrays at dumps,
      at the slippage boundary slice, and at the end of `&track`
- [x] **CUDA backend** — `CudaEngine`, second implementation of the same interface, tested on
      an RTX 5080 (sm_120) and built for A100 (80), L4 (89) and Blackwell (120) with PTX for
      anything newer. The kernels are transcriptions of the Metal ones, so the two agree with
      the CPU to the same figures; the port was the memory model and nothing else. All 72
      sweep cases pass. **Tracking loop 1.42 s against the M3 Max's 4.10 s on the 500-slice
      deck**, and 17x all sixteen CPU cores of the same machine.
- [x] **Multi-GPU device selection** — `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`, then
      `local_rank % devices_visible`, with `G4_CUDA_DEVICE` to override and the whole node's
      mapping reported in the backend line. Not yet exercised on a machine with more than one
      card; the selection and the reporting are, at 1, 2 and 4 ranks against one device.

**Measured (500 slices, zstop=10, ngrid=256, 1 harmonic, output_step=1). End-to-end wall
clock, on a Core Ultra 9 285K with 24 cores and an RTX 5080, and on an M3 Max with 12
performance cores and a 40-core GPU:**

| config | 285K + RTX 5080 | M3 Max |
|---|---|---|
| CPU 1 rank | 268.0 s | 309.1 s |
| CPU all cores | 25.6 s (16 ranks) | 35.2 s (12 ranks) |
| GPU 1 rank | **3.6 s** | **5.2 s** |
| GPU 8 / 12 ranks (one GPU) | **2.2 s** | **4.4 s** |

One to two seconds of every one of those is setup and output, on both paths, so on the
tracking loop alone the RTX 5080 is 1.42 s against about 24 s for sixteen CPU ranks
(**17×**) and about 266 s for one (**190×**). Against the M3 Max's 4.10 s it is **2.9×**,
where the bandwidth ratio alone predicted 2.4× before the backend existed.

Extra MPI ranks against a single GPU are worth **nothing**: the tracking loop is 1.42 s at
one rank and 1.38 s at four, and the device is already saturated at one rank. Several
*cards* are worth having, one rank each, and that is what the node-local device selection
above is for.

**Every GPU timing before this was overstated by about 4×.** Genesis' `Total Wall Clock
Time` was `clock()`, i.e. processor time, which equals the wall clock for a CPU run that
never waits and does not for a GPU run that does: waiting costs no CPU and the shader
compile runs in another process. `GenMain.cpp` now uses `std::chrono::steady_clock`, and a
GPU `&track` block reports the loop's wall clock and the device's own busy time beside it.
**A speedup measured with a timer that does not count waiting is not a speedup.**

Agreement with a rank-matched CPU reference: `Field/power` 1.6e-04, `Field/xsize` 4.8e-05,
`Beam/bunching` 2.7e-04, `Beam/energyspread` 3.7e-06 — the FP32 level throughout.

### What the validation sweep found

Six defects across the two ports, three of them in the GPU path and three upstream. The last
two are from the CUDA port and are both cases of a bug that macOS had been hiding:

5. **`&wake` in a steady-state run read one past the end of a one-element vector.**
   `Wake::init` took the slice separation as `s[1]-s[0]`, but `Time::getPosition` returns a
   single slice when there is no `&time` namelist. On macOS the word after it happened to be
   a usable number and the five wakefield sweep cases passed; on Linux it read equal to
   `s[0]`, giving `ds = 0`, then a NaN slice index inside `Collective::update`, then an
   `INT_MIN` cast and an out-of-range abort. Nothing to do with the GPU — the CPU-only path
   aborts identically. Fixed in `Wake.cpp`: a lone slice is one reference length long.
6. **The bunching factors and the auxiliary min/max output shared slots in the diagnostic
   reduction.** Both backends laid the per-slice output out with bunching at slot `12+2h` and
   the auxiliary extrema at 20, which collide for `bunchharm > 4`; the extrema are written
   second, so harmonics five to eight were silently replaced. Found by reading the layout
   while transcribing it, not by the sweep, which does not run `bunchharm = 8` and
   `auxiliar` together. The stride is now 48 and the extrema start at 28, in both backends.

And the original four:

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
- **A case that only fails on one machine is still a real failure.** This came up twice.
  The five wakefield cases above passed on the Mac and aborted on Linux, and the temptation
  was to call it a CUDA regression; it was undefined behaviour that one allocator had been
  answering conveniently. On the M3 Max
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
- [x] **Port `Incoherent`** — both `doLoss` and `doSpread`. The part worth remembering is the
      random numbers. `Incoherent::apply` draws once per *beamlet*, not per particle, and the
      generator is `RandomU`, which is `ran2` from Numerical Recipes: two linear congruential
      sequences behind a Bays-Durham shuffle table, and a shuffle table has no practical
      jump-ahead. So a device-side generator cannot reproduce the sequence, only replace it.
      Rather than accept a different noise realisation on the GPU, the draws are taken on the
      host in the order the CPU consumes them and uploaded one per beamlet per step, which
      keeps the ordinary step-tier comparison working: with radiation on, the beam still
      agrees to 3.2e-06 over 1104 steps, the same as with it off.
      **The backend needs its own generator, seeded identically, not a shared one.** Under
      `gpu_validate` both paths run, so a shared generator would hand the first path one set
      of draws and the second the next set, and the two would disagree by the size of the
      effect while both were correct.
      Cost: the loop goes from 4.64 s to 6.43 s on the 500-slice deck with `doSpread`, the
      device busy time only from 4.32 s to 4.75 s, so the extra 1.5 s is 100 million host
      draws at about 15 ns each. `doLoss` alone skips the draws entirely, because the spread
      scales them by zero, and costs 4.96 s. For scale, the radiation itself changes
      `Field/power` by 25% and `Beam/energyspread` by 8.9%.
- [x] **Port short-range space charge.** The solve is a set of azimuthal and longitudinal
      modes; per mode pair the particles of a slice are binned in radius into a complex
      source, a tridiagonal system is solved on the radial grid, and the answer is gathered
      back. One threadgroup per slice, radial arrays in threadgroup memory, and the
      recurrence run by a single thread while the others wait.
      **`rmax` is the part that cannot live on the device.** It grows to hold the widest
      slice seen so far, sequentially over slices and persistently over the run, so a slice
      is solved on a grid that already accounts for the ones before it. An analysis kernel
      reduces each slice to a centroid and a radius, the host replays the growth exactly as
      `analyseBeam` does, and the spacing comes back per slice: one round trip per step and
      three floats per slice instead of the particles.
      The radial grid is capped at 384 points by the 32 KB of threadgroup memory, at about
      72 bytes per point; above that the deck is refused by name. The default is 100.
      **Do not validate this against `Beam/SSCfield` early in a run.** It is the `l = 1`
      mode, and before the beam bunches the source term is thousands of unit phasors
      cancelling to nothing: 1e-18 in FP64 against 1e-10 in FP32, a ratio of 1e8 between two
      numbers that are both zero. That looked like a 14% error for a while. Compared on the
      same bunched particles the two fields agree to 4.0e-07 of peak, correlation
      1.000000000. `SSCfield` also reports whichever `m` came last, so with `nphi > 0` it is
      a dipole and averages to noise for a round beam.
      Two scalars have to be taken the way `BeamSolver::advance` takes them and not from the
      nearest similar-looking variable: `gammaz2` uses the lattice `aw`, not the one
      `TrackBeam` zeroes outside an undulator, and the wavenumber is the reference one from
      `&setup`.
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
- [x] **One command buffer per step instead of four.** `fieldStep`, `beamStep` and each
      `fieldMoments` call used to commit and wait on their own. On a 500-slice deck that is
      2%, because a step is 19 ms of real work; on a steady-state deck, where a step is one
      slice, it was the entire cost. The engine now keeps one encoder open and drains it only
      when the host has to look at a buffer, which needs no double buffering precisely because
      nothing is ever in flight while the host is reading: the head of the beam step drains it
      before overwriting the per-slice arrays. A 1104-step steady-state deck went from 3.91 s
      to 1.09 s, against 3.60 s on the CPU — from marginally slower than the CPU to three
      times faster. Device busy time fell from 2.62 s to 0.57 s, which says most of what was
      being counted as device time was submission overhead.
- [ ] The remaining per-step floor on a one-slice deck is about 0.99 ms, of which 0.51 ms is
      device time for around fifteen dispatches. That is dispatch overhead rather than
      arithmetic, so the next thing worth trying for small decks is fewer dispatches — the
      zero and deposit passes could be one kernel, and the field solve's four passes are only
      separable because a 256x256 slice does not fit in threadgroup memory.

- [x] **Port `source_filter`.** With the filter off the solve is four passes, because the
      transform is linear and the source can be added after the back transform. With it on the
      source is shaped in Fourier space, so it needs its own forward transform and the
      combination has to happen there: six passes. The filter table is built at startup from
      the same expression `FieldSolverFFT::init` uses, taken from the values the solver
      settled on rather than the ones the deck asked for, since an unphysical width disables
      it there. Cost on the 500-slice deck: 3.67 s to 4.31 s.
      Filtering *improves* GPU-CPU agreement, 2.1e-04 to 1.5e-05 in the field, because it
      removes the high spatial frequencies where FP32 is weakest — the same effect as raising
      `npart` at large `ngrid`, seen from the other side. The strong case is the one worth
      keeping in the sweep: it changes `Field/xsize` by 25% while the paths differ by 1.5e-05.

### The NVIDIA port, predicted and measured

Estimates made before the backend existed, scaled from the measured 6.0 GB moved per step and
287 GB/s achieved on the M3 Max (72% of its 400 GB/s), against what the RTX 5080 actually did.
The loop is memory bound, so bandwidth is the first-order predictor, and it was a good one.

| card | peak BW | estimated loop | measured | vs M3 Max |
|---|---:|---:|---:|---:|
| M3 Max 40-core | 400 GB/s | 4.1 s measured | 4.10 s | 1.0x |
| RTX 5080 | 960 GB/s | ~1.7 s | **1.42 s** | **2.9x** |
| L4 | 300 GB/s | ~5.5 s | not yet run | ~0.7x |
| A100 40GB | 1555 GB/s | ~1.1 s | not yet run | ~3.9x |

The 5080 came in 20% faster than the bandwidth estimate, which is what retuning the blocking
for a card with 100 KB of shared memory per block rather than Metal's 32 KB is worth. Expect
the same margin on the other two, so the L4 estimate is optimistic on the wrong side of 4.1 s
and the A100 should land near 0.9 s. **Neither has been run.** The code is compiled for both
(`sm_80` and `sm_89` are in the default `CUDA_ARCHS`) and nothing in it is specific to the
5080 — the two device properties that vary, shared memory per block and total memory, are read
from the device and reported rather than assumed — but that is an argument, not a measurement.

**A single L4 is slower than this laptop**, having less bandwidth than Apple's unified memory,
and no amount of FP32 throughput changes that for a bandwidth-bound loop. It is still the most
useful target of the three in context: one L4 lands within the uncertainty of one whole
128-core node (4-9 s by two independent estimates), sixteen of them sit idle, and a realistic
16-core share of a contended node would take about 45 s. Availability beats peak.

Memory is the other axis and cuts the other way. This machine has 128 GB of unified memory; a
deck at `ngrid = 1024` with four harmonics needs about 21 GB of field and scratch and does not
fit on a 16 GB 5080 or a 24 GB L4 as a single rank. Splitting it across cards solves that,
which is one more reason multi-GPU is worth doing early.

**Oversubscribing ranks per GPU is worth it only when the device is not saturated**, and the
report line says which case you are in. Measured here on one GPU: an ISR deck goes 4.94 s at
one rank to 4.15 s at four, because the draws are host side and divide; a deck without ISR goes
3.60 s to 3.71 s, because the device was already 95% busy.

`manual/GPU.md` carries the rest: what `GPUEngine` requires, the four constraints that carry
over, the per-step host round trips that become PCIe transfers on a discrete card, the FP64
question, cuFFT, and the multi-GPU section covering device selection from the MPI local rank.

### Constraints that must not be forgotten

- Apple GPUs have **no FP64**. Metal has no `double`. FP32 is mandatory, not a choice. On
  NVIDIA it is a choice: half rate on an A100, a sixty-fourth on an L4 or a 5080. A
  double-precision engine would be worth writing for the A100 and would be a second engine,
  not a flag, because the FP32 reformulations below are woven through every kernel.
- **`gamma` must be stored as an offset from a reference energy.** At γ₀ = 11357 the FP32
  quantum of absolute γ is 1.35e-3, larger than the per-step energy change (3.0e-4 at
  saturation, 5.5e-7 at seed). Storing absolute γ in FP32 gives 65%–276% error.
- Only the FFT field solver was converted to FP32. **ADI remains FP64-only**, so any FP32 or
  GPU run must set `fft_fieldsolver = true` in `&track`. Easy footgun.
- Metal has **no threadgroup `atomic_float`** — `atomic_float` exists in the device address
  space only. Threadgroup float accumulation needs `atomic_uint` + a CAS loop on the bit
  pattern (cheap in local memory). CUDA has `atomicAdd` on a shared float and does not.
- The tiled deposition requires `ngrid % 32 == 0`, one more reason to prefer 256 over 255.
- **On a discrete card, nothing the host writes may be overwritten while a copy that reads it
  is still queued.** The pinned per-step arrays are reused every step, so `beamStep` drains
  the stream once at the top before writing any of them, and does all its host-side work
  before queueing anything. Metal needed the same rule for a different reason.
- **The busy percentage is a saturation indicator, not a calibrated fraction.** Device
  timestamps and the host clock need not agree to better than a few percent; on WSL2 a
  saturated run reports 100–106%. It answers 95% against 50%, not 95% against 100%.
- **The diagnostic output buffer's slot layout is written out four times** — each backend's
  kernel and each backend's host-side read — and two of the four have already been wrong.
  Bunching occupies 12–27 at eight harmonics and the aux extrema now start at 28 with a stride
  of 48. Nothing about a wrong slot looks wrong, because every entry is a plausible float, so
  changes here need the `aux_slot_layout` sweep case rather than a careful re-reading.
- **Neither backend is bit-reproducible run to run**, and it is not a bug. The source
  deposition is one atomic add per particle per grid corner; the adds commute but their
  rounding does not, and the thread order varies. Measured on the 1104-step steady-state deck:
  two runs of the same binary are identical for 26 output steps and then diverge to 5.7e-07,
  while the CPU path is identical to the last bit. It only shows once the summands span a wide
  range of exponents, which is why it starts part way in. Anything resting on two GPU runs
  agreeing exactly has to be kept short; that is why `aux_slot_layout` uses `zstop = 1`.

## 3. Housekeeping / possible upstream reports

The first three entries below have all been sent and merged — the wall-clock timer as #276,
the on-axis cell as #277, and the steady-state `&wake` abort found during the CUDA port as
#279 — and are kept here only as the record of what they were. What remains open is everything
from the fourth entry down.

- **`Total Wall Clock Time` was `clock()`**, i.e. processor time, not wall clock. Identical
  for a CPU run that never waits; four times too small for one that waits on a device, and it
  also ignores every rank but the zeroth. `GenMain.cpp` now uses `std::chrono::steady_clock`.
  Merged as #276.
- **The on-axis near-field diagnostic sampled the wrong cell for an even `ngrid`.**
  `(ngrid*ngrid-1)/2` is the axis only when `ngrid` is odd; for an even grid it is the last
  column of the row below the middle, i.e. the edge of the grid, where the field is orders of
  magnitude smaller. `intensity-nearfield` and `phase-nearfield` were then reporting a corner
  of the box. `(ngrid/2)*ngrid + ngrid/2` is identical for odd `ngrid` and right for even.
  Nobody noticed because the traditional Genesis convention is an odd `ngrid`; the GPU only
  takes powers of two. Merged as #277.
- **`&wake` in a steady-state run read one past the end of a one-element vector**, described
  in full in section 2. Merged as #279.

- `export FI_PROVIDER=tcp` is a libfabric knob, not a universal one. Conda's Linux MPICH is
  built `ch4:ucx,ofi` and prefers UCX, so it never reaches libfabric and the variable is inert;
  `mpichversion | grep Device` says which netmod you have, and on UCX the equivalent is
  `UCX_TLS` (`self,sm` for a single node). `sweep.py` therefore sets it only when the
  environment has not, since forcing tcp where the ofi netmod has a fast provider is a
  downgrade. Required on macOS with conda MPICH — the default libfabric
  provider busy-polls and costs **5.9× at 8 ranks** (96.0 s → 16.3 s). Worth a manual note.
- `ngrid = 255 = 3 × 5 × 17` is a pathological FFT length: 59.7 s vs 38.3 s at 256 on the same
  case. Worth a manual note or a warning at startup.
- **`CMakeLists.txt` forced `/opt/local/bin/h5pcc` as the C++ compiler whenever that file
  existed**, silently overriding `CMAKE_CXX_COMPILER`, and no command-line argument could
  defeat it because it was a plain `set()`. Recorded here as a reading of the code until it
  bit a real user: a Mac Studio with MacPorts installed, following the GPU.md instructions
  with a conda toolchain, compiled everything and then failed to link with a page of
  undefined `_ompi_mpi_*` symbols. The override runs after `project()`, so CMake had already
  detected conda's clang and derived every flag from it; swapping only the C++ compiler left
  the C compiler and the flags conda's and pulled MacPorts' static parallel HDF5 into a link
  that had no MPI library in it. Now an option, `USE_MACPORTS_H5PCC`, still defaulting on so
  that a MacPorts-only machine is unaffected, and it warns at configure time when the two
  compilers come from different places, which is the whole of the diagnosis.
- If FFTW is not found, `FieldSolverFFT` silently produces wrong results rather than failing
  the build — the `fftw_execute` calls are inside `#ifdef FFTW` but the surrounding copy loops
  are not.
