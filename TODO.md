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
- [x] Threadgroup-tiled deposition — 10.2 → **3.3 µs/slice-step**
- [x] Engine scaffolding, upload/download round trip, GPU field solve, GPU beam step
- [x] Namelist control (`gpu`, `gpu_validate`) and hard errors for unported physics
- [x] **Validation matrix** — `examples/metal-gpu/sweep.py`, about 52 cases over undulator
      geometry, transport, lattice errors, grid size, harmonics, beam parameters, collective
      effects, time dependence and every refusal. It found four defects; see below.
- [x] **GPU diagnostic reductions** — the diagnostics were 76% of the GPU runtime
- [x] **Full residency** — no per-step marshalling; the host only sees the arrays at dumps,
      at the slippage boundary slice, and at the end of `&track`

**Measured (500 slices, zstop=10, ngrid=256, 1 harmonic, output_step=1, M1 Max):**

| config | wall | vs 1 CPU core | vs 8 CPU cores |
|---|---|---|---|
| CPU 1 rank | 375.8 s | 1× | — |
| CPU 8 ranks | 53.3 s | 7.0× | 1× |
| GPU 1 rank | **1.55 s** | **242×** | **34×** |
| GPU 4 ranks (one GPU) | **0.95 s** | **396×** | **56×** |

Extra MPI ranks against the single GPU are worth only about 1.6× and stop helping past
four: the ranks queue on one GPU and there is no longer any host work left to overlap.

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
  most that round-off alone can produce, and every case stays under it. `run_two_track_blocks`
  is the closest at 1.3x the bound.

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
- [ ] **Port the corrector kick.** This is now the highest-value remaining item, ahead of
      `applyR56`. `MetalEngine::beamStep` refuses any step with a non-zero `cx` or `cy`, and
      `orbiterror = true` is implemented as a per-step corrector, so a deck with undulator
      orbit errors puts **187 of 196 steps back on the CPU** and gets almost no benefit from
      the GPU. Orbit errors are a routine part of a realistic deck, so this is not an edge
      case. The work is small: `TrackBeam::applyCorrector` is `px += cx*gamma0`,
      `py += cy*gamma0` over every particle, so it is a constant addition folded into the
      existing beam step. `applyR56` for chicanes is the natural companion and is a single
      shear in the same kernel.
- [ ] `one4one` is refused outright (the GPU wants a rectangular particle array)
- [x] Host-side diagnostic assembly. It was ~5 s of the 7.4 s single-rank time. Most of it was
      not `storeValue` at all but `DiagBeamUser::getValues`, which evaluated a `sin`/`cos` per
      particle per slice per step for an output that is disabled by default and then discarded
      the result. Guarding that, plus hoisting the map lookups out of the slice loops in
      `DiagBeam` and `DiagField`, took `output_step = 1` from 7.33 s to 1.57 s. The same code
      runs in a CPU-only build, where a profile puts `DiagBeamUser` at 2.0% and all of
      `Diagnostic::calc` at 17.9% of the run — worth having, but below the run-to-run noise of
      a 370 s job. Should go upstream on its own branch off master.
- [ ] `fieldMoments` is now 22% of single-rank time and runs every step. It only feeds the
      diagnostics, so it could be skipped on steps that produce no output.

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

- `export FI_PROVIDER=tcp` is required on macOS with conda MPICH — the default libfabric
  provider busy-polls and costs **5.9× at 8 ranks** (96.0 s → 16.3 s). Worth a manual note.
- `ngrid = 255 = 3 × 5 × 17` is a pathological FFT length: 59.7 s vs 38.3 s at 256 on the same
  case. Worth a manual note or a warning at startup.
- `CMakeLists.txt` forces `/opt/local/bin/h5pcc` as the C++ compiler if that file exists,
  silently overriding `CMAKE_CXX_COMPILER`. Surprising on any machine with MacPorts.
- If FFTW is not found, `FieldSolverFFT` silently produces wrong results rather than failing
  the build — the `fftw_execute` calls are inside `#ifdef FFTW` but the surrounding copy loops
  are not.
