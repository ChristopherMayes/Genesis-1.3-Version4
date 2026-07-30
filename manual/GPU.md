# GPU acceleration

Genesis 4 has an optional GPU backend for the tracking loop. It is opt-in at build time and again in each `&track` block, so a binary built with it behaves exactly as it did before unless a deck asks for it. The only backend at present is for Apple Silicon and is written against Metal; the interface it implements is not Apple specific, and the final section of this chapter describes what a backend for another vendor would have to provide.

Everything the Apple backend does is in single precision. That is not a tuning choice: Apple GPUs have no double precision at all and the Metal shading language has no `double` type. Agreement with the CPU path is therefore at the single-precision level, a few times `1e-4` on field amplitudes for a saturating run, and the section on [accuracy](#what-agreement-to-expect) puts that number in context against the things a user changes without thinking about them.

- [Requirements](#requirements)
- [Building](#building)
- [Switching it on](#switching-it-on)
- [What is supported](#what-is-supported)
- [Collective effects and other physics](#collective-effects-and-other-physics)
- [The worked example](#the-worked-example)
- [What agreement to expect](#what-agreement-to-expect)
- [Performance notes](#performance-notes)
- [Where the code lives](#where-the-code-lives)
- [Porting to another GPU](#porting-to-another-gpu)

## Requirements

An Apple Silicon Mac is required. The backend refuses to start on a device without unified memory, which rules out the Intel Macs with discrete GPUs.

The command line developer tools supply the Metal framework headers and are installed with `xcode-select --install`; a full Xcode installation also works. The offline `metal` compiler is not needed, because the shaders are compiled from source when the first `&track` block starts.

The remaining dependencies are the same as for a CPU build: a C++17 toolchain, MPI, HDF5 and FFTW.

## Building

The conda-forge toolchain is the least surprising way to get the four dependencies to agree with each other on macOS.

```sh
conda create -n genesis4-dev -c conda-forge \
    cxx-compiler c-compiler cmake make pkg-config \
    mpich "hdf5=*=mpi_mpich_*" fftw
conda activate genesis4-dev
```

Configure with the backend enabled and build in the usual way.

```sh
cmake -S . -B build-metal \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DENABLE_METAL=ON
cmake --build build-metal -j8
```

Look for `-- Metal GPU backend enabled` in the configure output. `ENABLE_METAL` defaults to `OFF` and is a hard error on anything that is not macOS.

Two properties of `CMakeLists.txt` are worth knowing on a Mac. It will use `/opt/local/bin/h5pcc` as the C++ compiler if that file exists, overriding `CMAKE_CXX_COMPILER`, so a MacPorts installation can quietly hijack a conda build. And if FFTW is not found the build still succeeds, but a deck asking for `fft_fieldsolver = true` is then given the ADI solver without being told, so check for `-- FFTW found` as well.

## Switching it on

Two booleans in `&track` control the backend, and both default to `false`.

```
&track
gpu = true
&end
```

Setting `gpu = true` makes the GPU the answer rather than a check on the answer. The beam and the field live in GPU memory for the whole of that `&track` block, and the host sees them only at a `MARKER` dump, at the single slice the slippage exchanges each step, and once at the end.

```
&track
gpu = true
gpu_validate = true
fft_fieldsolver = true
&end
```

Setting `gpu_validate = true` additionally runs the CPU path at every step and reports the largest relative difference at the end. This is a test mode and is slower than either path on its own, because it performs both. `fft_fieldsolver = true` belongs with it: without it the CPU half of the comparison uses the ADI solver, and the reported difference then measures the two solvers against each other rather than the two processors.

At the end of a GPU `&track` block one line reports what the device did.

```
Metal: 196 steps in 4.09731 s, device busy 3.79265 s (93%)
```

The first figure is the wall clock of the tracking loop and the second is the time the device itself spent executing, which Metal timestamps on every command buffer at no cost. The percentage answers the question worth asking before reaching for more hardware: a device that is already busy will not go faster if more MPI ranks are pointed at it, and one that is idling means the host is the limit. Neither figure is Genesis' own `Total Wall Clock Time`, which covers the whole program including loading and output.

A deck that asks for the GPU and cannot have it is an error rather than a silent fallback, because the alternative is a run that quietly produces CPU numbers and CPU timings under a GPU label. Each message names the reason.

```
*** Error: gpu = true in &track, but this binary was built without the GPU
    backend. Reconfigure with -DENABLE_METAL=ON.
*** Error: gpu = true in &track, but ngrid = 151 is not supported by the Metal
    field solver, which handles powers of two from 64 to 1024. Set ngrid = 128
    in &field. ...
*** Error: gpu = true in &track, but ngrid = 512 in &efield, but the GPU
    space-charge solve holds the radial arrays in threadgroup memory and
    handles 3 to 384
```

## What is supported

The GPU runs the source deposition, the field propagation, the transverse map, the Runge-Kutta longitudinal push, the collective kicks and all of the per-slice diagnostics. The table below covers the boundaries of that.

| | |
|---|---|
| `ngrid` | a power of two from 64 to 1024 |
| field harmonics | up to four, all sharing the same `ngrid`, `dgrid` and `gridmax` |
| `bunchharm` | up to eight |
| `fft_fieldsolver` | must be `true`; there is no ADI solver on the GPU |
| `source_filter` | supported, at the cost of two extra transform passes per step |
| correctors | supported; the kick rides on the closing half step |
| chicanes | supported; the transfer map rides on the opening half step and the R56 shear sits before the closing one |
| wakefields (`&wake`) | supported, including the resistive wall, geometric and roughness wakes and the external loss |
| incoherent radiation (`&sponrad`) | supported; both the loss and the spread, reproducing the CPU run rather than only its statistics |
| space charge (`&efield`) | supported, long and short range; the radial grid runs to 384 points |
| `one4one` | not supported; the backend requires a rectangular particle array |

Everything the backend cannot do is a hard error rather than a fallback, and the reason is the same in each case. A run that completes and writes a plausible output file, having quietly done something other than what the deck asked for, is the failure this design works hardest to avoid. An ADI deck would be propagated by FFT instead, and a `bunchharm` above eight would be answered from host particle arrays that are stale, because the particles stay on the GPU. Neither would announce itself in the output file.

No lattice element falls back to the CPU. The machinery for a fallback is still present, and a step that took it would be counted and reported at the end of the run, but nothing currently triggers it.

### The transverse grid

The restriction to powers of two is the one users notice. Genesis decks traditionally use an odd `ngrid` so that a grid point sits exactly on the axis, but that convention buys nothing physically and costs a great deal in transform structure: `ngrid = 255` factors as 3 x 5 x 17, which is a poor length for an FFT and runs about 1.5 times slower than 256 even on the CPU. The error message names the nearest supported size.

Each grid size gets its own specialisation of the transform. The kernel is a four-step Cooley-Tukey decomposition `N = REGS x LANES`, in which every thread holds `REGS` points in registers, performs a short DFT over them, exchanges through threadgroup memory, and finishes with `LANES`-point DFTs. The two factors are chosen per grid size and injected into the shader as preprocessor macros when the Metal library is compiled, so there is no runtime branching in the inner loop.

| `ngrid` | REGS x LANES | radices used | tracking loop | device busy |
|---:|---:|---|---:|---:|
| 64 | 8 x 8 | 8, 8 | 0.93 s | 82% |
| 128 | 16 x 8 | 16, 8 | 1.60 s | 85% |
| 256 | 16 x 16 | 16, 16 | 4.09 s | 92% |
| 512 | 32 x 16 | 32, 16 | 15.9 s | 98% |
| 1024 | 32 x 32 | 32, 32 | 67.4 s | 98% |

These are for the SASE example at one rank on an M3 Max. Each row has four times the grid points of the row above it, and from 256 upwards each doubling costs about four times as much, which is what a memory-bound transform should do. The smaller grids cost less than that only because they do not fill the machine, which is what the busy column reports, and it is the reason a small deck sees less benefit from the GPU than a large one.

Agreement with the CPU degrades with grid size, but the grid is not really the cause. Over the 1104 steps of the validation deck, holding `npart` at 8192, the largest relative field difference is 9.4e-6 at `ngrid = 64`, 2.3e-4 at 256 and 8.5e-3 at 1024. What changes across that row is how many particles land in each cell: a bilinear deposition of a fixed number of particles onto a finer mesh leaves more shot noise in the source term, and that noise is of high spatial frequency, which is where single precision is weakest. Raising `npart` at fixed `ngrid = 1024` brings the difference back down.

| `npart` | field difference |
|---|---|
| 8192 | 8.5e-03 |
| 32768 | 3.2e-03 |
| 131072 | 1.2e-03 |

The rule is therefore to scale `npart` with `ngrid` rather than to distrust the large grids. A deck that puts a handful of particles in each cell is under-resolved in double precision as well; single precision merely makes it visible. The beam moments are unaffected and stay at 3.2e-6 throughout, because they are a reduction over particles and never touch the grid.

### The source filter

With `source_filter = false`, which is the default, the field solve takes four passes per slice and step:

    field = IFFT(FFT(field) * expK) / ngrid^2 + 2 * src

The source term is added after the back transform rather than before it. That is exact rather than approximate, because the transform is linear and `IFFT(FFT(src))/ngrid^2` is `src`, and it saves one of the three two-dimensional transforms. The CPU path does the same thing.

With the filter on, the source is shaped in Fourier space and so needs its own forward transform, and the combination has to happen there:

    field = IFFT(FFT(field) * expK + 2 * filter * FFT(src)) / ngrid^2

That is six passes rather than four. On the 500-slice example the tracking loop goes from 3.67 s to 4.31 s, which is the 50% more transform work diluted by everything in a step that is not the field solve. The filter itself is tabulated once at startup from the same expression `FieldSolverFFT::init` uses, and taken from the values the solver settled on rather than those the deck asked for, since an unphysical width or centre disables the filter there rather than being used.

Filtering improves agreement with the CPU rather than degrading it. On a deck with `xcut = ycut = 0.15` and `sigmoid = 0.05`, which changes `Field/xsize` by 25% and `Field/power` by 6%, the two paths agree to 1.5e-05 in the field where an unfiltered deck agrees to 2.1e-04. This is not surprising: the filter removes the high spatial frequencies where single precision is weakest, which is the same effect as the `npart` row above seen from the other side.

## Collective effects and other physics

### Wakefields

A `&wake` block works on the GPU, including the resistive wall, geometric and roughness wakes and the external `loss` term. The split of work follows the structure of the physics rather than the structure of the code. A wake is driven by the current of a slice rather than by the coordinates of individual particles, so the loss it produces is one number per slice and every particle in that slice receives the same energy kick. Building the loss profile requires the current of the whole bunch, which under MPI lives on several ranks and is gathered with `MPI_Allgather`; applying it requires only an addition.

The gather and the convolution therefore stay on the host, exactly as on the CPU path, and the GPU performs the addition. Nothing has to come back from the GPU for this to work, because the slice currents are fixed for the duration of a run and particles do not migrate between slices, so residency is preserved.

The cost is small. On the 500-slice example a resistive wall wake adds under 3% to the runtime with `transient = false`, which is the host-side convolution being computed once. With `transient = true` the convolution is repeated at every step and the run takes about 2.6 times as long, all of it on the host. That is the same host work the CPU path does, so it is not a limitation of the backend, but it does mean a transient wake dominates a run that is otherwise a second long.

### Incoherent synchrotron radiation

A `&sponrad` block works on the GPU, both the mean energy loss (`doLoss`) and the quantum spread (`doSpread`). The interesting part is where the random numbers come from.

`Incoherent::apply` draws one number per beamlet rather than per particle: it draws when `ip % nbins == 0` and gives the whole beamlet the same energy kick, so that the shot noise the beamlet was loaded with is not disturbed. The generator is `RandomU`, which is the `ran2` routine from Numerical Recipes, a pair of linear congruential sequences behind a Bays-Durham shuffle table. A shuffle table has no practical jump-ahead, so a kernel cannot compute the *n*-th number of that sequence directly.

The obvious response is to put a different generator on the device, one that can be indexed, and to accept that a GPU run then differs from a CPU run by a whole noise realisation rather than by round-off. This backend does not do that. The draws are taken on the host instead, from the generator Genesis already uses and in the order the CPU path consumes them, and uploaded as one number per beamlet per step. A GPU run therefore reproduces a CPU run particle for particle, and the ordinary step-by-step comparison under `gpu_validate` keeps working: with radiation switched on, the beam agrees to 3.2e-06 over 1104 steps, which is the figure a deck with no radiation reports. Had the two paths drawn from different sequences, that number would have been of the order of the kick itself.

The backend draws from its own generator, seeded identically to the one `Incoherent::apply` uses, rather than sharing it. Under `gpu_validate` both paths run, and a shared generator would have given the first path one set of numbers and the second the next set, so the two would have disagreed by the full size of the effect while both were behaving correctly.

What this costs is host time, and it is the one place in the backend where that is a significant cost. On the 500-slice example the tracking loop goes from 4.64 s to 6.43 s with `doSpread` on, while the device busy time goes only from 4.32 s to 4.75 s: the extra 1.5 s is 100 million draws on the host at about 15 ns each. The CPU path performs exactly the same work, so this is not a penalty relative to it, but the radiation does dilute the speedup.

With `doLoss` alone the kick is a per-step scalar, the same for every beamlet, and the draws are skipped entirely; the loop then takes 4.96 s, the difference being the buffer and one more dispatch. Skipping them is safe rather than a shortcut, because the spread scales the draw by zero and no result can depend on it.

For scale, on the steady-state deck the radiation changes `Field/power` by 25%, `Beam/energyspread` by 8.9% and `Beam/bunching` by 9.5%, against which the two processors differ by 2.2e-04 in the field and 3.2e-06 in the beam.

### Space charge

An `&efield` block works on the GPU, both the long-range field and the short-range solve that `nz` and `nphi` switch on.

The short-range solve is a set of azimuthal modes `m` and longitudinal modes `l`. For each pair the particles of a slice are binned in radius into a complex source term, a tridiagonal system is solved on the radial grid, and the result is gathered back onto the particles. That maps to one threadgroup per slice with the radial arrays in threadgroup memory. The tridiagonal solve is a recurrence, so a single thread runs it while the others wait; the grid is small enough that the parallel alternatives cost more than they save.

One part cannot stay on the device. `rmax` grows to hold the widest slice seen so far, and it does so as the slices are visited in order and persists across the whole run, so slice *k* is solved on a grid that already accounts for the slices before it. A backend that sized each slice independently would agree with the CPU only until the first slice wider than the grid. An analysis pass therefore reduces each slice to a centroid and a bounding radius, the host replays that growth exactly as `analyseBeam` does, including the message when the grid is enlarged, and the resulting spacing comes back per slice. That costs one round trip per step and three floats per slice, rather than the particles.

The radial grid is limited to 384 points, because the six complex and four real arrays of the solve come to about 72 bytes per point and a threadgroup has 32 KB of memory. A larger grid is refused by name. The default is 100.

The `SSCfield` diagnostic deserves a warning, because it is the natural thing to check and it is misleading in two separate ways. It reports the `l = 1` mode, and at the head of a run there is no bunching, so its source term is a sum of thousands of unit phasors that should cancel to nothing; what survives is the arithmetic, which is 1e-18 in the CPU's double precision and 1e-10 in the GPU's single precision, a ratio of 1e8 between two numbers that are both zero. It also reports whichever azimuthal mode came last, since the CPU writes it once per `m`, so with `nphi > 0` it is a dipole or higher and averages to noise for a round beam. Compared where the field is real, on the same bunched particles, the two paths agree to 4.0e-07 of peak with a correlation of 1.000000000.

## The worked example

Everything below is in `examples/metal-gpu/` and takes a few minutes.

### 1. The self-check

```sh
cd examples/metal-gpu
export FI_PROVIDER=tcp
../../build-metal/genesis4 validate.in
```

Set `FI_PROVIDER=tcp` before anything else if you are using conda's MPICH; the [performance notes](#performance-notes) explain why. It is worth 40% even on this single-rank run, which performs no communication at all, and a factor of four on a run with several ranks.

This is a steady-state run of the ARAMIS undulator with the shot noise switched off and a seeded field, so it is deterministic, and it runs both paths and compares them at every step. The last line is the point of it.

```
Metal vs CPU over 1104 steps: max relative error, field 0.000207, beam 3.18e-06
```

Those two numbers are the single-precision round-off level accumulated over 1104 steps. If the run on your machine reports something similar then the backend is working. A number above about `1e-3` means it is wrong rather than merely less precise.

At the start it also prints the device it selected and a check that the upload round trip is clean before any physics happens.

```
Metal backend: Apple M3 Max, 1 MB resident, gamma_ref = 11357.8
  host transfer check: field 4.5e-08   beam 4.4e-08 (relative, FP32 rounding is ~1e-7)
```

The timing line from this deck is not a performance figure and should not be read as one, because `gpu_validate` runs the CPU path as well and a steady-state deck has a single slice, so there is very little for the device to do. The performance notes give the same line for a deck that is representative.

### 2. Timing and a real comparison

`sase_cpu.in` and `sase_gpu.in` are the same 500-slice SASE run, 10 m of ARAMIS at `ngrid = 256` with diagnostics at every step, differing only in the `gpu` flag and the rootname. The shot-noise seed is fixed in both.

```sh
export FI_PROVIDER=tcp
mpirun -n 8 ../../build-metal/genesis4 sase_cpu.in
mpirun -n 8 ../../build-metal/genesis4 sase_gpu.in
python3 compare.py sase_cpu.out.h5 sase_gpu.out.h5
```

Run both at the same number of ranks. Genesis pads the slice count up to a multiple of the rank count and distributes the slices accordingly, so the shot-noise realisation depends on the rank count, and two CPU runs of this deck at 8 and at 12 ranks differ by 10% in `Field/power` and by 91% in per-slice `Beam/bunching`. Comparing across rank counts measures that rather than the GPU. `compare.py` will say so if the array shapes end up mismatched, but if they happen to match it cannot tell.

`compare.py` needs `h5py` and `numpy`, which the build environment above does not have. Use whichever environment you normally analyse Genesis output in, or add them to it.

```sh
conda install -n genesis4-dev -c conda-forge h5py numpy
```

### 3. The full validation matrix

`sweep.py` runs seventy-two decks against the CPU path on the same inputs and prints one line per case. Sixty-one of them compare per-step differences under `gpu_validate`, seven being decks the backend must refuse; the remaining eleven run to completion and compare the output against a bound derived from the per-step figure and the step count. A case that falls back to the CPU where it should not is a failure, because a fallback gives the right answer by the slow route and is exactly how an unported element would hide.

```sh
cd examples/metal-gpu
python3 sweep.py --workdir /tmp/g4sweep
```

It needs the same `h5py` and `numpy` as `compare.py` and takes some tens of minutes. Pass `--tier step` or `--tier run` for one tier only, and `--only <regex>` to select cases by name. This is the check to run after changing anything in the backend.

The matrix covers planar and helical undulators, tapers, phase shifters, undulator and quadrupole offsets, gradients and roll-off, field and orbit errors, every supported grid size, harmonics, the source filter, all four wakefield models, incoherent radiation, long-range and short-range space charge, chicanes, correctors, time-dependent and time-periodic running, dumps and multiple `&track` blocks.

`tools/fftcheck.mm` is a smaller and more direct check of the FFT kernels alone. It extracts the shader source from `src/Core/MetalEngine.mm`, so it always tests the code Genesis actually runs, compiles it for one grid shape, and compares a row transform, a column transform and a complete four-pass solve against a direct double-precision DFT. It is not part of the build, since it is a diagnostic rather than a regression test, and it is what to reach for when the sweep says the field is wrong but not where.

```sh
clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
    -O2 tools/fftcheck.mm -o /tmp/fftcheck
/tmp/fftcheck 256 16 16 8 16      # ngrid lanes regs rowsPerTG colsPerTG
```

Run it from the top of the source tree, since it reads the shader out of `src/Core/MetalEngine.mm` by a relative path. The five shapes the backend itself uses are `64 8 8 16 16`, `128 8 16 16 16`, `256 16 16 8 16`, `512 16 32 4 8` and `1024 32 32 2 4`, taken from `pickFFTShape` in that file. All five report a row and column error of a few times `1e-6` against a scale of order ten, and a round trip through the complete solve accurate to `4e-07`.

## What agreement to expect

The reference numbers below are from an M3 Max with both runs at 8 ranks. Nothing here should be much different on another Apple Silicon machine, since the arithmetic is the same.

```
amplitudes, relative                         error       scale
  Field/Global/intensity-farfield       6.614e-04   3.447e+20
  Field/intensity-farfield              6.533e-04   1.768e+21
  Field/ydivergence                     6.055e-04   1.599e-05
  Field/xdivergence                     4.219e-04   1.590e-05
  Field/intensity-nearfield             3.950e-04   1.104e+16
  Beam/bunching                         1.927e-04   1.241e-02
  Field/power                           1.448e-04   1.338e+07
  Field/xsize                           4.592e-05   7.737e-05
  worst of 51: 6.614e-04
```

Three things make a raw dataset-by-dataset comparison misleading, and `compare.py` separates them for that reason.

Amplitudes are the honest number, taken relative to their own peak. A few times `1e-4` on a SASE run is the expected level, because SASE amplifies exponentially and a `1e-7` difference in the first metre is not a rounding error at the end of the undulator but a rounding error multiplied by the gain. The steady-state figure of `2e-4` over 1104 steps is the same effect.

Centroids such as `xposition` and `pointing` average to zero for a symmetric beam, so their magnitude is set by round-off in the first place and a relative error against them means nothing. `Beam/xposition` on this deck is `2e-11` m against a beam 25 µm across. `compare.py` prints these as an absolute difference beside the scale of the quantity itself.

Phases are defined modulo 2π and are undefined where there is no amplitude to carry them. The near-field phase in particular is the argument of a single on-axis cell, which passes through zero at optical vortices while the slice as a whole is still bright. `compare.py` wraps them and masks out the points where the companion intensity is below a thousandth of its peak; what remains agrees to about `1e-2` radians.

The on-axis quantities deserve a note of their own, since a GPU run is where the problem showed. `intensity-nearfield` and `phase-nearfield` report the field in the cell at the centre of the grid, and that cell used to be selected as `(ngrid*ngrid-1)/2`, which is the centre only for an odd `ngrid`. For an even one it lands at the edge of the grid, where the field is orders of magnitude weaker and a percent-level difference between two runs means nothing. Since this backend accepts only powers of two, every GPU run encountered it. The index is now `(ngrid/2)*ngrid + ngrid/2`, which is unchanged for odd grids, and the end-to-end agreement on the validation cases improved by an order of magnitude with it.

The CPU is also not a fixed target to compare against. `FieldSolverFFT` plans its transforms with `FFTW_MEASURE`, so two runs of the same CPU binary on the same machine differ by `1e-15` to `1e-9` depending on which plan the planner happened to select. When chasing a discrepancy, run the same binary twice first and use that as the control.

It is worth seeing the same deck compared three ways, because the middle row is the one the backend is asking to be trusted and the rows around it are what that number should be judged against.

| comparison | `Field/power` | `Beam/bunching` | `Field/Global/intensity-farfield` | worst of 51 |
|---|---:|---:|---:|---:|
| CPU against itself, 8 ranks, run twice | 6.7e-15 | 0 | 1.1e-15 | 4.1e-08 |
| GPU against CPU, both at 8 ranks | 1.4e-04 | 1.9e-04 | 6.6e-04 | 6.6e-04 |
| CPU against CPU, 8 ranks against 12 | 1.0e-01 | 9.1e-01 | 1.6e-01 | 9.6e-01 |

The bottom row is what happens when a user changes something they are entitled to consider irrelevant, and it is documented behaviour rather than instability: the shot-noise realisation follows the slice distribution, so at 8 and at 12 ranks the same deck is a different noise seed and the per-slice quantities are unrelated in detail. Even the current-weighted global intensity, which averages over the whole bunch, moves by 16%. The GPU differs from the CPU by four orders of magnitude less than that, and by four orders of magnitude more than the CPU reproduces itself.

The physical comparison is the more useful one, and the wakefield case was measured both ways on the same runs, on the 500-slice deck with a copper resistive wall wake at a 2.5 mm radius and both paths at 4 ranks.

| | GPU vs CPU | wake vs no wake |
|---|---:|---:|
| `Beam/wakefield` | 0 | 1.0 |
| `Beam/energy` | 4.0e-10 | 4.0e-05 |
| `Beam/bunching` | 3.5e-04 | 8.5e-03 |
| `Field/power` | 1.7e-04 | 2.8e-03 |

`Beam/wakefield` is identical because it is the host-side loss profile, written straight out by both paths. Everywhere else the wake changes the answer by one to two orders of magnitude more than the two processors differ, and the CPU-to-GPU column is unchanged from the no-wake case, so adding the wake has cost no accuracy. The same holds for the other collective effects: incoherent radiation changes `Field/power` by 25% and space charge by 1.8%, against disagreements of 2.2e-04 and 3.2e-06 respectively.

## Performance notes

End-to-end wall clock of `sase_cpu.in` and `sase_gpu.in`, which are 500 slices at `ngrid = 256` over 196 steps with diagnostics at every step, on an M3 Max with 12 performance cores, 4 efficiency cores, a 40-core GPU and 128 GB.

| ranks | `sase_cpu.in` | `sase_gpu.in` |
|---:|---:|---:|
| 1 | 309.1 s | 5.2 s |
| 2 | 158.5 s | 4.8 s |
| 4 | 82.1 s | 4.6 s |
| 12 | 35.2 s | 4.4 s |

About a second of each of those is setup, loading and the output file, on both paths. The tracking loop itself is 4.1 s on the GPU and about 34 s on twelve CPU ranks, so the GPU is worth roughly eight times all twelve cores and seventy-five times one of them, and it leaves the cores free while it works.

Measure with a clock rather than with `clock()`. Genesis' `Total Wall Clock Time` was processor time until recently, which is the same thing for a CPU run that never waits and badly wrong for a GPU run that does: this deck reported 1.2 s while actually taking 5.2 s, because waiting costs no processor time and the shader compile happens in another process. That line is now genuine wall clock. Earlier versions of this chapter quoted the old figures and overstated the GPU by about a factor of four.

More MPI ranks against the one GPU achieve nothing. The tracking loop takes 4.1 s at one rank and 4.1 s at twelve, and the report at the end of the run says why: at one rank the device is already 92% busy, so there is nothing left to overlap and the extra ranks only divide up work they then queue for. Use one rank for a GPU run unless the deck needs the memory of several nodes.

Of the 4.1 s, 37% is the per-slice diagnostics, which `output_step = 100` reduces from 4.04 s to 2.53 s, 6% is the source deposition, and the rest is the four FFT passes, the Runge-Kutta push and the transverse map. The field solve runs close to the memory bandwidth of the machine, so the diagnostics are where the remaining headroom is, and `output_step` is the knob that costs nothing to turn. Most of the diagnostic cost is the far-field branch, which transforms every slice a second time.

A small deck is a different problem, and used to be a bad one. The 500-slice example performs 19 ms of real work per step, so what it costs to hand work to the device does not matter. A steady-state deck is a single slice and a few tens of microseconds of arithmetic, and there the handover was everything: the engine now encodes a whole step into one command buffer and submits once, where it previously submitted four times and waited four times.

| 1104-step steady-state deck | before | after |
|---|---:|---:|
| tracking loop | 3.91 s | 1.08 s |
| device busy | 2.62 s | 0.54 s |

The same deck takes 3.75 s on one CPU core, so a steady-state run went from marginally slower than the CPU to three times faster. That the device busy time fell by as much as the wall clock is the point: most of what Metal was attributing to the device was the cost of starting work rather than of doing it. The remaining floor is about 1 ms per step, half of it device time for the fifteen or so dispatches a step encodes, so the next thing to try for small decks is fewer dispatches rather than faster ones.

A step the backend refuses would cost more than it looks, because the particles and the field would have to come back to the host before it and go out again afterwards, making that step slower than the same step on the CPU alone. Correctors used to be refused, and since `orbiterror = true` is implemented as a corrector at every step, an orbit-error deck spent 187 of its 196 steps that way and lost about two thirds of the benefit of the GPU. Nothing falls back today.

`export FI_PROVIDER=tcp` is not optional if you are using conda's MPICH on macOS. The default libfabric provider busy-polls while waiting, and the faster the compute becomes the more of the machine that wastes.

| | without | with |
|---|---:|---:|
| `validate.in`, serial, no `mpirun` | 8.9 s | 6.3 s |
| `sase_gpu.in`, 8 ranks | 18.9 s | 5.0 s |

The second row is the one that matters, since eight ranks waiting on each other and on one GPU spend nearly four times the wall clock spinning. The first has no MPI communication in it at all, being one rank launched directly, and still loses 40%. Put the export in your shell profile. The size of the penalty varies with the MPICH build; an earlier one lost a factor of four even on the serial row.

Startup is not free but it is not the problem either. The Metal shaders are compiled from source when the first `&track` block starts, in another process, and the cost does not scale with the problem. On this deck the whole fixed cost of a GPU run, covering loading, the shader compile and the output file, is about 1 s against about 0.9 s for the CPU path, whose own fixed cost is mostly FFTW planning its transforms. Neither is worth optimising unless the tracking loop is shorter than either.

## Where the code lives

Six files, of which two are the Apple backend.

| | |
|---|---|
| `include/GPUEngine.h` | the interface the tracking loop talks to, and the design constraints any backend has to respect |
| `src/Core/GPUEngine.cpp` | which backend is compiled in and how it is created; one branch per backend |
| `include/MetalEngine.h`, `src/Core/MetalEngine.mm` | the Apple Silicon backend: the Metal shaders and the host code that drives them |
| `include/SliceMoments.h` | the per-slice diagnostic moments, in the normalisation `DiagBeam` and `DiagField` expect |
| `src/Core/Gencore.cpp` | the tracking loop, which contains no device-specific code and no preprocessor conditionals |

`Gencore.cpp` is worth reading against the interface rather than against Metal. It holds a `GPUEngine *` and has no `#ifdef` in it: the device-specific half of every decision is inside the backend, and the loop's half is only ever whether the host needs the data now.

The comments at the head of `MetalEngine.mm` record the mistakes that have already been made once. The parameter structs are written twice, once in the shader string and once in host C++ immediately below it, so editing one alone skews the buffer layout silently and produces plausible but wrong physics. Dispatch geometry has to be parameterised at every dispatch site, because hard-coding it at one site only breaks at grid sizes other than the one being developed against. And the order of operations within a step has to mirror `Beam::track` exactly, down to the corrector kick landing before the longitudinal momentum factor is formed and only on the closing half step.

## Porting to another GPU

A second backend is a new implementation of `GPUEngine` and one branch in `GPUEngine::create()`. Nothing in the tracking loop changes, and nothing in the physics classes changes either: the host-side work that the collective effects need has already been separated from the particle work, so a CUDA backend calls the same `Beam::computeWakeLoss`, `Beam::computeIncoherentKick`, `Beam::planShortRangeSC` and `TrackBeam::chicaneMatrix` that the Metal one does. The seams are cut; what remains is the device code.

### What has to be implemented

`GPUEngine` declares twenty methods, and they fall into four groups. The setup group is `init`, which allocates the resident buffers and reports by name anything the deck asks for that the backend cannot do. The transfer group is `upload`, `uploadBeam`, `download`, `downloadField`, `downloadBeam` and the two single-slice transfers the slippage uses. The step group is `fieldStep` and `beamStep`. The diagnostic group is `beamMoments` and `fieldMoments`, which produce the per-slice reductions in the normalisation `SliceMoments.h` documents, plus `compare` for `gpu_validate`, `deviceName`, `gammaRef`, `bytesResident` and `deviceSeconds`.

The Metal backend is about 2,000 lines, of which roughly half is shader source. A reasonable order of work is the field solve first, since it is the largest single piece and `tools/fftcheck.mm` can check it in isolation, then the beam step, then the diagnostics, then the collective effects one at a time. Each of those can be validated on its own with `sweep.py --only <regex>` before the next is started.

### The four constraints that carry over

The beam and the field must stay resident for the whole `&track` block. Marshalling the host arrays in and out at every step costs 38 ms per step on this problem against about 11 ms of compute, so a solver plugged in behind `FieldSolver::advance`, which would have to copy on every call, cannot pay off however fast its kernels are. The beam and the field are coupled in both directions at every step, since the deposition reads the particles and the Runge-Kutta push gathers the field, so neither can move to the device without the other.

The diagnostics have to be reduced on the device as well. They are not a small share of the arithmetic once the tracking is fast, being 37% of the tracking loop at `output_step = 1`, and leaving them on the host would break residency in any case, because they read every particle and every grid point.

Unsupported physics must be refused by name rather than worked around. A run that completes and writes a plausible output file having quietly dropped an effect the deck asked for is the worst failure mode available here, and it is what the hard errors and the sweep's treatment of an unexpected CPU fallback exist to prevent.

Single precision has to be arranged for rather than merely accepted, if the backend uses it. Two reformulations in `MetalEngine.mm` are the ones to copy: gamma is carried as an offset from a reference energy, because at gamma = 11357 the single-precision quantum of absolute gamma is 1.35e-3 and the per-step energy change is 3.0e-4 at saturation, so storing absolute gamma gives errors between 65% and 276%; and the longitudinal momentum is formed as `gamma*sqrt(1-r)` rather than `sqrt(gamma^2-1-aw^2-p^2)`, since the single-precision quantum of gamma squared is about 15 and the subtraction loses the whole transverse contribution.

### What is different about a discrete NVIDIA card

The largest difference is that the memory is not unified. This backend allocates every buffer as shared storage and the host reads and writes it directly, which is why the round trips described above are cheap enough to be acceptable. On a discrete card each of them becomes a transfer across PCIe, and they are worth listing because they happen every step.

| round trip | size per step | when |
|---|---|---|
| incoherent radiation draws | one float per beamlet, 2 MB for 500 slices at 8192 particles and `nbins = 8` | host to device, only with `doSpread` |
| space-charge analysis | three floats per slice down, one float per slice up | both directions, only with short-range space charge |
| space-charge diagnostic | one float per slice | device to host, only with short-range space charge |
| wakefield loss | one float per slice | host to device, only with `&wake` |
| slippage | one field slice, `ngrid^2` complex, 512 KB at `ngrid = 256` | both directions, once per slip event in a time-dependent run |

None of these is large in itself, but each is a synchronisation point, and on a discrete card a synchronisation costs more than the transfer does. The measurement that decides the design is worth repeating on the target machine before optimising anything: the report line at the end of a `&track` block gives the wall clock of the loop and the device busy time, and if the second is much smaller than the first then the host, the transfers or the launches are the limit rather than the arithmetic.

Double precision is available on NVIDIA hardware, and that changes what is worth doing. On a datacentre card it runs at half the single-precision rate and would give agreement with the CPU limited only by the order of operations, which would make `gpu_validate` a much sharper instrument; on a consumer card it runs at a thirty-second or a sixty-fourth of the rate and is not worth having. A backend could reasonably offer both and let the deck choose. Note that the interface already exposes `gammaRef`, which a double-precision backend does not need but costs nothing to keep.

`cuFFT` exists and the hand-written transform does not have to be reproduced. Three things about the field solve are fused into the transform passes here and would have to be handled either with callbacks or as separate elementwise kernels: the `expK` multiply on the inverse row pass, the source addition on the inverse column pass, and, when the source filter is on, the filter multiply on the forward column pass of the source. Whether that is faster than a hand-written kernel is a question for measurement; on this hardware the four-pass fused form runs at close to the memory bandwidth of the machine.

The remaining mapping is mechanical. A threadgroup is a block and threadgroup memory is shared memory; a SIMD group is a warp, and the `simd_sum` reductions in the diagnostics become warp shuffles or `cub::BlockReduce`. `threadgroup_barrier` is `__syncthreads`. One point of friction disappears: Metal has `atomic_float` in the device address space only, so the source deposition and the space-charge accumulation use a compare-and-swap loop on the bit pattern in shared memory, whereas CUDA has `atomicAdd` for floats in shared memory directly. The shaders are compiled from source at startup here, with the grid size and the transform shape injected as preprocessor macros, which on CUDA would more naturally be template parameters instantiated at build time; that also removes the startup cost.

### Validating a new backend

Use the same instruments. `gpu_validate = true` runs both paths from the same state at every step and reports the largest relative difference, which is what makes a wrong kernel show up as an unmistakable jump rather than as chaotic growth; `sweep.py` drives that across the whole matrix and needs only `--genesis` pointed at the new binary. The sweep is not Apple specific and nothing in it assumes Metal.

Two habits are worth adopting from this port. When a comparison looks catastrophic, check whether the quantity being compared is meaningfully non-zero before believing it: the `SSCfield` case above spent a while looking like a 14% error when both numbers were zero to within their own arithmetic. And when a before-and-after measurement comes out as exactly zero, suspect the measurement: building a reference binary with `git checkout` rather than a separate worktree carries staged changes across and silently compares a binary with itself.
