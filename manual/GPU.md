# GPU acceleration

Genesis 4 has an optional GPU backend for the tracking loop. It is opt-in at build time and again in each `&track` block, so a binary built with it behaves exactly as it did before unless a deck asks for it.

There are two backends. `ENABLE_METAL` builds the Apple Silicon one, written against Metal; `ENABLE_CUDA` builds the NVIDIA one, written against CUDA, measured on an RTX 5080 and compiled for the L4 and the A100 as well. They implement the same interface, `GPUEngine`, and the tracking loop cannot tell them apart. The physics, the transform and the order of operations are the same code transcribed, so the two produce the same numbers; what differs between them is the memory model, and everything that follows from a discrete card having its own.

Everything both backends do is in single precision. On Apple that is not a choice at all, since Apple GPUs have no double precision and the Metal shading language has no `double` type. On NVIDIA it is a choice, and the reasoning is in [double precision](#double-precision-on-nvidia). Agreement with the CPU path is therefore at the single-precision level, a few times `1e-4` on field amplitudes for a saturating run, and the section on [accuracy](#what-agreement-to-expect) puts that number in context against the things a user changes without thinking about them.

- [Requirements](#requirements)
- [Building](#building)
- [Switching it on](#switching-it-on)
- [What is supported](#what-is-supported)
- [Collective effects and other physics](#collective-effects-and-other-physics)
- [The worked example](#the-worked-example)
- [What agreement to expect](#what-agreement-to-expect)
- [Performance notes](#performance-notes)
- [Running on more than one GPU](#running-on-more-than-one-gpu)
- [Where the code lives](#where-the-code-lives)
- [Porting to another GPU](#porting-to-another-gpu)

## Requirements

**NVIDIA.** A card of compute capability 7.0 or newer, a driver, and the CUDA toolkit. The backend is compiled for the architectures in `CUDA_ARCHS`, which defaults to `80;89;120;120-virtual` — an A100, an L4, and the consumer Blackwell parts such as the RTX 5080, plus PTX so that a newer card still runs by JIT rather than failing to launch. Nothing is compiled at run time, so a binary built this way needs only the driver on the machine it runs on.

Memory is the practical limit rather than the architecture. The resident state is `nslice * ngrid^2` complex floats per harmonic plus about as much again in scratch, and the line printed at the start of a `&track` block reports it; a deck that does not fit is refused by name, with what it wanted and what was free. Splitting the run over more MPI ranks divides that, and over more cards divides it again — see [running on more than one GPU](#running-on-more-than-one-gpu).

**Apple.** An Apple Silicon Mac. The backend refuses to start on a device without unified memory, which rules out the Intel Macs with discrete GPUs. The command line developer tools supply the Metal framework headers and are installed with `xcode-select --install`; a full Xcode installation also works. The offline `metal` compiler is not needed, because the shaders are compiled from source when the first `&track` block starts.

The remaining dependencies are the same as for a CPU build in both cases: a C++17 toolchain, MPI, HDF5 and FFTW.

## Building

The conda-forge toolchain is the least surprising way to get the dependencies to agree with each other.

### NVIDIA

```sh
conda create -n genesis4-cuda -c conda-forge \
    cxx-compiler c-compiler cmake make pkg-config \
    mpich "hdf5=*=mpi_mpich_*" fftw \
    cuda-nvcc cuda-cudart-dev cuda-version=13.2
conda activate genesis4-cuda
```

Match `cuda-version` to what the driver supports; `nvidia-smi` prints it in the top right. A toolkit newer than the driver's CUDA version usually still works within the same major release, but there is no reason to find out the hard way.

```sh
cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DENABLE_CUDA=ON
cmake --build build-cuda -j8
```

Look for `-- CUDA GPU backend enabled, architectures 80;89;120;120-virtual`. Add `-DCUDA_ARCHS=native` to compile only for the card in this machine, which is quicker and is what to use while developing; `-DCUDA_ARCHS=80` or similar pins one target for a cluster whose nodes are uniform.

CMake 3.18 or newer is required with `ENABLE_CUDA`, for `CMAKE_CUDA_ARCHITECTURES`. The build pins `nvcc`'s host compiler to `CMAKE_CXX_COMPILER`, because `nvcc` otherwise picks the system compiler and the two then disagree about the C++ ABI in a way that only shows up at link time.

### Apple Silicon

```sh
conda create -n genesis4-dev -c conda-forge \
    cxx-compiler c-compiler cmake make pkg-config \
    mpich "hdf5=*=mpi_mpich_*" fftw
conda activate genesis4-dev

cmake -S . -B build-metal \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DENABLE_METAL=ON
cmake --build build-metal -j8
```

Look for `-- Metal GPU backend enabled` in the configure output. `ENABLE_METAL` defaults to `OFF` and is a hard error on anything that is not macOS.

Two properties of `CMakeLists.txt` are worth knowing on a Mac.

The first is that MacPorts can hijack the build. If `/opt/local/bin/h5pcc` exists, it is taken as the C++ compiler in preference to whatever else was chosen, which is right on a machine where MacPorts is the whole toolchain and wrong on one where it is merely also installed: the C compiler and every flag still come from the conda environment, and the link then goes out against two HDF5 and MPI installations. Everything compiles and the failure arrives at the very end, as a page of undefined `_ompi_mpi_*` symbols that says nothing about its cause. Configure now warns when the two compilers come from different places; if that describes your machine, add `-DUSE_MACPORTS_H5PCC=OFF` and check that the output says `MPI Found` and `HDF5 Found` rather than naming a wrapper.

The second is that if FFTW is not found the build still succeeds, but a deck asking for `fft_fieldsolver = true` is then given the ADI solver without being told, so check for `-- FFTW found` as well. That one applies to both backends.

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

At the start of a GPU `&track` block one line names the device and the resident footprint, and at the end another reports what the device did.

```
CUDA backend: NVIDIA GeForce RTX 5080, 1032 MB resident, gamma_ref = 11357.8
  host transfer check: field 2.9e-08   beam 4.4e-08 (relative, FP32 rounding is ~1e-7)
...
CUDA: 196 steps in 1.71 s, device busy 1.62 s (95%)
```

The first figure is the wall clock of the tracking loop and the second is the time the device itself spent executing. The percentage answers the question worth asking before reaching for more hardware: a device that is already busy will not go faster if more MPI ranks are pointed at it, and one that is idling means the host is the limit. Neither figure is Genesis' own `Total Wall Clock Time`, which covers the whole program including loading and output.

Read the percentage as a saturation indicator rather than as a calibrated fraction, particularly on CUDA. Metal timestamps every command buffer, so its figure is the sum of the intervals the device was executing. CUDA brackets each engine call with a pair of events, which keeps host work between calls out of the measurement, but the event records cost device time of their own and the device timestamps and the host clock need not agree to better than a few percent: a saturated run reports between 100% and 106%. What the number answers is 95% against 50%, not 95% against 100%.

A deck that asks for the GPU and cannot have it is an error rather than a silent fallback, because the alternative is a run that quietly produces CPU numbers and CPU timings under a GPU label. Each message names the reason.

```
*** Error: gpu = true in &track, but this binary was built without the GPU
    backend. Reconfigure with -DENABLE_CUDA=ON for an NVIDIA card or
    -DENABLE_METAL=ON on an Apple Silicon Mac
*** Error: gpu = true in &track, but ngrid = 151 is not supported by the CUDA
    field solver, which handles powers of two from 64 to 1024. Set ngrid = 128
    in &field. ...
*** Error: gpu = true in &track, but ngrid = 2048 in &efield, but the GPU
    space-charge solve holds the radial arrays in shared memory and NVIDIA
    GeForce RTX 5080 has room for 3 to 1583
*** Error: gpu = true in &track, but the resident buffers do not fit: 21140 MB
    wanted, 15290 MB free of 16303 MB on NVIDIA GeForce RTX 5080. Spread the
    run over more MPI ranks, and more cards if there are any, or reduce ngrid
    or the particle count
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
| space charge (`&efield`) | supported, long and short range; the radial grid is limited by the shared memory of the device, 384 points on Metal and 1583 on a card with 100 KB per block |
| `one4one` | not supported; the backend requires a rectangular particle array |

Everything the backend cannot do is a hard error rather than a fallback, and the reason is the same in each case. A run that completes and writes a plausible output file, having quietly done something other than what the deck asked for, is the failure this design works hardest to avoid. An ADI deck would be propagated by FFT instead, and a `bunchharm` above eight would be answered from host particle arrays that are stale, because the particles stay on the GPU. Neither would announce itself in the output file.

No lattice element falls back to the CPU. The machinery for a fallback is still present, and a step that took it would be counted and reported at the end of the run, but nothing currently triggers it.

### The transverse grid

The restriction to powers of two is the one users notice. Genesis decks traditionally use an odd `ngrid` so that a grid point sits exactly on the axis, but that convention buys nothing physically and costs a great deal in transform structure: `ngrid = 255` factors as 3 x 5 x 17, which is a poor length for an FFT and runs about 1.5 times slower than 256 even on the CPU. The error message names the nearest supported size.

Each grid size gets its own specialisation of the transform. The kernel is a four-step Cooley-Tukey decomposition `N = REGS x LANES`, in which every thread holds `REGS` points in registers, performs a short DFT over them, exchanges through shared memory, and finishes with `LANES`-point DFTs. The two factors are chosen per grid size, so there is no runtime branching in the inner loop: Metal injects them into the shader as preprocessor macros when it compiles the library at startup, and CUDA takes them as template parameters and instantiates every supported size at build time. The two backends use the same `REGS x LANES` for a given grid and therefore perform the same operations on the same operands in the transform; they differ in how many transforms share a block, which is a blocking choice and affects nothing but speed. They are not bit-identical to each other, because their block reductions fold a warp in a different order — Metal calls `simd_sum` where CUDA uses a butterfly of `__shfl_xor_sync` — and that is the whole of the difference between the two backends' agreement figures with the CPU.

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

The short-range solve is a set of azimuthal modes `m` and longitudinal modes `l`. For each pair the particles of a slice are binned in radius into a complex source term, a tridiagonal system is solved on the radial grid, and the result is gathered back onto the particles. That maps to one block per slice with the radial arrays in shared memory. The tridiagonal solve is a recurrence, so a single thread runs it while the others wait; the grid is small enough that the parallel alternatives cost more than they save.

One part cannot stay on the device. `rmax` grows to hold the widest slice seen so far, and it does so as the slices are visited in order and persists across the whole run, so slice *k* is solved on a grid that already accounts for the slices before it. A backend that sized each slice independently would agree with the CPU only until the first slice wider than the grid. An analysis pass therefore reduces each slice to a centroid and a bounding radius, the host replays that growth exactly as `analyseBeam` does, including the message when the grid is enlarged, and the resulting spacing comes back per slice. That costs one round trip per step and three floats per slice, rather than the particles.

The radial grid is limited by the shared memory of one block, because the six complex and four real arrays of the solve come to 64 bytes per point and they all have to live there: 384 points on Metal's 32 KB, and 1584 on a card offering 100 KB. The CUDA backend reads the limit from the device rather than assuming one, opts in to the larger allocation where that is needed, and names the actual number when it refuses. The default is 100, so this is a limit few decks reach.

The `SSCfield` diagnostic deserves a warning, because it is the natural thing to check and it is misleading in two separate ways. It reports the `l = 1` mode, and at the head of a run there is no bunching, so its source term is a sum of thousands of unit phasors that should cancel to nothing; what survives is the arithmetic, which is 1e-18 in the CPU's double precision and 1e-10 in the GPU's single precision, a ratio of 1e8 between two numbers that are both zero. It also reports whichever azimuthal mode came last, since the CPU writes it once per `m`, so with `nphi > 0` it is a dipole or higher and averages to noise for a round beam. Compared where the field is real, on the same bunched particles, the two paths agree to 4.0e-07 of peak with a correlation of 1.000000000.

## The worked example

Everything below is in `examples/gpu/` and takes a few minutes.

### 1. The self-check

```sh
cd examples/gpu
export FI_PROVIDER=tcp
../../build-cuda/genesis4 validate.in       # or ../../build-metal/genesis4
```

Set `FI_PROVIDER=tcp` before anything else if your MPICH goes through libfabric, as conda's macOS build does; the [performance notes](#performance-notes) explain why, and how to tell whether it applies to your build at all. Where it does apply it is worth 40% even on this single-rank run, which performs no communication whatever, and a factor of four on a run with several ranks.

This is a steady-state run of the ARAMIS undulator with the shot noise switched off and a seeded field, so it is deterministic, and it runs both paths and compares them at every step. The last line is the point of it.

```
CUDA vs CPU over 1104 steps: max relative error, field 0.000212, beam 3.18e-06
Metal vs CPU over 1104 steps: max relative error, field 0.000207, beam 3.18e-06
```

Those two numbers are the single-precision round-off level accumulated over 1104 steps. That the two backends land within 2% of each other on the field and agree to three figures on the beam is the point of writing them as transcriptions of one another: the residual is the arithmetic, not the vendor. If the run on your machine reports something similar then the backend is working. A number above about `1e-3` means it is wrong rather than merely less precise.

At the start it also prints the device it selected and a check that the upload round trip is clean before any physics happens.

```
CUDA backend: NVIDIA GeForce RTX 5080, 1 MB resident, gamma_ref = 11357.8
  host transfer check: field 2.9e-08   beam 4.4e-08 (relative, FP32 rounding is ~1e-7)
```

The timing line from this deck is not a performance figure and should not be read as one, because `gpu_validate` runs the CPU path as well and a steady-state deck has a single slice, so there is very little for the device to do. The performance notes give the same line for a deck that is representative.

### 2. Timing and a real comparison

`sase_cpu.in` and `sase_gpu.in` are the same 500-slice SASE run, 10 m of ARAMIS at `ngrid = 256` with diagnostics at every step, differing only in the `gpu` flag and the rootname. The shot-noise seed is fixed in both.

```sh
export FI_PROVIDER=tcp
mpirun -n 8 ../../build-cuda/genesis4 sase_cpu.in
mpirun -n 8 ../../build-cuda/genesis4 sase_gpu.in
python3 compare.py sase_cpu.out.h5 sase_gpu.out.h5
```

Run both at the same number of ranks. Genesis pads the slice count up to a multiple of the rank count and distributes the slices accordingly, so the shot-noise realisation depends on the rank count, and two CPU runs of this deck at 8 and at 12 ranks differ by 10% in `Field/power` and by 91% in per-slice `Beam/bunching`. Comparing across rank counts measures that rather than the GPU. `compare.py` will say so if the array shapes end up mismatched, but if they happen to match it cannot tell.

`compare.py` needs `h5py` and `numpy`, which the build environment above does not have. Use whichever environment you normally analyse Genesis output in, or add them to it.

```sh
conda install -n genesis4-cuda -c conda-forge h5py numpy
```

### 3. The full validation matrix

`sweep.py` runs seventy-three decks against the CPU path on the same inputs and prints one line per case. Sixty-two of them compare per-step differences under `gpu_validate`, seven being decks the backend must refuse; the other eleven run to completion and compare the output against a bound derived from the per-step figure and the step count. A case that falls back to the CPU where it should not is a failure, because a fallback gives the right answer by the slow route and is exactly how an unported element would hide.

One case also carries a check of a third kind, because comparing against the CPU cannot police everything. A quantity whose CPU value is itself too small or too chaotic to compare against slips through both of the other tiers: a bunching factor at a high harmonic is a cancelling sum near zero early in a run, where two correct answers differ relatively by order one, and chaotic late in a saturating one, where it exceeds any round-off bound however the deck is arranged. Such a case is instead run twice on the GPU with one option changed, and the datasets that option cannot legitimately reach have to come out the same. `aux_slot_layout` uses that to check the diagnostic buffer, where the bunching factors and the auxiliary extrema are neighbours: switching the extrema off must leave every bunching factor exactly where it was, which is a statement no comparison against the CPU can make.

```sh
cd examples/gpu
python3 sweep.py --genesis ../../build-cuda/genesis4 --workdir /tmp/g4sweep
```

It needs the same `h5py` and `numpy` as `compare.py` and takes some tens of minutes. Pass `--tier step` or `--tier run` for one tier only, and `--only <regex>` to select cases by name. This is the check to run after changing anything in the backend.

The matrix covers planar and helical undulators, tapers, phase shifters, undulator and quadrupole offsets, gradients and roll-off, field and orbit errors, every supported grid size, harmonics, the source filter, all four wakefield models, incoherent radiation, long-range and short-range space charge, chicanes, correctors, time-dependent and time-periodic running, dumps and multiple `&track` blocks.

`tools/fftcheck` is a smaller and more direct check of the FFT kernels alone: it compares a row transform, a column transform, the out-of-place row pass the far-field diagnostic uses and a complete four-pass solve against a direct double-precision DFT. It is not part of the build, since it is a diagnostic rather than a regression test, and it is what to reach for when the sweep says the field is wrong but not where. Each version tests the code Genesis actually runs rather than a copy of it -- `fftcheck.cu` includes the same `src/Core/CudaFFT.cuh` the field solve does, and `fftcheck.mm` extracts the shader string out of `src/Core/MetalEngine.mm`.

```sh
nvcc -std=c++17 -O2 -arch=native -Isrc/Core tools/fftcheck.cu -o /tmp/fftcheck
/tmp/fftcheck            # every supported shape
/tmp/fftcheck 256        # just one
```

```sh
clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
    -O2 tools/fftcheck.mm -o /tmp/fftcheck
/tmp/fftcheck 256 16 16 8 16      # ngrid lanes regs rowsPerTG colsPerTG
```

Run either from the top of the source tree. The CUDA one enumerates the shapes itself from `pickFFTShape`; the Metal one has to be given them, and they are `64 8 8 16 16`, `128 8 16 16 16`, `256 16 16 8 16`, `512 16 32 4 8` and `1024 32 32 2 4`. All five report a row and column error of a few times `1e-6` against a scale of order ten, and a round trip through the complete solve accurate to `4e-07`.

```
N=256   16x16 rows:  max abs err 1.769e-06  (scale 1.549e+01)  -> ok
N=256   16x16 cols:  max abs err 1.629e-06  (scale 1.517e+01)  -> ok
N=256   16x16 solve: max abs err 3.210e-07  (scale 7.055e-01)  -> ok
N=256   16x16 out:   max abs err 1.822e-06  (scale 1.448e+01)  -> ok
```

## What agreement to expect

The reference numbers below are from an RTX 5080 with both runs at 8 ranks, with the M3 Max figures for the same deck beside them. Nothing here should be much different on any other card, since the arithmetic is the same on all of them; that the two columns agree to within a couple of percent, on a quantity that is a round-off difference amplified by the SASE gain, is the evidence for that.

```
amplitudes, relative                         error       scale        Metal
  Field/Global/intensity-farfield       6.722e-04   3.447e+20     6.614e-04
  Field/intensity-farfield              6.557e-04   1.768e+21     6.533e-04
  Field/ydivergence                     6.055e-04   1.599e-05     6.055e-04
  Field/xdivergence                     4.219e-04   1.590e-05     4.219e-04
  Field/intensity-nearfield             3.683e-04   1.104e+16     3.950e-04
  Beam/bunching                         1.922e-04   1.241e-02     1.927e-04
  Field/power                           1.472e-04   1.338e+07     1.448e-04
  Field/xsize                           4.336e-05   7.737e-05     4.592e-05
  worst of 51: 6.722e-04                                          6.614e-04
```

Three things make a raw dataset-by-dataset comparison misleading, and `compare.py` separates them for that reason.

Amplitudes are the honest number, taken relative to their own peak. A few times `1e-4` on a SASE run is the expected level, because SASE amplifies exponentially and a `1e-7` difference in the first metre is not a rounding error at the end of the undulator but a rounding error multiplied by the gain. The steady-state figure of `2e-4` over 1104 steps is the same effect.

Centroids such as `xposition` and `pointing` average to zero for a symmetric beam, so their magnitude is set by round-off in the first place and a relative error against them means nothing. `Beam/xposition` on this deck is `2e-11` m against a beam 25 µm across. `compare.py` prints these as an absolute difference beside the scale of the quantity itself.

Phases are defined modulo 2π and are undefined where there is no amplitude to carry them. The near-field phase in particular is the argument of a single on-axis cell, which passes through zero at optical vortices while the slice as a whole is still bright. `compare.py` wraps them and masks out the points where the companion intensity is below a thousandth of its peak; what remains agrees to about `1e-2` radians.

The on-axis quantities deserve a note of their own, since a GPU run is where the problem showed. `intensity-nearfield` and `phase-nearfield` report the field in the cell at the centre of the grid, and that cell used to be selected as `(ngrid*ngrid-1)/2`, which is the centre only for an odd `ngrid`. For an even one it lands at the edge of the grid, where the field is orders of magnitude weaker and a percent-level difference between two runs means nothing. Since this backend accepts only powers of two, every GPU run encountered it. The index is now `(ngrid/2)*ngrid + ngrid/2`, which is unchanged for odd grids, and the end-to-end agreement on the validation cases improved by an order of magnitude with it.

The CPU is also not a fixed target to compare against. `FieldSolverFFT` plans its transforms with `FFTW_MEASURE`, so two runs of the same CPU binary on the same machine differ by `1e-15` to `1e-9` depending on which plan the planner happened to select. When chasing a discrepancy, run the same binary twice first and use that as the control.

A GPU run is not a fixed target either, and this surprises people more. Two runs of the same deck with the same binary on the same card do not agree to the last bit: on the 1104-step steady-state deck they are identical for the first two dozen integration steps and then part company, reaching a few times `1e-7` by the end. The cause is the source deposition, which adds each particle's contribution to four grid cells with an atomic add. The additions commute but their rounding does not, and the order in which thousands of threads reach the same cell is not fixed, so the sum differs in its last bit or two. It only becomes visible once the field spans a wide range of magnitudes, which is why it appears part way into a run rather than at the first step. Both backends do this and both behave the same way. It is two orders of magnitude below the difference from the CPU that single precision already produces, so it changes no conclusion, but it does mean that a GPU run should not be used as a bit-exact reference for another GPU run.

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

End-to-end wall clock of `sase_cpu.in` and `sase_gpu.in`, which are 500 slices at `ngrid = 256` over 196 steps with diagnostics at every step.

| ranks | RTX 5080, CPU | RTX 5080, GPU | M3 Max, CPU | M3 Max, GPU |
|---:|---:|---:|---:|---:|
| 1 | 268.0 s | 3.6 s | 309.1 s | 5.2 s |
| 8 | 35.8 s | 2.2 s | | |
| 12 | | | 35.2 s | 4.4 s |
| 16 | 25.6 s | | | |

The NVIDIA column is a Core Ultra 9 285K with 24 cores and an RTX 5080; the Apple one is an M3 Max with 12 performance cores, 4 efficiency cores and a 40-core GPU. About one to two seconds of every figure is setup, loading and the output file, on both paths. On the tracking loop alone the RTX 5080 takes 1.42 s against about 24 s for sixteen CPU cores and about 266 s for one, so the card is worth roughly seventeen times all sixteen cores and 190 times one of them, and it leaves the cores free while it works.

Between the two GPUs the tracking loop is 1.42 s against 4.10 s, a factor of 2.9. The manual's own prediction, made before the CUDA backend existed and derived only from the memory bandwidth of the two devices, was 1.7 s. The transform is memory bound, so bandwidth is the first-order predictor and it got the sign and the order right; the remaining factor is that the RTX 5080 has more shared memory per block than Metal's 32 KB and the blocking was retuned for it.

The grid-size scaling on the RTX 5080, same deck at one rank:

| `ngrid` | REGS x LANES | tracking loop | vs M3 Max |
|---:|---:|---:|---:|
| 64 | 8 x 8 | 0.36 s | 2.6x |
| 128 | 16 x 8 | 0.50 s | 3.2x |
| 256 | 16 x 16 | 1.40 s | 2.9x |
| 512 | 32 x 16 | 4.80 s | 3.3x |
| 1024 | 32 x 32 | 19.2 s | 3.5x |

Each row has four times the grid points of the row above it and costs between three and four times as much, which is what a memory-bound transform should do; the smaller grids cost less than that only because they do not fill the machine.

Measure with a clock rather than with `clock()`. Genesis' `Total Wall Clock Time` was processor time until recently, which is the same thing for a CPU run that never waits and badly wrong for a GPU run that does: this deck reported 1.2 s while actually taking 5.2 s, because waiting costs no processor time. That line is now genuine wall clock. Earlier versions of this chapter quoted the old figures and overstated the GPU by about a factor of four.

Read the busy percentage as a saturation indicator rather than as a calibrated fraction. It comes from device timestamps and the wall clock from the host, and the two clocks need not agree to better than a few percent; on the WSL2 machine these figures were taken on, a saturated run reports between 100% and 106%. What the number is for is the difference between 95% and 50%, not between 95% and 100%.

More MPI ranks against one GPU achieve nothing on this deck. The tracking loop takes 1.42 s at one rank and 1.38 s at four, and the report at the end of the run says why: at one rank the device is already saturated, so the extra ranks only divide up work they then queue for. Use one rank per GPU unless the deck needs the memory of several, or unless the busy figure says the host is the limit; [running on more than one GPU](#running-on-more-than-one-gpu) covers both cases.

Where the 4.1 s of the Metal loop goes, and the shape is the same on CUDA: 37% is the per-slice diagnostics, which `output_step = 100` reduces from 4.04 s to 2.53 s, 6% is the source deposition, and the rest is the four FFT passes, the Runge-Kutta push and the transverse map. The field solve runs close to the memory bandwidth of the machine, so the diagnostics are where the remaining headroom is, and `output_step` is the knob that costs nothing to turn. Most of the diagnostic cost is the far-field branch, which transforms every slice a second time.

A small deck is a different problem, and used to be a bad one. The 500-slice example performs 19 ms of real work per step, so what it costs to hand work to the device does not matter. A steady-state deck is a single slice and a few tens of microseconds of arithmetic, and there the handover is everything. Metal fixed this by encoding a whole step into one command buffer and submitting once, where it previously submitted and waited four times; CUDA never had the problem, because launches on a stream are asynchronous and the whole step is queued before anything is waited on.

| 1104-step steady-state deck | Metal, one submission per pass | Metal, one per step | CUDA |
|---|---:|---:|---:|
| tracking loop | 3.91 s | 1.08 s | 0.73 s |
| device busy | 2.62 s | 0.54 s | 0.55 s |

The same deck takes 3.70 s on one core of the 285K and 3.75 s on one of the M3 Max, so a steady-state run is five times faster on the GPU rather than marginally slower. That the Metal device busy time fell by as much as its wall clock, and that CUDA's busy time matches the improved Metal figure while its wall clock is lower still, says the same thing twice: most of what was being attributed to the device was the cost of starting work rather than of doing it. The remaining floor is a few hundred microseconds per step for the fifteen or so launches a step makes, so the next thing to try for small decks is fewer launches rather than faster ones.

A step the backend refuses would cost more than it looks, because the particles and the field would have to come back to the host before it and go out again afterwards, making that step slower than the same step on the CPU alone. Correctors used to be refused, and since `orbiterror = true` is implemented as a corrector at every step, an orbit-error deck spent 187 of its 196 steps that way and lost about two thirds of the benefit of the GPU. Nothing falls back today.

`export FI_PROVIDER=tcp` is not optional if your MPICH talks through libfabric, which conda's macOS build does by default. Its default provider busy-polls while waiting, and the faster the compute becomes the more of the machine that wastes. The variable is a libfabric one, so it is worth checking whether it applies before concluding anything from a timing: `mpichversion | grep Device` reports the netmod, and a build configured `ch4:ucx` never reaches libfabric, which makes `FI_PROVIDER` inert. Conda's Linux MPICH is built `ch4:ucx,ofi` and prefers UCX, so on a cluster the knob is usually `UCX_TLS` instead — `UCX_TLS=self,sm` confines a single-node run to shared memory — and `UCX_LOG_LEVEL=error` silences the RoCE probe warnings UCX prints at startup, which are harmless and unrelated to Genesis.

| | without | with |
|---|---:|---:|
| `validate.in`, serial, no `mpirun` | 8.9 s | 6.3 s |
| `sase_gpu.in`, 8 ranks | 18.9 s | 5.0 s |

The second row is the one that matters, since eight ranks waiting on each other and on one GPU spend nearly four times the wall clock spinning. The first has no MPI communication in it at all, being one rank launched directly, and still loses 40%. Put the export in your shell profile. The size of the penalty varies with the MPICH build; an earlier one lost a factor of four even on the serial row.

Startup is not free but it is not the problem either. The Metal shaders are compiled from source when the first `&track` block starts, in another process, and the cost does not scale with the problem. On this deck the whole fixed cost of a GPU run, covering loading, the shader compile and the output file, is about 1 s against about 0.9 s for the CPU path, whose own fixed cost is mostly FFTW planning its transforms. Neither is worth optimising unless the tracking loop is shorter than either.

## Running on more than one GPU

Genesis parallelises over slices with MPI and each rank constructs its own engine and owns its own slices, so a job spread over several cards is the arrangement that already exists with a different device attached to each rank. Nothing in `GPUEngine` or in the tracking loop assumes there is only one device, and nothing in a deck has to change.

### How the device is chosen

The CUDA backend picks its device from the rank's position within its node, not from its rank in the job. It splits `MPI_COMM_WORLD` with `MPI_Comm_split_type` and `MPI_COMM_TYPE_SHARED`, takes the rank within that communicator, and selects `local_rank % devices_visible`. Using the global rank instead would put every rank of a four-node job on device 0 of its node and leave three quarters of the hardware idle, which produces a correct answer at a quarter of the speed and is invisible unless someone looks — so the line printed at the start of a `&track` block says what happened.

```
CUDA backend: NVIDIA A100-SXM4-40GB (rank 0 of 8 on this node -> cuda:0 of 4,
    2 ranks per device), 5310 MB resident, gamma_ref = 11357.8
```

Only rank 0 prints, so the line reports its own device and the shape of the whole node's mapping rather than a list. A run on a single card with a single rank prints just the device name.

Two overrides exist. `CUDA_VISIBLE_DEVICES` is handled by the CUDA runtime and needs nothing from Genesis, since it simply changes what the backend can see; this is the right tool when a scheduler has already assigned cards. `G4_CUDA_DEVICE` sets the device for one process outright and is for a launcher that pins ranks itself. Both are reported in the line above when they take effect.

### How many ranks per GPU

One rank per GPU is the right default, because a single rank already saturates the device. On the 500-slice example on an RTX 5080 the tracking loop takes 1.42 s at one rank and 1.38 s at four, all pointed at the same card.

The exception is a deck whose host-side work is significant, and the report line identifies it: a device busy percentage well below 100% means the host, not the device, is setting the pace. Incoherent radiation with `doSpread` is that case in both backends, since the draws are taken on the host to keep the noise realisation identical to the CPU's. Oversubscribing ranks divides the host work and leaves the device work where it was, so it helps that deck and does nothing for one that is already device-bound. Choose the rank count from the busy percentage rather than from the core count.

### What splitting the job buys and costs

Splitting across GPUs splits the memory, since each rank holds only the slices it owns, and on cards with modest memory that is the more important consequence. The 500-slice example needs 612 MB as one rank and 153 MB as four. Scaled up, a deck at `ngrid = 1024` with four harmonics needs about 21 GB of field and scratch, which does not fit on a 16 GB RTX 5080 or a 24 GB L4 as one rank but fits comfortably as four, on one card or on several. A deck that does not fit is refused with what it wanted and what was free, rather than failing inside an allocation.

The scaling is not free. The slippage exchanges one field slice per step between neighbouring ranks, which on a single device is a pull and a push through the host and between devices becomes a host round trip on each side unless the MPI is CUDA-aware. At `ngrid = 256` that slice is 512 KB. The wakefield and the long-range space charge additionally perform an `MPI_Allgather` over all slices at every step, which is host-side work that does not shrink as ranks are added and which sets a floor for those decks. And as ranks multiply the work per rank falls: 500 slices across sixteen cards is 31 slices each, against launch and synchronisation overheads of the order of 0.1 ms per step. Expect the useful limit to arrive well before the arithmetic says it should, and measure rather than extrapolate.

One caution when comparing runs across configurations: Genesis pads the slice count to a multiple of the rank count and distributes the slices accordingly, so the shot-noise realisation depends on the rank count. A four-rank run and a sixteen-rank run of the same deck are different noise seeds, and the difference between them is much larger than anything the GPU contributes. Compare like with like.

## Where the code lives

Eight files, of which two are the Apple backend and three the NVIDIA one.

| | |
|---|---|
| `include/GPUEngine.h` | the interface the tracking loop talks to, and the design constraints any backend has to respect |
| `src/Core/GPUEngine.cpp` | which backend is compiled in and how it is created; one branch per backend |
| `include/CudaEngine.h`, `src/Core/CudaEngine.cu` | the NVIDIA backend: the kernels and the host code that drives them |
| `src/Core/CudaFFT.cuh` | the transform, in a header so that `tools/fftcheck.cu` exercises the same code the field solve does |
| `include/MetalEngine.h`, `src/Core/MetalEngine.mm` | the Apple Silicon backend: the Metal shaders and the host code that drives them |
| `include/SliceMoments.h` | the per-slice diagnostic moments, in the normalisation `DiagBeam` and `DiagField` expect |
| `src/Core/Gencore.cpp` | the tracking loop, which contains no device-specific code and no preprocessor conditionals |

`Gencore.cpp` is worth reading against the interface rather than against either backend. It holds a `GPUEngine *` and has no `#ifdef` in it: the device-specific half of every decision is inside the backend, and the loop's half is only ever whether the host needs the data now. Adding CUDA changed four lines of `GPUEngine.cpp` and nothing at all in the tracking loop or in the physics classes.

The comments at the head of both backends record the mistakes that have already been made once. The order of operations within a step has to mirror `Beam::track` exactly, down to the corrector kick landing before the longitudinal momentum factor is formed and only on the closing half step. Dispatch geometry has to be parameterised at every dispatch site, because hard-coding it at one site only breaks at grid sizes other than the one being developed against. And in Metal the parameter structs are written twice, once in the shader string and once in host C++ immediately below it, so editing one alone skews the buffer layout silently and produces plausible but wrong physics; that particular hazard does not exist in CUDA, where host and device share the declaration.

## Porting to another GPU

A second backend is a new implementation of `GPUEngine` and one branch in `GPUEngine::create()`. Nothing in the tracking loop changes, and nothing in the physics classes changes either: the host-side work that the collective effects need has already been separated from the particle work, so a CUDA backend calls the same `Beam::computeWakeLoss`, `Beam::computeIncoherentKick`, `Beam::planShortRangeSC` and `TrackBeam::chicaneMatrix` that the Metal one does. The seams are cut; what remains is the device code.

### What has to be implemented

`GPUEngine` declares twenty methods, and they fall into four groups. The setup group is `init`, which allocates the resident buffers and reports by name anything the deck asks for that the backend cannot do. The transfer group is `upload`, `uploadBeam`, `download`, `downloadField`, `downloadBeam` and the two single-slice transfers the slippage uses. The step group is `fieldStep` and `beamStep`. The diagnostic group is `beamMoments` and `fieldMoments`, which produce the per-slice reductions in the normalisation `SliceMoments.h` documents, plus `compare` for `gpu_validate`, `deviceName`, `gammaRef`, `bytesResident` and `deviceSeconds`.

The Metal backend is about 2,000 lines, of which roughly half is shader source. A reasonable order of work is the field solve first, since it is the largest single piece and `tools/fftcheck.mm` can check it in isolation, then the beam step, then the diagnostics, then the collective effects one at a time. Each of those can be validated on its own with `sweep.py --only <regex>` before the next is started.

### The four constraints that carry over

The beam and the field must stay resident for the whole `&track` block. Marshalling the host arrays in and out at every step costs 38 ms per step on this problem against about 11 ms of compute, so a solver plugged in behind `FieldSolver::advance`, which would have to copy on every call, cannot pay off however fast its kernels are. The beam and the field are coupled in both directions at every step, since the deposition reads the particles and the Runge-Kutta push gathers the field, so neither can move to the device without the other.

The diagnostics have to be reduced on the device as well. They are not a small share of the arithmetic once the tracking is fast, being 37% of the tracking loop at `output_step = 1`, and leaving them on the host would break residency in any case, because they read every particle and every grid point.

Unsupported physics must be refused by name rather than worked around. A run that completes and writes a plausible output file having quietly dropped an effect the deck asked for is the worst failure mode available here, and it is what the hard errors and the sweep's treatment of an unexpected CPU fallback exist to prevent.

Single precision has to be arranged for rather than merely accepted, if the backend uses it. Two reformulations are the ones to copy: gamma is carried as an offset from a reference energy, because at gamma = 11357 the single-precision quantum of absolute gamma is 1.35e-3 and the per-step energy change is 3.0e-4 at saturation, so storing absolute gamma gives errors between 65% and 276%; and the longitudinal momentum is formed as `gamma*sqrt(1-r)` rather than `sqrt(gamma^2-1-aw^2-p^2)`, since the single-precision quantum of gamma squared is about 15 and the subtraction loses the whole transverse contribution.

### What the discrete card changed

The CUDA backend is the second one and is worth reading as the answer to "how much of this was Apple specific?". The answer is: the memory model, and nothing else. Every kernel is a transcription, the transform shapes are the same, and the two backends agree with the CPU to the same figures. What the port actually consisted of is below, and a third backend should expect the same division.

The memory is not unified, so every buffer is a device allocation with a pinned host staging area beside it. Metal reads and writes its buffers directly, which is why the round trips are cheap enough there to be unremarkable; on a discrete card each becomes a PCIe transfer and a synchronisation. They happen every step and are worth listing.

| round trip | size per step | when |
|---|---|---|
| incoherent radiation draws | one float per beamlet, 2 MB for 500 slices at 8192 particles and `nbins = 8` | host to device, only with `doSpread` |
| space-charge analysis | three floats per slice down, one float per slice up | both directions, only with short-range space charge |
| space-charge diagnostic | one float per slice | device to host, only with short-range space charge |
| wakefield loss | one float per slice | host to device, only with `&wake` |
| long-range space charge | one float per slice | host to device, every step |
| slippage | one field slice, `ngrid^2` complex, 512 KB at `ngrid = 256` | both directions, once per slip event in a time-dependent run |

None is large, but each is an ordering constraint, and the rule that keeps them correct is the one Metal already needed for a different reason: nothing the host writes may be overwritten while an operation that reads it is still queued. `beamStep` therefore drains the stream once, at the top, before it writes anything, and everything the host has to compute for a step is done before the first thing is queued.

Two host reads that Metal gets for nothing had to be moved into kernels. The on-axis field cell, which `DiagField` needs for `intensity-nearfield`, was one strided read per slice out of the resident field; the near-field reduction now writes it into its own output, since it is already touching the slice. And the whole-state transfers convert between the host's FP64 arrays and the device's FP32 ones through a pinned buffer holding a run of slices, because doing it a slice at a time costs a synchronisation per slice and `gpu_validate` does the whole state at every step.

The rest was subtraction rather than addition. Metal has no threadgroup `atomic_float`, so the source deposition and the space-charge accumulation use a compare-and-swap loop on the bit pattern; CUDA has `atomicAdd` on a shared float, and two scratch arrays in the space-charge solve went away with it. Metal compiles its shaders from source when the first `&track` block starts, with the grid size and transform shape injected as preprocessor macros; in CUDA those are template parameters, every supported size is instantiated at build time, and the startup cost is gone. A threadgroup is a block, threadgroup memory is shared memory, a SIMD group is a warp, `simd_sum` is a warp shuffle and `threadgroup_barrier` is `__syncthreads`.

One limit stopped being a constant. The radial grid of the space-charge solve is bounded by the shared memory of one block, which Metal fixes at 32 KB and which on NVIDIA is a device property: 48 KB without opting in, 100 KB on an L4 or a consumer Blackwell part, 164 KB on an A100. The backend reads it from the device, sizes the allocation dynamically, opts in above 48 KB, and names the actual limit when it refuses.

`cuFFT` exists and the hand-written transform did not have to be reproduced; it was, because three things are fused into the passes here — the `expK` multiply on the inverse row pass, the source addition on the inverse column pass, and the filter multiply on the forward column pass of the source — and each would otherwise become a separate elementwise kernel reading and writing the whole field again. On this hardware the fused four-pass form runs at close to the memory bandwidth of the machine, so there was nothing to gain.

### Double precision on NVIDIA

Double precision is available on NVIDIA hardware, which changes what is worth doing, and the answer depends on the card. On an A100 it runs at half the single-precision rate and would give agreement with the CPU limited only by the order of operations, which would make `gpu_validate` a much sharper instrument. On an L4 or an RTX 5080 it runs at a sixty-fourth of the rate and is not worth having.

This backend is single precision on all of them. A double-precision variant would be a second engine rather than a flag on this one, because the FP32 reformulations above are woven through every kernel: `gammaRef` would be unnecessary, the detuning would be the literal expression rather than its series, and `gamma*beta_z` could be formed the way the CPU forms it. The interface already exposes `gammaRef`, which such a backend does not need but costs nothing to keep.

### Validating a new backend

Use the same instruments. `gpu_validate = true` runs both paths from the same state at every step and reports the largest relative difference, which is what makes a wrong kernel show up as an unmistakable jump rather than as chaotic growth; `sweep.py` drives that across the whole matrix and needs only `--genesis` pointed at the new binary. Nothing in the sweep is specific to a backend, and porting it to CUDA needed two changes, both of which were the sweep being too specific rather than the backend misbehaving: one case matched on Metal's word for shared memory, and one asked for a radial grid that Metal cannot hold but a 100 KB card can.

Three habits are worth adopting from these ports. When a comparison looks catastrophic, check whether the quantity being compared is meaningfully non-zero before believing it: the `SSCfield` case above spent a while looking like a 14% error when both numbers were zero to within their own arithmetic. When a before-and-after measurement comes out as exactly zero, suspect the measurement: building a reference binary with `git checkout` rather than a separate worktree carries staged changes across and silently compares a binary with itself. And when a case fails on the new machine and passed on the old one, find out which of the two was lucky before assuming it was the port: the five wakefield cases failed on Linux because `&wake` in a steady-state run read one past the end of a one-element vector, which the macOS allocator had been answering with a number that happened to work.

Two habits are worth adopting from this port. When a comparison looks catastrophic, check whether the quantity being compared is meaningfully non-zero before believing it: the `SSCfield` case above spent a while looking like a 14% error when both numbers were zero to within their own arithmetic. And when a before-and-after measurement comes out as exactly zero, suspect the measurement: building a reference binary with `git checkout` rather than a separate worktree carries staged changes across and silently compares a binary with itself.
