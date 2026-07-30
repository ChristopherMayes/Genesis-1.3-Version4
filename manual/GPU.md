# GPU acceleration on Apple Silicon

Genesis 4 has an optional GPU backend for Apple Silicon, written against Metal.
It is opt-in at build time and again per `&track` block, so a binary built with
it behaves exactly as before unless a deck asks for it.

Everything on the GPU is single precision. That is not a tuning choice: Apple
GPUs have no double precision at all, and the Metal shading language has no
`double` type. The agreement with the CPU is therefore at the FP32 level, a few
times `1e-4` on field amplitudes, which the section on
[accuracy](#what-agreement-to-expect) puts in context.

- [Requirements](#requirements)
- [Building](#building)
- [Switching it on](#switching-it-on)
- [What is supported](#what-is-supported)
- [The worked example](#the-worked-example)
- [What agreement to expect](#what-agreement-to-expect)
- [Performance notes](#performance-notes)
- [Where the code lives, and adding another backend](#where-the-code-lives-and-adding-another-backend)

## Requirements

- An Apple Silicon Mac. The backend refuses to start on a device without
  unified memory, which rules out the Intel Macs with discrete GPUs.
- The command line developer tools, for the Metal framework headers:
  `xcode-select --install`. A full Xcode install also works. The offline
  `metal` compiler is *not* needed — the shaders are compiled from source at
  startup, which costs about five seconds once per run.
- A C++17 toolchain, MPI, HDF5 and FFTW, exactly as for a CPU build.

## Building

The conda-forge toolchain is the least surprising way to get all four
dependencies to agree with each other on macOS:

```sh
conda create -n genesis4-dev -c conda-forge \
    cxx-compiler c-compiler cmake make pkg-config \
    mpich "hdf5=*=mpi_mpich_*" fftw
conda activate genesis4-dev
```

Then configure with the backend enabled and build:

```sh
cmake -S . -B build-metal \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DENABLE_METAL=ON
cmake --build build-metal -j8
```

Look for `-- Metal GPU backend enabled` in the configure output. `ENABLE_METAL`
defaults to `OFF` and is a hard error on anything that is not macOS.

Two things about `CMakeLists.txt` are worth knowing on a Mac. It will use
`/opt/local/bin/h5pcc` as the C++ compiler if that file exists, overriding
`CMAKE_CXX_COMPILER`, so a MacPorts installation can hijack a conda build.
And if FFTW is not found the build still succeeds but `FieldSolverFFT` produces
wrong results, so check for `-- FFTW found` as well.

## Switching it on

Two booleans in `&track`, both `false` by default:

```
&track
gpu = true
&end
```

`gpu = true` makes the GPU the answer. The beam and the field live in GPU
memory for the whole of that `&track` block; the host only sees them at a
`MARKER` dump, at the single slice the slippage exchanges each step, and once
at the end.

```
&track
gpu = true
gpu_validate = true
fft_fieldsolver = true
&end
```

`gpu_validate = true` additionally runs the CPU path every step and prints the
largest relative difference at the end. It is a test mode and is slower than
either path on its own, because it does both. `fft_fieldsolver = true` belongs
with it: without it the CPU half uses the ADI solver and the reported
difference measures the two solvers against each other rather than the two
processors.

At the end of a GPU `&track` block one line reports what the device did:

```
Metal: 196 steps in 4.09731 s, device busy 3.79265 s (93%)
```

The wall clock of the tracking loop, then the time the device itself spent
executing, which Metal timestamps on every command buffer at no cost. The
percentage answers the only question worth asking before reaching for more
hardware: a device that is already busy will not go faster if more MPI ranks
are pointed at it, and one that is idling means the host is the limit. Neither
number is Genesis' own `Total Wall Clock Time`, which covers the whole program.

A deck that asks for the GPU and cannot have it is an error, not a silent
fallback — otherwise the run would quietly produce CPU numbers and CPU
timings under a GPU label. The messages are explicit about the reason:

```
*** Error: gpu = true in &track, but this binary was built without the GPU
    backend. Reconfigure with -DENABLE_METAL=ON.
*** Error: gpu = true in &track, but ngrid = 151 is not supported by the Metal
    field solver, which handles powers of two from 64 to 1024. Set ngrid = 128
    in &field. ...
*** Error: gpu = true in &track, but incoherent synchrotron radiation
    (isr_loss/isr_spread) is not implemented on the GPU yet
```

## What is supported

The GPU runs the deposition, the field propagation, the transverse tracking,
the longitudinal push and all of the per-slice diagnostics. Around that:

| | |
|---|---|
| `ngrid` | a power of two from **64 to 1024** |
| field harmonics | up to 4, all sharing the same `ngrid`, `dgrid` and `gridmax` |
| `bunchharm` | up to 8 |
| `fft_fieldsolver` | must be `true`; there is no ADI solver on the GPU |
| `source_filter` | not supported |
| `one4one` | not supported; the backend wants a rectangular particle array |
| correctors | supported; the kick rides on the closing half step |
| chicanes | supported; the transfer map rides on the opening half step and the R56 shear sits before the closing one |
| wakefields (`&wake`) | supported, including the resistive wall |
| ISR, short-range space charge | hard error |

Everything in that table that is not supported is a hard error rather than a
fallback. The reason is the same in each case: the alternative is a run that
completes and writes a plausible output file having quietly done something
other than what the deck asked for. `source_filter` would be ignored, an ADI
deck would be propagated by FFT instead, and a `bunchharm` above 8 would be
answered from host particle arrays that are stale, because the particles stay
on the GPU. None of those announce themselves in the output file.

The `ngrid` restriction is the awkward one. Genesis decks traditionally use an
odd `ngrid` so that a grid point sits exactly on axis, but that convention buys
nothing physically and costs a lot in transform structure — `ngrid = 255` is
`3 x 5 x 17`, a pathological FFT length that is about 1.5 times slower than 256
even on the CPU. The error message names the nearest supported size.

Each grid size gets its own specialisation of the transform. The kernel is a
four-step Cooley-Tukey decomposition `N = REGS x LANES`: every thread holds
`REGS` points in registers, does a short DFT over them, exchanges through
threadgroup memory, and finishes with `LANES`-point DFTs. The two factors are
chosen per grid size and injected into the shader as preprocessor macros when
the Metal library is compiled, so there is no runtime branching in the inner
loop.

| `ngrid` | REGS x LANES | radices used | tracking loop | device busy |
|---:|---:|---|---:|---:|
| 64 | 8 x 8 | 8, 8 | 0.93 s | 82% |
| 128 | 16 x 8 | 16, 8 | 1.60 s | 85% |
| 256 | 16 x 16 | 16, 16 | 4.09 s | 92% |
| 512 | 32 x 16 | 32, 16 | 15.9 s | 98% |
| 1024 | 32 x 32 | 32, 32 | 67.4 s | 98% |

The SASE example at one rank, on an M3 Max. Each row has four times the grid
points of the one above, and from 256 upwards each doubling costs about four
times as much, which is what a memory-bound transform should do. The small
grids cost less than that only because they do not fill the machine: the busy
column is what says so, and it is the reason a small deck sees less of the GPU
than a large one does.

Agreement with the CPU degrades with grid size, but the grid is not really the
cause. Over the 1104 steps of the validation deck, holding `npart` at 8192, the
largest relative field difference is 9.4e-6 at `ngrid = 64`, 2.3e-4 at 256 and
8.5e-3 at 1024. What changes across that row is how many particles land in each
cell. A bilinear deposition of a fixed number of particles onto a finer mesh
leaves more shot noise in the source term, and that noise is high spatial
frequency, which is where single precision is weakest. Raising `npart` at fixed
`ngrid = 1024` brings the difference back down:

| `npart` | field difference |
|---|---|
| 8192 | 8.5e-03 |
| 32768 | 3.2e-03 |
| 131072 | 1.2e-03 |

So the rule is to scale `npart` with `ngrid` rather than to distrust the large
grids. A deck that puts a handful of particles in each cell is under-resolved
in double precision too; single precision merely makes it visible. The beam
moments are unaffected and stay at 3.2e-6 throughout, because they are a
reduction over particles and never touch the grid.

The field solver on the GPU is always the FFT one, and `fft_fieldsolver = true`
is required rather than assumed: a `gpu = true` run with the ADI solver selected
is refused, so that a deck cannot be propagated by a method it did not ask for.

### Wakefields

A `&wake` block works on the GPU, including the resistive wall wake, the
geometric and roughness wakes and the external `loss` term. The split of work
follows the structure of the physics rather than the structure of the code. A
wake is driven by the current of a slice, not by the coordinates of individual
particles, so the loss it produces is one number per slice and every particle in
that slice receives the same energy kick. Building the loss profile requires the
current of the whole bunch, which under MPI lives on several ranks and is
gathered with `MPI_Allgather`; applying it requires only an addition.

The gather and the convolution therefore stay on the host, exactly as on the CPU
path, and the GPU does the addition. Nothing has to come back from the GPU to
make this work, because the slice currents are fixed for the duration of a run:
particles do not migrate between slices. Residency is preserved.

The cost is small. On the 500-slice example the resistive wall wake adds under
3% to the runtime with `transient = false`, which is the host-side convolution
being computed once. With `transient = true` the convolution is repeated on
every step and the run takes about 2.6 times as long, all of it on the host.
That is the same host work the CPU path does, so it is not a GPU limitation, but
it does mean a transient wake dominates a run that is otherwise a second long.

## The worked example

Everything below is in `examples/metal-gpu/` and takes a few minutes.

### 1. The self-check

```sh
cd examples/metal-gpu
export FI_PROVIDER=tcp
../../build-metal/genesis4 validate.in
```

Set `FI_PROVIDER=tcp` before anything else if you are using conda's MPICH; see
[the performance notes](#performance-notes). It is worth 40% even on this
single-rank run, which does no communication at all, and a factor of four on a
run with several ranks.

This is a steady-state run of the ARAMIS undulator with the shot noise off and
a seeded field, so it is deterministic, and it runs both paths and compares
them every step. The last line is the point of it:

```
Metal vs CPU over 1104 steps: max relative error, field 0.000207, beam 3.18e-06
```

Those two numbers are the FP32 round-off level accumulated over 1104 steps. If
the run on your machine reports something similar, the backend is working. A
number above about `1e-3` means it is wrong, not merely less precise.

It also prints, at the start,

```
Metal backend: Apple M3 Max, 1 MB resident, gamma_ref = 11357.8
  host transfer check: field 4.5e-08   beam 4.4e-08 (relative, FP32 rounding is ~1e-7)
```

which confirms the device that was picked and that the upload round trip is
clean before any physics happens. The timing line from this deck,

```
Metal: 1104 steps in 6.09251 s, device busy 1.00931 s (17%)
```

is not a performance figure and should not be read as one: `gpu_validate` runs
the CPU path as well, and a steady-state deck has one slice, so there is almost
nothing for the device to do. It is the small-problem case in the plainest
possible form — see the performance notes for the same line on a real run.

### 2. Timing and a real comparison

`sase_cpu.in` and `sase_gpu.in` are the same 500-slice SASE run — 10 m of
ARAMIS at `ngrid = 256`, diagnostics every step — differing only in the `gpu`
flag and the rootname. The shot-noise seed is fixed in both.

```sh
export FI_PROVIDER=tcp
mpirun -n 8 ../../build-metal/genesis4 sase_cpu.in
mpirun -n 8 ../../build-metal/genesis4 sase_gpu.in
python3 compare.py sase_cpu.out.h5 sase_gpu.out.h5
```

**Run both at the same number of ranks.** Genesis pads the slice count up to a
multiple of the rank count and distributes the slices accordingly, so the
shot-noise realization depends on the rank count: two *CPU* runs of this deck
at 1 and at 8 ranks differ by 7e-2 in `Field/power`. Comparing across rank
counts measures that, not the GPU. `compare.py` will say so if the array shapes
end up mismatched, but if they happen to match it cannot tell.

`compare.py` needs `h5py` and `numpy`, which the build environment above does
not have. Use whichever environment you normally analyse Genesis output in, or
add them:

```sh
conda install -n genesis4-dev -c conda-forge h5py numpy
```

### 3. The full validation matrix

`sweep.py` runs sixty-two decks against the CPU path on the same inputs and
prints one line per case. Fifty-three of them compare per-step differences
under `gpu_validate`, ten of those being decks the backend must refuse; the
remaining nine run to completion and compare the output against a bound derived
from the per-step figure times the step count. A case that falls back to the CPU
where it should not is a failure, because a fallback gives the right answer by
the slow route and is exactly how an unported element would hide.

```sh
cd examples/metal-gpu
python3 sweep.py --workdir /tmp/g4sweep
```

It needs the same `h5py` and `numpy` as `compare.py`, and takes some minutes.
Pass `--tier step` or `--tier run` for one tier only, and `--only <regex>` to
select cases by name. This is the check to run after changing anything in the
backend.

`tools/fftcheck.mm` is a smaller and more direct check of the FFT kernels alone.
It extracts the shader source from `src/Core/MetalEngine.mm`, so it always tests
the code Genesis actually runs, compiles it for one grid shape, and compares a
row, a column and a complete four-pass solve against a direct DFT in double
precision. It is not part of the build, since it is a diagnostic rather than a
regression test, and it is useful when the sweep says the field is wrong but not
where:

```sh
clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
    -O2 tools/fftcheck.mm -o /tmp/fftcheck
/tmp/fftcheck 256 16 16 8 16      # ngrid lanes regs rowsPerTG colsPerTG
```

Run it from the top of the source tree, since it reads the shader out of
`src/Core/MetalEngine.mm` by relative path. The five shapes the backend itself
uses are `64 8 8 16 16`, `128 8 16 16 16`, `256 16 16 8 16`, `512 16 32 4 8` and
`1024 32 32 2 4`, taken from `pickFFTShape` in that file. All five report a row
and column error of a few times `1e-6` against a scale of order ten, and a round
trip through the complete solve accurate to `4e-07`.

## What agreement to expect

The reference numbers below are from an M3 Max, both runs at 8 ranks. Nothing
here should be much different on another Apple Silicon machine, since the
arithmetic is the same.

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

Three things make raw dataset-by-dataset comparison misleading, and
`compare.py` separates them out for that reason.

**Amplitudes** are the honest number, relative to their own peak. A few times
`1e-4` on a SASE run is the expected level: SASE amplifies exponentially, so a
`1e-7` difference in the first metre is not a rounding error at the end of the
undulator, it is a rounding error multiplied by the gain. The steady-state
`validate.in` figure of `2e-4` over 1104 steps is the same effect.

**Centroids** — `xposition`, `pointing` — average to zero for a symmetric beam,
so their size is set by round-off in the first place and a relative error
against them means nothing. `Beam/xposition` on this deck is `2e-11` m against
a beam 25 µm across. `compare.py` prints those as an absolute difference beside
the scale of the quantity itself.

**Phases** are defined modulo 2π and are undefined where there is no amplitude
to carry them. The near-field phase in particular is the argument of a single
on-axis cell, which passes through zero at optical vortices while the slice as
a whole is still bright. `compare.py` wraps them and masks out the points where
the companion intensity is below a thousandth of its peak; what is left agrees
to about `1e-2` radians.

One more thing about the on-axis quantities, since a GPU run is where it shows.
`intensity-nearfield` and `phase-nearfield` are the field in the cell at the
centre of the grid, and that cell used to be picked as `(ngrid*ngrid-1)/2` —
which is the centre only for an odd `ngrid`. For an even one it lands at the
edge of the grid, where the field is orders of magnitude smaller and a
percent-level difference between two runs means nothing. Since this backend
accepts only powers of two, every GPU run hit it. The index is now
`(ngrid/2)*ngrid + ngrid/2`, unchanged for odd grids, and the end-to-end
agreement on the validation cases improved by an order of magnitude with it.

Finally, the CPU is not a fixed target to compare against. `FieldSolverFFT`
plans its transforms with `FFTW_MEASURE`, so two runs of the *same* CPU binary
on the same machine differ by `1e-15` to `1e-9` depending on which plan the
planner happened to pick. If you are chasing a discrepancy, run the same binary
twice first and use that as the control.

Wakefields were checked the same way, on the 500-slice deck with a copper
resistive wall wake at a 2.5 mm radius, CPU and GPU both at 4 ranks. The number
that matters is not the CPU-to-GPU difference on its own but its size relative
to the effect being modelled, so the third column repeats the comparison against
a run with the `&wake` block removed:

| | GPU vs CPU | wake vs no wake |
|---|---:|---:|
| `Beam/wakefield` | 0 | 1.0 |
| `Beam/energy` | 4.0e-10 | 4.0e-05 |
| `Beam/bunching` | 3.5e-04 | 8.5e-03 |
| `Field/power` | 1.7e-04 | 2.8e-03 |

`Beam/wakefield` is identical because it is the host-side loss profile, written
straight out by both paths. Everywhere else the wake changes the answer by one
to two orders of magnitude more than the two processors differ, and the
CPU-to-GPU column is unchanged from the no-wake case, so adding the wake has not
cost any accuracy.

## Performance notes

End-to-end wall clock of `sase_cpu.in` and `sase_gpu.in` — 500 slices,
`ngrid = 256`, 196 steps, diagnostics at every step — on an M3 Max
(12 performance + 4 efficiency cores, 40-core GPU, 128 GB):

| ranks | `sase_cpu.in` | `sase_gpu.in` |
|---:|---:|---:|
| 1 | 309.1 s | 5.2 s |
| 2 | 158.5 s | 4.8 s |
| 4 | 82.1 s | 4.6 s |
| 12 | 35.2 s | 4.4 s |

About a second of each of those is setup, loading and the output file, on both
paths. The tracking loop itself is 4.1 s on the GPU and about 34 s on twelve
CPU ranks, so **the GPU is worth roughly 8x all twelve cores, and 75x one of
them** — and it leaves the cores free while it works.

**Measure with a clock, not with `clock()`.** Until recently Genesis' own
`Total Wall Clock Time` was processor time, not wall clock, which is the same
thing for a CPU run that never waits and badly wrong for a GPU run that does:
this deck reported 1.2 s while actually taking 5.2 s, because the waiting costs
no CPU and the shader compile happens in another process. That line is now
genuine wall clock. Earlier versions of this chapter quoted the old figures and
so overstated the GPU by about a factor of four; the numbers above supersede
them.

**More MPI ranks against the one GPU do nothing.** The tracking loop takes 4.1 s
at one rank and 4.1 s at twelve. The report at the end of the run says why: at
one rank the device is already 92% busy, so there is nothing to overlap and the
extra ranks only divide up work they then queue for. Use one rank for a GPU run
unless the deck needs the memory of several nodes; the earlier advice to use
four came from the mismeasurement above.

**Where the 4.1 s goes**, for anyone looking to make it smaller: 37% is the
per-slice diagnostics (`output_step = 100` takes the loop from 4.04 s to
2.53 s), 6% is the source deposition, and the rest is the four FFT passes, the
Runge-Kutta push and the transverse map. The field solve runs at close to the
memory bandwidth of the machine, so the diagnostics are where the remaining
headroom is, and `output_step` is the knob that costs nothing to turn.

**A small deck is a different problem, and used to be a bad one.** The 500-slice
example does 19 ms of real work per step, so what it costs to hand work to the
device does not matter. A steady-state deck is one slice, a few tens of
microseconds of arithmetic, and there the handover was everything: the engine
now encodes a whole step into one command buffer and submits once, where it used
to submit four times and wait four times. On the 1104-step `validate.in` geometry
without `gpu_validate`:

| | before | after |
|---|---:|---:|
| tracking loop | 3.91 s | 1.08 s |
| device busy | 2.62 s | 0.54 s |

The same deck takes 3.75 s on one CPU core, so a steady-state run went from
marginally slower than the CPU to three times faster. That the device busy time
fell by as much as the wall clock is the point: most of what Metal was
attributing to the device was the cost of starting work, not of doing it. The
remaining floor is about 1 ms per step, half of it device time for the fifteen
or so dispatches a step encodes, so the next thing to try for small decks is
fewer dispatches rather than faster ones.

**One fallback step costs more than it looks.** A step the GPU refuses is not
merely a step taken at CPU speed: the particles and the field have to come back
to the host before it and go out again afterwards, so the step is slower than
it would have been on the CPU alone. Correctors used to be refused, and since
`orbiterror = true` is implemented as a corrector at every step, an orbit-error
deck spent 187 of its 196 steps that way, which cost about two thirds of the
benefit of the GPU on that deck. No lattice element falls back any more. The
machinery is still there, and a step that does fall back is still counted and
reported at the end of the run, but nothing currently triggers it. Everything
the backend cannot do is a hard error instead.

**`export FI_PROVIDER=tcp` is not optional** if you are using conda's MPICH on
macOS. The default libfabric provider busy-polls while waiting, and the faster
the compute gets the more of the machine that wastes:

| | without | with |
|---|---:|---:|
| `validate.in`, serial, no `mpirun` | 8.9 s | 6.3 s |
| `sase_gpu.in`, 8 ranks | 18.9 s | 5.0 s |

The second row is the one that matters: eight ranks waiting on each other and
on one GPU spend nearly four times the wall clock spinning. The first has no
MPI communication in it at all — one rank, launched directly — and still loses
40%. Put the export in your shell profile. The cost varies with the MPICH build;
an earlier one lost a factor of four even on the serial row.

**Startup is not free, but it is not the problem either.** The Metal shaders are
compiled from source when the first `&track` block starts, in another process,
and it does not scale with the problem. On this deck the whole fixed cost of a
GPU run — loading, shader compile, output file — is about 1 s, against about
0.9 s for the CPU path, whose own fixed cost is mostly FFTW planning its
transforms with `FFTW_MEASURE`. Neither is worth optimising unless the tracking
loop is shorter than either.

**`output_step` is the one free knob.** The per-slice diagnostics are reduced on
the GPU, by one threadgroup per slice, and they are still 37% of the tracking
loop at `output_step = 1`: 4.04 s against 2.53 s at `output_step = 100` on this
deck. Most of that is the far-field branch, which transforms every slice again.
An earlier version of this chapter said the difference was 1.57 s against
1.45 s; that pair came from the timer that did not count waiting.

## Where the code lives, and adding another backend

Six files, of which one is the backend:

| | |
|---|---|
| `include/GPUEngine.h` | the interface the tracking loop talks to, and the design constraints any backend has to respect |
| `src/Core/GPUEngine.cpp` | which backend to build and use; one branch per backend |
| `include/MetalEngine.h`, `src/Core/MetalEngine.mm` | the Apple Silicon backend: the Metal shaders and the host code that drives them |
| `include/SliceMoments.h` | the per-slice diagnostic moments, in the normalisation `DiagBeam` and `DiagField` expect |
| `src/Core/Gencore.cpp` | the tracking loop, which contains no device-specific code and no preprocessor conditionals |

A second backend — CUDA, HIP, SYCL — is a new implementation of `GPUEngine`
plus one branch in `GPUEngine::create()`, and nothing in the tracking loop
changes. The interface is deliberately coarse: whole steps and whole
reductions, not individual kernels, because the cost that dominates is not
arithmetic but synchronisation.

Four things about the design are not Apple specific and would have to be
reproduced rather than reconsidered.

**The beam and the field stay resident for the whole `&track` block.** This is
the reason the interface looks the way it does. Marshalling the host arrays in
and out every step costs 38 ms per step on this problem against about 11 ms of
compute, so a solver plugged in behind `FieldSolver::advance`, which would have
to copy on every call, cannot pay off however fast its kernels are. The beam
and the field are coupled in both directions every step — the deposition reads
the particles, the Runge-Kutta push gathers the field — so neither can move to
the device without the other.

**The diagnostics have to be reduced on the device too.** They are not a small
share of the arithmetic once the tracking is fast: on this deck they are 38% of
the tracking loop with `output_step = 1`. Leaving them on the host would also
break residency, because they read every particle and every grid point.

**Unsupported physics is refused by name, never worked around.** A run that
completes and writes a plausible output file having quietly dropped an effect
the deck asked for is the worst failure mode available here, and it is what the
hard errors and `sweep.py`'s treatment of an unexpected CPU fallback exist to
prevent.

**Single precision has to be arranged for, not merely accepted.** Metal has no
`double` at all, so this is not optional on Apple hardware; a backend with FP64
would not need `gammaRef()`, but it costs nothing to keep. Two reformulations
in `MetalEngine.mm` are the ones to copy: gamma carried as an offset from a
reference energy, and the longitudinal momentum formed as `gamma*sqrt(1-r)`
rather than `sqrt(gamma^2-1-aw^2-p^2)`, since at gamma = 11357 the FP32 quantum
of gamma squared is about 15 and the subtraction loses the whole transverse
contribution.

The comments at the head of `MetalEngine.mm` list the mistakes that were made
once already: the parameter structs are written twice, once in the shader
string and once in host C++, so editing one alone skews the layout silently;
dispatch geometry has to be parameterised at every dispatch site, since
hard-coding it at one site only breaks at grid sizes other than the one being
developed against; and the order of operations within a step has to mirror
`Beam::track` exactly, down to the corrector kick landing before the
longitudinal momentum factor is formed and only on the closing half step.
