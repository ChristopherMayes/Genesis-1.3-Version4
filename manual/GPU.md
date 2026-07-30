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
| chicanes, correctors | that one step falls back to the CPU and is reported once |
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

| `ngrid` | REGS x LANES | radices used | time for the SASE example |
|---:|---:|---|---:|
| 64 | 8 x 8 | 8, 8 | 0.78 s |
| 128 | 16 x 8 | 16, 8 | 0.95 s |
| 256 | 16 x 16 | 16, 16 | 1.53 s |
| 512 | 32 x 16 | 32, 16 | 3.93 s |
| 1024 | 32 x 32 | 32, 32 | 14.8 s |

The cost is close to the `N^2 log N` the transform implies once the grid is big
enough to fill the machine: 64 to 128 barely moves because a 64-point grid does
not saturate the GPU, and from 256 upwards each doubling costs about 2.6x, then
3.8x. Earlier versions of this table were flat at around 7 s across the whole
range because a host-side diagnostic bug dominated every configuration; see the
performance notes below.

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
[the performance notes](#performance-notes). It is worth a factor of four even
on this single-rank run.

This is a steady-state run of the ARAMIS undulator with the shot noise off and
a seeded field, so it is deterministic, and it runs both paths and compares
them every step. The last line is the point of it:

```
Metal vs CPU over 1104 steps: max relative error, field 0.00022943, beam 3.25662e-06
```

Those two numbers are the FP32 round-off level accumulated over 1104 steps. If
the run on your machine reports something similar, the backend is working. A
number above about `1e-3` means it is wrong, not merely less precise.

It also prints, at the start,

```
Metal backend: Apple M1 Max, 1 MB resident, gamma_ref = 11357.8
  host transfer check: field 4.5e-08   beam 4.4e-08 (relative, FP32 rounding is ~1e-7)
```

which confirms the device that was picked and that the upload round trip is
clean before any physics happens.

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

## What agreement to expect

The reference numbers below are from an M1 Max, both runs at 8 ranks. Nothing
here should be much different on another Apple Silicon machine, since the
arithmetic is the same.

```
amplitudes, relative                         error       scale
  Field/Global/intensity-farfield       6.605e-04   3.447e+20
  Field/intensity-farfield              6.246e-04   1.768e+21
  Field/ydivergence                     6.054e-04   1.599e-05
  Field/intensity-nearfield             5.354e-04   2.600e+14
  Beam/bunching                         2.202e-04   1.241e-02
  Field/power                           1.728e-04   1.338e+07
  Field/xsize                           4.891e-05   7.737e-05
  worst of 51: 6.605e-04
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

Wall clock for the tracking loop, as Genesis reports it, on an M1 Max
(8 performance + 2 efficiency cores, 32-core GPU, 64 GB):

| ranks | `sase_cpu.in` | `sase_gpu.in` |
|---:|---:|---:|
| 1 | 375.8 s | 1.55 s |
| 2 | — | 1.14 s |
| 4 | 97.8 s | 0.95 s |
| 8 | 53.3 s | 1.02 s |

So the single GPU at one rank is about 240x one core and about 34x all eight,
and the best GPU configuration is about 56x the best CPU one.

**Running several MPI ranks against the one GPU helps, but only up to a point.**
Two ranks are worth 1.4x and four are worth 1.6x; eight are slower than four.
The ranks queue on a single GPU, so once the host is out of the way there is
nothing left to overlap and the extra ranks only add communication and startup.
Four is a good default on this machine. Earlier versions of this document
claimed 2.9x at four ranks and 3.8x at eight, and explained it as host
diagnostic work overlapping GPU work. That explanation was correct about the
mechanism but the host work in question was almost entirely a bug, and with it
fixed the effect largely disappears.

**`export FI_PROVIDER=tcp` is not optional** if you are using conda's MPICH on
macOS. The default libfabric provider busy-polls while waiting, and the faster
the compute gets the more of the machine that wastes:

| | without | with |
|---|---:|---:|
| `validate.in`, serial, no `mpirun` | 21.8 s | 5.02 s |
| `sase_gpu.in`, 8 ranks | 31.2 s | 1.15 s |

The first row has no MPI communication in it at all — one rank, launched
directly — and still loses a factor of four. Put the export in your shell
profile.

**Startup is not free.** The Metal shaders are compiled from source when the
first `&track` block starts. It does not scale with the problem, but now that
the tracking loop itself is around a second it is a visible part of the wall
clock end to end. Longer runs amortise it; it is only worth thinking about if
you are timing something small.

**`output_step` matters much less than it used to.** The per-slice diagnostics
are computed on the GPU, by one threadgroup per slice, so `output_step = 1`
costs very little: 1.57 s against 1.45 s at `output_step = 100` for this deck.
That was not true before the host-side diagnostic assembly was fixed, when the
same pair was 7.33 s and 1.48 s.
