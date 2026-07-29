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
*** Error: gpu = true in &track, but collective effects / wakefields is not
    implemented on the GPU yet
```

## What is supported

The GPU runs the deposition, the field propagation, the transverse tracking,
the longitudinal push and all of the per-slice diagnostics. Around that:

| | |
|---|---|
| `ngrid` | a power of two from **64 to 1024** |
| field harmonics | up to 4, all sharing the same `ngrid`, `dgrid` and `gridmax` |
| bunching harmonics | up to 8; beyond that the diagnostics fall back to the CPU |
| `one4one` | not supported; the backend wants a rectangular particle array |
| chicanes, correctors | that one step falls back to the CPU and is reported once |
| ISR, wakefields, short-range space charge | hard error |

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
| 64 | 8 x 8 | 8, 8 | 6.60 s |
| 128 | 16 x 8 | 16, 8 | 6.75 s |
| 256 | 16 x 16 | 16, 16 | 7.45 s |
| 512 | 32 x 16 | 32, 16 | 11.0 s |
| 1024 | 32 x 32 | 32, 32 | 23.5 s |

Agreement with the CPU degrades gently with grid size, because the transform is
single precision and a larger grid means more rounding steps in each butterfly
chain. Over the 1104 steps of the validation deck the largest relative field
difference is 9.2e-6 at `ngrid = 64`, 2.3e-4 at 256 and 8.5e-3 at 1024; the
beam moments stay at 3.2e-6 throughout.

Note that the field solver on the GPU is always the FFT one. `fft_fieldsolver`
in `&track` only selects the CPU solver, so it makes no difference to a
`gpu = true` run, but it does need to be set on both sides of any comparison.

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

## Performance notes

Wall clock for the tracking loop, as Genesis reports it, on an M1 Max
(8 performance + 2 efficiency cores, 32-core GPU, 64 GB):

| ranks | `sase_cpu.in` | `sase_gpu.in` |
|---:|---:|---:|
| 1 | 415.7 s | 7.45 s |
| 2 | — | 4.24 s |
| 4 | 98.5 s | 2.58 s |
| 8 | 54.2 s | 1.94 s |

So the single GPU at one rank is about 56x one core and about 7x all eight,
and about 28x all eight once the ranks are used on both sides.

**Run several MPI ranks against the one GPU.** It is worth another 2.9x at
four ranks and 3.8x at eight, even though there is only one GPU and the ranks
are queueing on it. The reason is that a good part of a step is still host
work — assembling the diagnostics into the output buffers, the global sums,
the HDF5 writes — and one rank's host work overlaps another rank's GPU work.
There is no advantage in going past the number of performance cores.

**`export FI_PROVIDER=tcp` is not optional** if you are using conda's MPICH on
macOS. The default libfabric provider busy-polls while waiting, and the faster
the compute gets the more of the machine that wastes:

| | without | with |
|---|---:|---:|
| `validate.in`, serial, no `mpirun` | 22.1 s | 5.08 s |
| `sase_gpu.in`, 8 ranks | 40.9 s | 1.94 s |

The first row has no MPI communication in it at all — one rank, launched
directly — and still loses a factor of four. Put the export in your shell
profile.

**Startup is not free.** The Metal shaders are compiled from source when the
first `&track` block starts, which is about five seconds. It does not scale
with the problem, but it is why the 1.94 s tracking run above takes about 8 s
of wall clock end to end. Longer runs amortise it; it is only worth thinking
about if you are timing something small.

**`output_step` matters much less than it used to.** The per-slice diagnostics
are computed on the GPU now, by one threadgroup per slice, so `output_step = 1`
costs very little. On the CPU path it was around three quarters of the runtime
for this deck.
