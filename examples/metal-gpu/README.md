# GPU acceleration on Apple Silicon

Three decks that check the Metal GPU backend on a Mac and measure what it is
worth. The full documentation is in [manual/GPU.md](../../manual/GPU.md); this
is the short version.

The binary has to be built with the backend compiled in:

```sh
cmake -S ../.. -B ../../build-metal \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DENABLE_METAL=ON
cmake --build ../../build-metal -j8
```

## 1. Does it work?

```sh
export FI_PROVIDER=tcp
../../build-metal/genesis4 validate.in
```

A steady-state run with the shot noise off, which tracks the GPU and the CPU
side by side and compares them every step. Takes a few seconds. The last line
is the answer:

```
Metal vs CPU over 1104 steps: max relative error, field 0.00022943, beam 3.25662e-06
```

That is the FP32 round-off level accumulated over the whole undulator. Anything
above about `1e-3` means the backend is broken on your machine rather than
merely less precise.

## 2. What is it worth?

`sase_cpu.in` and `sase_gpu.in` are the same 500-slice SASE run, 10 m of
ARAMIS at `ngrid = 256`, with the same shot-noise seed. They differ only in the
`gpu` flag.

```sh
export FI_PROVIDER=tcp
mpirun -n 8 ../../build-metal/genesis4 sase_cpu.in
mpirun -n 8 ../../build-metal/genesis4 sase_gpu.in
python3 compare.py sase_cpu.out.h5 sase_gpu.out.h5
```

**Use the same number of ranks for both.** The shot-noise realization depends
on how the slices are spread over the ranks, so a 1-rank and an 8-rank CPU run
of this deck already differ by 7e-2 in `Field/power`; comparing across rank
counts measures that and not the GPU.

Tracking time on an M1 Max (8 performance cores, 32-core GPU):

| ranks | `sase_cpu.in` | `sase_gpu.in` |
|---:|---:|---:|
| 1 | 415.7 s | 7.45 s |
| 4 | 98.5 s | 2.58 s |
| 8 | 54.2 s | 1.94 s |

Several MPI ranks sharing the one GPU is worth almost 4x, because the host-side
diagnostic and HDF5 work of one rank overlaps the GPU work of another.

`export FI_PROVIDER=tcp` is not optional. Conda's MPICH busy-polls by default,
which costs a factor of 21 on the 8-rank GPU run (40.9 s instead of 1.94 s) and
still a factor of four on a serial run with no MPI communication in it at all.

`compare.py` needs `h5py` and `numpy`, which the conda environment used to
*build* Genesis will not have unless you add them
(`conda install -c conda-forge h5py numpy`). Any environment that can read the
output files will do. It reports amplitudes, centroids and phases separately,
because a plain relative error is misleading for the last two — see
[manual/GPU.md](../../manual/GPU.md#what-agreement-to-expect).

## 3. Is it right for *my* deck?

`validate.in` checks one configuration. `sweep.py` checks about fifty, which is
what you want before trusting the backend with something you care about:

```sh
python3 sweep.py --mpirun mpirun --workdir /tmp/g4sweep
```

It writes every deck itself, so it needs nothing but the binary and an
interpreter with `h5py`. `--only` takes a regular expression if you want a
subset, and `--tier step` or `--tier run` selects one of the two kinds of check.

The **step** tier runs `gpu_validate = true`, so the CPU and the GPU take the
same step from the same state and the difference is reported before it can
grow. This isolates the error of a single step, and it is the tier that will
name the physics you have got wrong. It cannot see a residency bug, because it
copies the GPU state back to the host every step and so silently repairs
anything that should have been transferred and was not.

The **run** tier therefore runs the whole undulator twice, once on each device,
and compares the output files. An end-to-end difference cannot be judged
against a fixed tolerance, because it depends on how many steps the deck takes
and on how much round-off each one contributes. On a 10 m ARAMIS deck the
on-axis intensity difference grows smoothly from 5.5e-08 at the entrance to
2.9e-02 at the exit, five orders of magnitude, with no step anywhere along the
way, and raising `npart` sixteenfold does not move it.

Nor can it be judged against a small perturbation of the input. Scaling the
seed power by one part in 1e7 moves the answer by only one part in 1e6, because
a uniform scale is very nearly an eigenmode of the amplifier and so is barely
amplified — a tempting control that measures nothing.

Each case is therefore run a third time under `gpu_validate`, which reports the
largest single-step difference and the number of steps. Their product is what
would result if every step's round-off added coherently, which is the worst
that round-off alone can do, and the end-to-end difference has to stay under
it. A missed transfer is a finite error rather than a round-off one, so it goes
straight through the bound.

Every case is checked against what it is supposed to do, so the runs that are
*expected* to fail count as passes: the ten `refuse_*` cases must each die with
the error message that names the keyword at fault. Every other case must stay
on the GPU for the whole run, and a fallback to the CPU is a failure, since it
would otherwise hide an unported element behind a correct answer.
