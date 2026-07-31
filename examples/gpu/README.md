# GPU acceleration

Three decks that check the GPU backend and measure what it is worth. Nothing
here is specific to a backend: the same decks and the same sweep drive the
NVIDIA and the Apple Silicon builds, and the binary names its own in the lines
that are compared. The full documentation is in
[manual/GPU.md](../../manual/GPU.md); this is the short version.

The binary has to be built with a backend compiled in:

```sh
cmake -S ../.. -B ../../build-cuda \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DENABLE_CUDA=ON
cmake --build ../../build-cuda -j8
```

Substitute `-DENABLE_METAL=ON` and `build-metal` on a Mac; everything below is
otherwise unchanged.

## 1. Does it work?

```sh
export FI_PROVIDER=tcp
../../build-cuda/genesis4 validate.in
```

A steady-state run with the shot noise off, which tracks the GPU and the CPU
side by side and compares them every step. Takes a few seconds. The last line
is the answer:

```
CUDA  vs CPU over 1104 steps: max relative error, field 0.000212, beam 3.18e-06
Metal vs CPU over 1104 steps: max relative error, field 0.000207, beam 3.18e-06
```

That is the FP32 round-off level accumulated over the whole undulator, and the
two backends land on the same number because they are the same arithmetic.
Anything above about `1e-3` means the backend is broken on your machine rather
than merely less precise.

## 2. What is it worth?

`sase_cpu.in` and `sase_gpu.in` are the same 500-slice SASE run, 10 m of
ARAMIS at `ngrid = 256`, with the same shot-noise seed. They differ only in the
`gpu` flag.

```sh
export FI_PROVIDER=tcp
mpirun -n 8 ../../build-cuda/genesis4 sase_cpu.in
mpirun -n 8 ../../build-cuda/genesis4 sase_gpu.in
python3 compare.py sase_cpu.out.h5 sase_gpu.out.h5
```

**Use the same number of ranks for both.** The shot-noise realization depends
on how the slices are spread over the ranks, so a 1-rank and an 8-rank CPU run
of this deck already differ by 7e-2 in `Field/power`; comparing across rank
counts measures that and not the GPU.

Wall clock end to end, on a Core Ultra 9 285K with 24 cores and an RTX 5080,
and on an M3 Max with 12 performance cores and a 40-core GPU:

| ranks | 285K, CPU | RTX 5080 | M3 Max, CPU | M3 Max GPU |
|---:|---:|---:|---:|---:|
| 1 | 268.0 s | 3.6 s | 309.1 s | 5.2 s |
| 8 | 35.8 s | 2.2 s | | |
| 12 | | | 35.2 s | 4.4 s |
| 16 | 25.6 s | | | |

So the RTX 5080 is worth about 17x all sixteen cores and about 190x one of them,
on the tracking loop alone. **Extra MPI ranks against one card do not help**:
the line printed at the end of the run reports the device already saturated at
one rank, so the ranks only divide up work they then queue for. Several *cards*
do help, and are used by running one rank against each — see
[manual/GPU.md](../../manual/GPU.md#running-on-more-than-one-gpu).

A steady-state deck is worth trying too, since that is where an interactive
parameter scan usually lives. `validate.in` with `gpu_validate` switched off is
one slice over 1104 steps.

`export FI_PROVIDER=tcp` is not optional. Conda's MPICH busy-polls while it
waits, and the faster the compute the more of the machine that wastes: a factor
of four even on a serial run with no MPI communication in it at all.

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
python3 sweep.py --genesis ../../build-cuda/genesis4 --workdir /tmp/g4sweep
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
*expected* to fail count as passes: the `refuse_*` cases must each die with the
error message that names the keyword at fault. Every other case must stay
on the GPU for the whole run, and a fallback to the CPU is a failure, since it
would otherwise hide an unported element behind a correct answer.
