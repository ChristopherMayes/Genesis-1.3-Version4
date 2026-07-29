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
