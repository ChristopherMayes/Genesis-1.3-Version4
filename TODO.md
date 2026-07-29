# TODO

## 1. The two upstream pull requests

Both are open **as drafts** against `svenreiche/Genesis-1.3-Version4`, ready for you to edit and
then mark ready for review. Take #272 out of draft first — #273's body references it, because
the three-FFT branch it touches is unreachable without the filter fix.

| PR | branch | commit | title |
|---|---|---|---|
| [#272](https://github.com/svenreiche/Genesis-1.3-Version4/pull/272) | `fix/source-filter-noop` | `4fe13fe` | Fix `source_filter` being silently ignored in `FieldSolverFFT` |
| [#273](https://github.com/svenreiche/Genesis-1.3-Version4/pull/273) | `perf/fft-two-pass` | `e101b30` | `FieldSolverFFT`: drop the source-term FFT when the source filter is off |

The bodies as submitted are `PR-1-source-filter-fix.md` and `PR-2-fft-two-pass.md`. Edit them on
GitHub, not here — these files are only the record of what was sent.

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
- [x] **GPU diagnostic reductions** — the diagnostics were 76% of the GPU runtime
- [x] **Full residency** — no per-step marshalling; the host only sees the arrays at dumps,
      at the slippage boundary slice, and at the end of `&track`

**Measured (500 slices, zstop=10, ngrid=256, 1 harmonic, output_step=1, M1 Max):**

| config | wall | vs 1 CPU core | vs 8 CPU cores |
|---|---|---|---|
| CPU 1 rank | 379.5 s | 1× | — |
| CPU 8 ranks | 61.1 s | 6.2× | 1× |
| GPU 1 rank | **7.47 s** | **50.8×** | **8.2×** |
| GPU 4 ranks (one GPU) | **2.58 s** | **147×** | **23.7×** |

Running several MPI ranks against the single GPU is worth another 2.9×: the host-side
diagnostic assembly and HDF5 buffering overlap with GPU work.

Agreement with a rank-matched CPU reference: `Field/power` 1.6e-04, `Field/xsize` 4.8e-05,
`Beam/bunching` 2.7e-04, `Beam/energyspread` 3.7e-06 — the FP32 level throughout.

### Still to do

- [x] Generalise `ngrid` beyond 256. The shader is now specialised per grid size through
      preprocessor macros and supports every power of two from 64 to 1024; anything else is a
      hard error naming the nearest supported size.
- [ ] Port `Incoherent`, `Collective`/wakefields and short-range space charge (hard error today)
- [ ] Chicanes and correctors fall back to the CPU for that step — port `applyR56`
- [ ] `one4one` is refused outright (the GPU wants a rectangular particle array)
- [ ] The remaining ~5 s of single-rank host time is diagnostic assembly in `Diagnostic.cpp`
      (the `storeValue` loops and the global sums), not GPU work

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
