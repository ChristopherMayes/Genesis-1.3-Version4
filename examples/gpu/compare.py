#!/usr/bin/env python3
"""Compare two Genesis 4 output files dataset by dataset.

    python3 compare.py sase_cpu.out.h5 sase_gpu.out.h5

Three kinds of quantity are reported separately, because a single relative
error is misleading for two of them:

  amplitudes    reported relative to their own maximum.

  centroids     transverse offsets and pointing angles average to zero for a
                symmetric beam, so their size is set by round-off. Reported as
                an absolute difference next to the scale of the quantity
                itself, which is the only fair comparison.

  phases        meaningless where there is no amplitude to carry them, and
                defined modulo 2*pi. Reported in radians, wrapped, and only
                over the points where the companion amplitude is above a
                thousandth of its peak.

Needs h5py and numpy.
"""

import sys
import numpy as np
import h5py

TWO_PI = 2.0 * np.pi

# Averages to zero for a symmetric beam, so its scale is set by round-off.
NEAR_ZERO = ("position", "pointing")

# A phase is only meaningful where the companion amplitude is not negligible.
# The near- and far-field phases are the argument of a single cell, so the
# right companion is the on-axis intensity, not the power integrated over the
# whole slice: the on-axis field can pass through zero while the slice is still
# bright, and the phase is undefined there.
PHASE_PARTNER = {
    "Field/phase-nearfield": "Field/intensity-nearfield",
    "Field/phase-farfield": "Field/intensity-farfield",
    "Beam/bunchingphase": "Beam/bunching",
}


def collect(f):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f" and obj.size > 1:
            out[name] = np.asarray(obj[()], dtype=float)

    f.visititems(visit)
    return out


def main(path_a, path_b):
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        a, b = collect(fa), collect(fb)

    amplitude, centroid, phase, skipped = [], [], [], []

    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        if x.shape != y.shape:
            skipped.append(f"{name}: {x.shape} vs {y.shape}")
            continue
        d = np.abs(x - y)
        scale = np.abs(x).max()

        partner = PHASE_PARTNER.get(name)
        if partner is not None and partner in a and a[partner].shape == x.shape:
            d = np.minimum(d % TWO_PI, TWO_PI - d % TWO_PI)
            w = a[partner]
            mask = w > 1e-3 * w.max()
            phase.append((d[mask].max() if mask.any() else 0.0, name, int(mask.sum())))
        elif scale == 0.0 or any(k in name for k in NEAR_ZERO):
            centroid.append((d.max(), name, scale))
        else:
            amplitude.append((d.max() / scale, name, scale))

    nonfinite = [n for n in sorted(set(a) & set(b)) if not np.isfinite(b[n]).all()]

    amplitude.sort(reverse=True)
    print(f"amplitudes, relative                    {'error':>10}  {'scale':>10}")
    for v, name, scale in amplitude[:12]:
        print(f"  {name:<36} {v:10.3e}  {scale:10.3e}")
    print(f"  worst of {len(amplitude)}: {amplitude[0][0]:.3e}")

    print(f"\ncentroids, absolute                     {'error':>10}  {'own scale':>10}")
    for v, name, scale in sorted(centroid, reverse=True)[:8]:
        print(f"  {name:<36} {v:10.3e}  {scale:10.3e}")

    print(f"\nphases, radians                         {'error':>10}  {'points':>10}")
    for v, name, n in sorted(phase, reverse=True):
        print(f"  {name:<36} {v:10.3e}  {n:10d}")

    if skipped:
        print("\nSKIPPED, shape mismatch. Were both runs made with the same")
        print("number of MPI ranks? Genesis pads the slice count to a multiple")
        print("of the rank count.")
        for s in skipped:
            print("  " + s)

    if nonfinite:
        print(f"\nNON-FINITE VALUES in {path_b}: {nonfinite}")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
