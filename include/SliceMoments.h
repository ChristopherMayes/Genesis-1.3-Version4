#ifndef __GENESIS_SLICEMOMENTS__
#define __GENESIS_SLICEMOMENTS__

#include <vector>

// Per-slice moments of the beam and of the field, in the same normalisation
// that DiagBeam::getValues and DiagField::getValues produce them.
//
// These structs exist so that the reduction can be done somewhere other than
// the diagnostic classes -- on the GPU, where the particles and the grid
// already live -- without the diagnostic code having to know about it.
//
// A note on precision. The reductions themselves run in single precision, so
// they accumulate the CENTRED second moments; forming <x^2> - <x>^2 from raw
// single-precision sums loses essentially all significance. The centring is
// undone in double precision by whoever fills these structs, so that the
// consumer sees ordinary raw moments and every expression downstream --
// emittance, Twiss, the current-weighted global sums -- is unchanged.

struct BeamSliceMoments {
    int nslice {0};
    int nharm {0};
    bool hasAux {false};

    std::vector<double> x1, x2, y1, y2;
    std::vector<double> px1, px2, py1, py2;
    std::vector<double> g1, g2;
    std::vector<double> xpx, ypy;

    // bunching factor, slice major: index is*nharm + iharm
    std::vector<double> bre, bim;

    // only filled when hasAux
    std::vector<double> xmin, xmax, ymin, ymax;
    std::vector<double> pxmin, pxmax, pymin, pymax;
    std::vector<double> gmin, gmax;
};

struct FieldSliceMoments {
    int nslice {0};
    bool hasFar {false};

    // intensity weighted and unnormalised, in grid cells relative to the axis:
    // power is sum |E|^2, x1 is sum dx |E|^2, x2 is sum dx^2 |E|^2
    std::vector<double> power, x1, x2, y1, y2;

    // coherent sum of the complex field over the slice
    std::vector<double> ffre, ffim;

    // on-axis cell
    std::vector<double> midre, midim;

    // the same for the transform of the slice, only when hasFar
    std::vector<double> fpower, fx1, fx2, fy1, fy2;
};

#endif
