#ifndef __GENESIS_EFIELDSOLVER__
#define __GENESIS_EFIELDSOLVER__

#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <cmath>
#include <mpi.h>

#include "Particle.h"

class Beam;

using namespace std;

extern const double vacimp;
extern const double eev;



// What a backend needs in order to run the short-range solve itself, for the
// case where the particles are not in the host arrays. Everything here is
// either a per-step scalar or one number per slice; the per-particle work is
// the caller's.
struct SCPlan {
    int nz {0}, nphi {0}, ngrid {0};
    double coef {0};            // -gammaz^2/ks^2, the scaling of the Laplacian
    std::vector<double> dr;     // radial grid spacing, per slice
};

class EFieldSolver {
public:
    EFieldSolver();
    virtual ~EFieldSolver();
    void init(double, int, int, int, double, bool);
    void shortRange(vector<Particle> *, double, double, int);
    void longRange(Beam *beam, double gamma, double aw);
    double getEField(unsigned long i);
    bool hasShortRange() const;
    [[nodiscard]] int scGridSize() const { return ngrid; }
    void allocateForOutput(unsigned long nslice);
    double getSCField(int);

    // Advances the shared radial grid over a set of slices and reports the
    // resulting spacing for each, without needing the particles: 'rbound' is
    // the largest particle radius of each slice, which is all analyseBeam()
    // uses them for.
    //
    // The growth is sequential and has to stay that way. rmax only ever
    // increases, and it does so as the slices are visited in order, so slice k
    // is solved on a grid that already accounts for slices 0 to k-1. A backend
    // which sized each slice independently would agree with this one only until
    // the first slice wider than the grid.
    bool planShortRange(const std::vector<double> &rbound, double gz2, SCPlan &plan);

    // Writes the value getSCField() would report, for a backend that computed
    // the solve elsewhere. One per slice, in units of the electron rest mass.
    void setSCField(int islice, double v) { efield[islice] = v; }

private:
    void analyseBeam(vector<Particle> *beam);
    void constructLaplaceOperator();
    void tridiag();

    vector<double> work1, work2, fcurrent, fsize;  // used for long range calculation
    vector<complex<double> > cwork;
    vector<int> idxr;
    vector<double> lmid, rlog, vol, ldig;
    vector<complex<double> > csrc, clow, cmid, cupp, celm, gam; // used for tridiag routine
    vector<double> ez,efield;

    int nz, nphi, ngrid, rank;
    double rmax, ks, xcen, ycen, dr;
    bool longrange;

};

inline double EFieldSolver::getSCField(int islice) {
    return efield[islice]*eev;  // convert from Lorent mass unit to eV /m
}

inline bool EFieldSolver::hasShortRange() const{
    return (nz>0) & (ngrid > 2);
}

#endif
