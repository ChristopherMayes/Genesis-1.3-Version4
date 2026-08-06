#ifndef __GENESIS_FIELDSOLVER__
#define __GENESIS_FIELDSOLVER__

#include <vector>
#include <iostream>
#include <string>
#include <complex>


class Field;
class Beam;

#include "Particle.h"
#include "Undulator.h"


using namespace std;


class FieldSolver{
 public:
    virtual ~FieldSolver() {};
    virtual void init(double,double,double,unsigned int) = 0;
    virtual void advance(double, Field *, Beam *, Undulator *) = 0;
    virtual void initSourceFilter(double,double,double,bool) = 0;

    // Reports the source filter this solver was given, for a backend which has
    // to reproduce it. Returns false if the solver does not filter, in which
    // case the parameters are untouched. The values are the ones the solver
    // settled on rather than the ones the deck asked for, since an unphysical
    // width or centre disables the filter rather than being used.
    virtual bool getSourceFilter(double &xc, double &yc, double &sig) const { return false; }
};


#endif
