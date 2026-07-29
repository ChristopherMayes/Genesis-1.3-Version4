#ifndef __GENESIS_CONTROL__
#define __GENESIS_CONTROL__

#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <string>

#include <mpi.h>
#include "Field.h"
#include "Beam.h"
#include "Undulator.h"
#include "HDF5base.h"
#include "Output.h"
#include "Diagnostic.h"

using namespace std;

// Hook that lets an accelerator backend keep its own copy of the field in step
// with the slippage without Control having to know anything about it. Slippage
// rewrites exactly one slice per slip event, so only that slice has to move.
class SliceSync {
  public:
    virtual ~SliceSync() = default;
    virtual void pullSlice(int ifld, int islice) = 0;   // backend -> host
    virtual void pushSlice(int ifld, int islice) = 0;   // host -> backend
};

class Control : public HDF5Base{
 public:
   Control();
   virtual ~Control();
   void applySlippage(double, Field *, int ifld = 0, SliceSync *sync = nullptr);
   bool applyMarker(Beam *, vector<Field *> *, Undulator *, bool&);
   bool init(int, int, const std::string, Beam *, vector<Field *> *, Undulator *,bool,bool,bool);
   // void output(Beam *, vector<Field*> *,Undulator *,Diagnostic &);

 private:
   bool timerun,scanrun,one4one,periodic;
   int nslice,ntotal,noffset;
   int rank, size;
   double sample,reflen,slen;
   int nzout;
   int nwork;
   double *work;
   string root;
};


#endif
