#ifndef __GENESIS_TRACKBEAM__
#define __GENESIS_TRACKBEAM__

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <map>
#include <stdlib.h>


class Beam;
#include "Undulator.h"
#include "Particle.h"


using namespace std;

class TrackBeam{
 public:
   TrackBeam();
   virtual ~TrackBeam();
   void track(double, Beam *, Undulator *, bool);

   void (TrackBeam::*ApplyX) (double, double, double *, double *, double, double);
   void (TrackBeam::*ApplyY) (double, double, double *, double *, double, double);
   void applyDrift(double, double, double *, double *, double, double);
   void applyFQuad(double, double, double *, double *, double, double);
   void applyDQuad(double, double, double *, double *, double, double);
   void applyCorrector(Beam *, double, double);
   void applyChicane(Beam *, double, double, double, double,double);
   void applyR56(Beam *, Undulator *, double);

   // The chicane transfer matrix, including the backwards drift over the total
   // length. Static so that a backend which holds the particles elsewhere can
   // build the same matrix rather than transcribing it.
   static void chicaneMatrix(double angle, double lb, double ld, double lt,
                             double m[][4]);

 private:
   static void matmul(double a[][4], double b[][4]);
};


#endif
