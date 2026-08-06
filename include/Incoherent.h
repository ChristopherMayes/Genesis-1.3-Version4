#ifndef __GENESIS_INCOHERENT__
#define __GENESIS_INCOHERENT__

#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <math.h>

#include "Undulator.h"
#include "Particle.h"
#include "Sequence.h"
#include "RandomU.h"

class Beam;

using namespace std;

extern const double vacimp;
extern const double eev;



class Incoherent{
 public:
   Incoherent();
   virtual ~Incoherent();
   void init(int, int,bool,bool);
   void apply(Beam *,Undulator *und, double );

   // Computes the energy kick of one integration step into 'dg', one value per
   // beamlet in the same order that apply() would consume them, without
   // touching the particles. This is for backends which hold the particles
   // somewhere other than the host arrays. Returns false if there is nothing to
   // apply, either because the step is outside an undulator or because neither
   // effect is switched on, in which case 'dg' is left alone.
   //
   // It draws from its own generator rather than the one apply() uses. The two
   // are seeded identically and consume in the same order, so they produce the
   // same numbers, and a run which exercises both paths -- gpu_validate does
   // exactly that -- gets the same kick on each rather than consuming one
   // stream twice and diverging.
   bool computeKick(Beam *, Undulator *und, double, std::vector<double> &dg);

   [[nodiscard]] bool isEnabled() const { return doLoss || doSpread; }

 private:
   // The scalars of one step: the mean loss and the half width of the flat
   // distribution the spread is drawn from. Returns false if there is nothing
   // to apply.
   bool stepScalars(Undulator *und, double delz, double &dgamavg, double &dgamsig) const;

   bool doLoss,doSpread;
   RandomU *sran;
   RandomU *sranAlt;
};

#endif
