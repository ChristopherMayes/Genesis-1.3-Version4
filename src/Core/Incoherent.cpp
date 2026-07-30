#include "Incoherent.h"
#include "Beam.h"

Incoherent::Incoherent(){
  sran=NULL;
  sranAlt=NULL;
  doLoss=false;
  doSpread=false;
}

Incoherent::~Incoherent(){}

void Incoherent::init(int base, int rank, bool doLoss_in,bool doSpread_in)
{

  doLoss=doLoss_in;
  doSpread=doSpread_in;


  RandomU rseed(base);
  double val;
  for (int i=0; i<=rank;i++){
    val=rseed.getElement();
  }
  val*=1e9;
  int locseed=static_cast<int> (round(val));
  if (sran !=NULL) { delete sran; }
  sran  = new RandomU (locseed);
  // Same seed, so that computeKick() reproduces what apply() would have drawn.
  // See the comment on computeKick() in the header.
  if (sranAlt !=NULL) { delete sranAlt; }
  sranAlt = new RandomU (locseed);
  return;
}


// The two scalars of one integration step. Kept in one place because apply()
// and computeKick() must agree on them exactly.
bool Incoherent::stepScalars(Undulator *und, double delz, double &dgamavg, double &dgamsig) const
{
  if (!und->inUndulator()) { return false; }
  if ((!doLoss) && (!doSpread)) { return false; }

  double gam0=und->getGammaRef();
  double awz=und->getaw();
  double xkw0=und->getku();


  dgamsig=1.015e-27* xkw0 * xkw0 * awz * awz;

  if (und->isHelical()){
    dgamsig*= 1.42 *awz + 1./(1.+1.5*awz+0.95*awz*awz);
  } else {
    dgamsig*= 1.697*awz + 1./(1.+1.88*awz+0.8*awz*awz);
  }

  if (!doSpread){ dgamsig=0;}

  dgamsig=sqrt(dgamsig*gam0*gam0*gam0*gam0*xkw0*delz)*sqrt(3.);


  dgamavg=xkw0*gam0*awz;
  if(!doLoss) { dgamavg=0;}

  dgamavg=dgamavg*dgamavg*1.88e-15*delz;

  return true;
}


bool Incoherent::computeKick(Beam *beam, Undulator *und, double delz, std::vector<double> &dg)
{
  double dgamavg, dgamsig;
  if (!this->stepScalars(und, delz, dgamavg, dgamsig)) { return false; }

  int nbins=beam->nbins;
  if (beam->one4one){ nbins=1;}

  // One draw per beamlet, slice by slice, which is the order apply() consumes
  // them in.
  //
  // With the spread switched off the draw is scaled by zero, so every beamlet
  // gets the same -dgamavg and the generator cannot influence any result, this
  // step or later. It is skipped in that case, which matters: the draws are the
  // expensive part of this, at about 15 ns each and one per beamlet per step.
  dg.clear();
  for (int islice=0;islice< beam->beam.size();islice++){
    int npart=beam->beam.at(islice).size();
    int nbeamlet=(npart+nbins-1)/nbins;
    if (dgamsig == 0){
      dg.insert(dg.end(), nbeamlet, -dgamavg);
      continue;
    }
    for (int ib=0; ib<nbeamlet; ib++){
      dg.push_back(-dgamavg+dgamsig*(2*sranAlt->getElement()-1));
    }
  }
  return true;
}


void Incoherent::apply(Beam *beam, Undulator *und, double delz)
{
  double dgamavg, dgamsig;
  if (!this->stepScalars(und, delz, dgamavg, dgamsig)) { return; }

  // apply energy change to electorn bunch
  int nbins=beam->nbins;
  if (beam->one4one){ nbins=1;}
  double dg=0;

  for (int islice=0;islice< beam->beam.size();islice++){
    int npart=beam->beam.at(islice).size();
    for (int ip=0; ip<npart; ip++){
      if ((ip % nbins) == 0){
         dg=-dgamavg+dgamsig*(2*sran->getElement()-1);
      }
      beam->beam.at(islice).at(ip).gamma+=dg;
    }
  }


  return;

}
