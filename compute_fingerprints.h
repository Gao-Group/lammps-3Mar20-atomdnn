
#ifdef COMPUTE_CLASS

ComputeStyle(fingerprints,ComputeFingerprints)

#else

#ifndef COMPUTE_FINGERPRINTS_H
#define COMPUTE_FINGERPRINTS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeFingerprints : public Compute {
 public: 
  ComputeFingerprints(class LAMMPS *, int, char **);
  ~ComputeFingerprints();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  void compute_local();
  double memory_usage();

 private:
  double cutsq;
  double *eta_G2;
  double *zeta;
  double *eta_G4;
  int *lambda;
  int n_etaG2;
  int n_etaG4;
  int n_zeta;
  int n_lambda;
  int g2_flag;
  int g4_flag;

  class NeighList *list;
  double **alocal;
  double **fingerprints;
  int nmax_atom;
  int nmax_local;

  int compute_derivatives(int);
  void reallocate(int);
};

}

#endif
#endif

// compute ID group-ID fingerprints Rc keyword values ... end
// Keyword options: etaG2, etaG4, zeta, lambda

