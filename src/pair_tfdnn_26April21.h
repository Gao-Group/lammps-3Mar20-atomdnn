/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(tfdnn,PairTFDNN)

#else

#ifndef LMP_PAIR_TFDNN_H
#define LMP_PAIR_TFDNN_H

#include "pair.h"
#include "tensorflow/c/c_api.h"

namespace LAMMPS_NS {

class PairTFDNN : public Pair {
  public:
  PairTFDNN(class LAMMPS *);
  ~PairTFDNN();
  void read_file(char *);
  void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  static void NoOpDeallocator(void* , size_t , void*);
  void create_tensorflow_model();
  void compute_fingerprints();
  int getwords(char*, char**, int);
 

private:
  char *descriptor;           // type of descriptor, e.g. ACSF
  double *eta_G2;             // eta parameters for ACSF G2 descriptor 
  double *zeta;               // zeta parameters for ACSF G4 descriptor 
  double *eta_G4;             // eta parameters for ACSF G4 descriptor 
  double *lambda;             // lambda parameters for ACSF G4 descriptor 
  float max_fp;               // maximum value of fingerprints used for nomalization
  float min_fp;               // minimum value of fingerprints used for nomalization
  int n_parameter;            // number of parameters for descriptor, = 4 for ACSF  
  int n_etaG2;                // number of eta parameters for ACSF G2 descriptor
  int n_etaG4;                // number of eta parameters for ACSF G4 descriptor
  int n_zeta;                 // number of zeta parameters for ACSF G4 descriptor
  int n_lambda;               // number of lambda parameters for ACSF G4 descriptor
  int g2_flag;                // = 1, if ACSF G2 descriptor is set
  int g4_flag;                // = 1, if ACSF G4 descriptor is set

  float *fingerpts;          // fingerprints
  int size_peratom_cols;      // cols of fingerpts array, = numnber of fingerprints
  int size_rows;              // rows of fingerpts array, = number of local atom

  int64_t *dims;               // dims vector define the size of the input tensors, e.g. {1,2,3} means a 1x2x3 tensor
  // float *data; // define vector for fingerprints

  // class NeighList *list;

protected:
  double cut_global;
  int tf_input_number, tf_output_number;
  int ndims; // dimension of input tensor
  int *tf_atom_type;
  char *tf_model_dir;
  char **tf_input_tensor,**tf_output_tensor,**tf_output_tag;
  char *tf_model_tags = "serve";
  TF_Graph *Graph;
  TF_Output *Input,*Output;
  TF_Status *Status;
  TF_SessionOptions *SessionOpts;
  TF_Buffer *RunOpts;
  int ntags;
  TF_Session *Session;
  TF_Tensor **InputValues, **OutputValues;
  
  class Compute *comp_fp;
  char *id_comp_fp;
  int fpflag;
  virtual void allocate();
};
  
}

#endif
#endif
  
