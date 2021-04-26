/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Wei Gao (University of Texas at San Antonio)
                         Daniela Posso (University of Texas at San Antonio)
------------------------------------------------------------------------- */

#include "pair_tensorflow.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "modify.h"
#include "compute.h"

#include "tensorflow/c/c_api.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace LAMMPS_NS;
//using namespace MathConst;
#define INVOKED_PERATOM 8


/* ---------------------------------------------------------------------- */

PairTensorflow::PairTensorflow(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  // one_coeff = 1;
  no_virial_fdotr_compute = 1;
  manybody_flag = 1;


  // create a new compute fingerprints style
  // id = fix-ID + temp
  // compute group = all since pressure is always global (group all)
  //   and thus its KE/temperature contribution should use group all
  
}

/* ---------------------------------------------------------------------- */

PairTensorflow::~PairTensorflow()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    //  delete [] map;
  }
  delete [] tf_model_dir;
  delete [] tf_input_tensor;
  delete [] tf_output_tensor;
  delete [] tf_atom_type;
  delete [] tf_output_tag;
  
  TF_DeleteGraph(Graph);
  TF_DeleteSession(Session, Status);
  TF_DeleteSessionOptions(SessionOpts);
  TF_DeleteStatus(Status);

}

/* ---------------------------------------------------------------------- */

void PairTensorflow::coeff(int narg, char **arg)
{
  int n = atom->ntypes;
  
  if (!allocated) allocate();

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  
  if (strcmp(arg[2],"input") == 0){

    if (narg > n*2+3)
      error->all(FLERR,"Incorrect args for pair coefficients");

    if ((narg-3) % 2 != 0) //odd number
      error->all(FLERR,"Incorrect args for pair coefficients");

    tf_input_number = (narg-3)/2;
    tf_atom_type = new int[tf_input_number];
    tf_input_tensor = new char* [tf_input_number];

    for (int i=0;i<tf_input_number;i++){
      if (atoi(arg[2*i+1]) > n)
	error->all(FLERR,"Incorrect args for pair coefficients");
      tf_atom_type[i] = atoi(arg[2*i+3]);
      tf_input_tensor[i] = new char[strlen(arg[2*i+4])];
      strcpy(tf_input_tensor[i],arg[2*i+4]);
    }
    
  }else if (strcmp(arg[2],"output") == 0){
  
    if (narg > 9)
      error->all(FLERR,"Incorrect args for pair coefficients");
    
    if ((narg-3) % 2 !=0) //odd number
      error->all(FLERR,"Incorrect args for pair coefficients");

    tf_output_number = (narg-3)/2;
    tf_output_tag = new char*[tf_output_number];
    tf_output_tensor = new char*[tf_output_number];

    for (int i=0;i<tf_output_number;i++){
      if (strcmp(arg[2*i+3],"pe") == 0 || strcmp(arg[2*i+3],"force") == 0 || strcmp(arg[2*i+3],"stress") == 0){
	tf_output_tag[i] = new char[strlen(arg[2*i+3])];
	strcpy(tf_output_tag[i],arg[2*i+3]);
	tf_output_tensor[i] = new char[strlen(arg[2*i+4])];
	strcpy(tf_output_tensor[i],arg[2*i+4]);
      }else
	error->all(FLERR,"Incorrect args for pair coefficients");
    }
    
  }else
    error->all(FLERR,"Incorrect args for pair coefficients");


  // set setflag i,j for type pairs where both are assigned by input tensors

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (tf_atom_type[i] >= 0 && tf_atom_type[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }
  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

  
}
/* ---------------------------------------------------------------------- */
void PairTensorflow::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag,n+1,n+1,"pair:setflag");
  cutsq = memory->create(cutsq,n+1,n+1,"pair:cutsq");

  /*
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cutghost,n+1,n+1,"pair:cutghost"); 
  */
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTensorflow::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);
  
  tf_model_dir = new char[strlen(arg[1])];
  strcpy(tf_model_dir,arg[1]);

  // reset cutoffs that have been explicitly set
  /*
  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
	}*/
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTensorflow::init_style()
{
  int me;
  MPI_Comm_rank(world,&me);

  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style tensorflow requires atom IDs");
  
  if (force->newton_pair != 1)
    error->all(FLERR,"Pair style tensorflow requires newton pair on");

  // Initialise neighbor list, including neighbors of ghosts
  int irequest_full = neighbor->request(this);
  neighbor->requests[irequest_full]->id = 1;
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;
  neighbor->requests[irequest_full]->ghost = 1;
  
  // int igroup;
  // int ncompute,find_compute;
  // ncompute = modify->ncompute;
  // find_compute = 0;
  // for (int i=0;i<ncompute;i++){
  //   if (strcmp(modify->compute[i]->style,"fingerprints")==0){
  //     cout << "=====> find the compute\n";
  //     find_compute = 1;
  //     //modify->compute[i]->compute_peratom();
  //     igroup = modify->compute[i]->igroup;
  //     printf("======> igroup = %d\n", igroup);
  //     break;
  //   }
  // }
  // create compute fingerprints
  // id_comp_fp = new char[12];
  // strcpy(id_comp_fp,"comp_fp_temp");

  // char **newarg = new char *[3];
  // newarg[0] = id_comp_fp;
  // newarg[1] = (char *) "all";
  
  // create tensorflow model
  create_tensorflow_model();
  printf ("===========================> create tensorflow model is done on proc %d\n",me);  
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTensorflow::init_one(int /*i*/, int /*j*/)
{
  return cut_global;
}


/*-----------------------------------------------------------------------*/

void PairTensorflow::NoOpDeallocator(void* data, size_t a, void* b) {}


void PairTensorflow::create_tensorflow_model()
{
  Graph = TF_NewGraph();
  Status = TF_NewStatus();
  SessionOpts = TF_NewSessionOptions();
  RunOpts = NULL;
  ntags =1;
  ndims = 3;

  //create a session for the saved tensorflow model
  Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, tf_model_dir, &tf_model_tags, ntags, Graph, NULL, Status);

  if(TF_GetCode(Status) != TF_OK)
    error->all(FLERR,TF_Message(Status));

  // get the tensorflow model input and output tensors
  Input = (TF_Output*)malloc(sizeof(TF_Output) * tf_input_number);
  for (int i=0;i<tf_input_number;i++){
    Input[i] = {TF_GraphOperationByName(Graph, tf_input_tensor[i]), 0};
    if(Input[i].oper == NULL)
      error->all(FLERR,"Failed load tensorflow inputs by names");      
  }

  Output = (TF_Output*)malloc(sizeof(TF_Output) * tf_output_number);
  for (int i=0;i<tf_output_number;i++){
    Output[i] = {TF_GraphOperationByName(Graph, tf_output_tensor[i]), 0};
    if(Output[i].oper == NULL)
      error->all(FLERR,"Failed load tensorflow outputs by names");      
  }

  // allocate input and output tensors for prediction
  InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*tf_input_number);
  OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*tf_output_number);
}

/*----------------------------------------------------------------------*/



/*-----------------------------------------------------------------------*/

void PairTensorflow::compute_fingerprints()
{
  
  // read fingerprints from data file 
  ifstream file("./dump_fingerprints.1");  // data file for fingerprints
  
  dims = new int64_t[ndims]; // allocate dims vector
    
  dims[0] = 1; // the first dimension is 1
  dims[1] = 96;
  dims[2] = 42;

  float max_fingerprint = 52.6217; 
  float min_fingerprint = 0.00618013;
  string line;

  data = (float *)malloc(dims[1] * dims[2]* sizeof(float));
  int i,j,atom_id, atom_type;

  for (i=0;i<9;i++) // skip the first 9 lines
    getline(file,line);
  
  for (i=0;i<dims[1];i++){
    file >> atom_id >> atom_type;
    for (j=0;j<dims[2];j++){
      file >> data[i*dims[2]+j];
      data[i*dims[2]+j]=data[i*dims[2]+j]/(max_fingerprint - min_fingerprint);
    }
  }

  /*
  // read the fingerprints for the first element
  file >> dims_1[1] >> dims_1[2]; 
  data_1 = (float *)malloc(dims_1[1] * dims_1[2]* sizeof(float));  
  for (int i=0;i<dims_1[1];i++)
    for (int j=0;j<dims_1[2];j++)
      file >> data_1[i*dims_1[2]+j];
  
  // read the fingerprints for the second element
  file >> dims_2[1] >> dims_2[2];
  printf("===> read file: %d,%d\n",dims_2[1],dims_2[2]);
  data_2 = (float *)malloc(dims_2[1] * dims_2[2]* sizeof(float));
  for (int i=0;i<dims_2[1];i++)
    for (int j=0;j<dims_2[2];j++){
      file >> data_2[i*dims_2[2]+j];
      //printf("===> read file: %f\n",data_2[i*dims_2[2]+j]);
    }
  */
    
 
  // length of the input tensors, in terms of bits
  int ndata = sizeof(float)*dims[1]*dims[2]; 
 
  //create tensorflow input and output tensors for inference   
  InputValues[0] = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    
  if (InputValues[0] == NULL)
    error->all(FLERR,"Failed TF_NewTensor\n");
  
  //file.close();
}


/* ---------------------------------------------------------------------- */

void PairTensorflow::compute(int eflag, int vflag)
{

  
  int ncompute,find_compute;
  ncompute = modify->ncompute;
  find_compute = 0;
  double **fingerprints;
  int size_peratom_cols;
  int me;

  MPI_Comm_rank(world,&me);
  for (int i=0;i<ncompute;i++){
    if (strcmp(modify->compute[i]->style,"fingerprints")==0){
      cout << "=====> find the compute\n";
      printf ("===============> compute_id = %d on proc %d\n",i,me);
      find_compute = 1;
      modify->compute[i]->compute_peratom();
      fingerprints = modify->compute[i]->array_atom;
      printf ("===============> fingerprints[0][2] = %f\n",fingerprints[0][2]);
      size_peratom_cols = modify->compute[i]->size_peratom_cols;
      break;
    }
  }

  printf ("===============> size_peratom_cols = %d\n",size_peratom_cols);
  //  printf ("===============> fingerprints[0][2] = %d\n",fingerprints[0][2]);
   
  if (find_compute == 0)
    error->all(FLERR,"Pair tensorflow can not find compute fingerprints");

  dims = new int64_t[ndims]; // allocate dims vector
    
  dims[0] = 1; // the first dimension is 1
  dims[1] = atom->nlocal;
  dims[2] = size_peratom_cols;

  float max_fingerprint = 52.6217; 
  float min_fingerprint = 0.00618013;

  data = (float *)malloc(dims[1] * dims[2]* sizeof(float));
  // length of the input tensors, in terms of bits
  int ndata = sizeof(float)*dims[1]*dims[2]; 
  
  for (int i=0;i<dims[1];i++)
    for (int j=0;j<dims[2];j++)
      data[i*dims[2]+j]=fingerprints[i][j]/(max_fingerprint - min_fingerprint);

  printf ("===============> nlocal = %d on proc %d\n",atom->nlocal,me);
  printf ("===============> data[2] = %f on proc %d\n",data[2],me);
  int temp = dims[2]*dims[1];
  printf ("===============> data[last] = %f on proc %d\n",data[temp],me);
  //create tensorflow input and output tensors for inference   
  InputValues[0] = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    
  if (InputValues[0] == NULL)
    error->all(FLERR,"Failed TF_NewTensor\n");
  
    //compute_fingerprints();
  
  // Run the tensorflow Session for prediction
  TF_SessionRun(Session, NULL, Input, InputValues, tf_input_number, Output, OutputValues, tf_output_number, NULL, 0,NULL , Status);
  
  if(TF_GetCode(Status) != TF_OK)
    error->all(FLERR,TF_Message(Status));

  void* buff = TF_TensorData(OutputValues[0]);
  float* model_output = (float*)buff;
  float eng_local;
  float eng_total;
  eng_local = model_output[0];
  
  MPI_Allreduce(&eng_local,&eng_total,1,MPI_FLOAT,MPI_SUM,world);
  
  if(eflag_global){
    eng_vdwl = eng_total;
    printf("=====> eng_vdwl = %f on proc %d\n",eng_vdwl,me);
  }

  printf("=====> model_output = %f\n",model_output[0]);

  /*
  if(eflag_atom) {
    for (ii = 0; ii < ntotal; ii++) {
      eatom[ii] = 0;
    }
  }
  */

  if (vflag_global) {
      virial[0] = 0;
      virial[1] = 0;
      virial[2] = 0;
      virial[3] = 0;
      virial[4] = 0;
      virial[5] = 0;
  }  
}