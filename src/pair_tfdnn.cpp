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

#include "pair_tfdnn.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <numeric>
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
#include <string.h>
#include <stddef.h>
#include <ctype.h>


using namespace std;
using namespace LAMMPS_NS;

#define MAXLINE 1000
#define MAXARGS 1000
#define MAXWORDS 100



/*---------------------------------------------------------------------------------------------
read the information of descriptor parameters and nerual network
---------------------------------------------------------------------------------------------*/
void PairTFDNN::read_file(char *filename)
{
  char line[MAXLINE];
  char *ptr;
  char *args[MAXWORDS];
  char keyword[MAXWORDS];
  char parameter[MAXWORDS];
  int eof=0;
  int n, nwords;
  
  n_etaG2 = 0;
  n_etaG4 = 0;
  n_zeta = 0;
  n_lambda = 0;
  g2_flag = 0;
  g4_flag = 0;

  MPI_Comm_rank(world,&me);
  
  FILE *fp = NULL;

  char *para_file = new char(strlen(filename) + strlen("/parameter"));
  strcpy(para_file, filename);
  strcat(para_file,"/parameter");

  // read potential file on root proc 0

  if (me == 0){
    fp = force->open_potential(para_file);
    if (fp == NULL){
      char str[128];
      snprintf(str, 128,"Cannot open TensorflowDNN potential file %s", filename);
      error->one(FLERR,str);
    }
  }

  while (1){

    // read line in root proc 0 and then broadcast the line
    if (me == 0){
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL){  
	eof = 1;
	fclose(fp);
      }
      else
	n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    if (!strncmp(line,"tf_model_dir",12)){
      sscanf(line,"%s %s",keyword,parameter);
      tf_model_dir = new char[strlen(filename)+strlen("/")+strlen(parameter)];
      strcpy(tf_model_dir, fileneme);
      strcat(tf_model_dir,"/");
      strcat(tf_model_dir,parameter);
    }
    else if (!strncmp(line,"element",7)){
      //sscanf(line,"%s %d",keyword,&tf_nelement);
      tf_nelement = getwords(line,args,MAXARGS)-1;
      tf_element = new char*[tf_nelement];
      for (int i=0;i<tf_nelement;i++){
	tf_element[i] = new char[strlen(args[i+1])];
	strcpy(tf_element[i],args[i+1]);
      }
    }

    else if (!strncmp(line,"input",5)){     
      sscanf(line,"%s %d",keyword,&tf_input_number);
      tf_input_tag = new char*[tf_input_number];
      tf_input_tensor = new char* [tf_input_number];
      for (int i=0;i<tf_input_number;i++){
	
	// read line on root proc 0 and then broadcast the line
	if (me == 0){
	  utils::sfgets(FLERR,line,MAXLINE,fp,filename,error);
	  n = strlen(line) + 1;
	}
	MPI_Bcast(&n,1,MPI_INT,0,world);
	MPI_Bcast(line,n,MPI_CHAR,0,world);

	// extract tf input tensor tag and names
	sscanf(line,"%s %*s",parameter);
	tf_input_tag[i] = new char[strlen(parameter)];
	strcpy(tf_input_tag[i],parameter);
	sscanf(line,"%*s %s",parameter);
	tf_input_tensor[i] = new char[strlen(parameter)];
	strcpy(tf_input_tensor[i],parameter);	
      }
    }

    else if (!strncmp(line,"output",6)){     
      sscanf(line,"%s %d",keyword,&tf_output_number);
      tf_output_tag = new char*[tf_output_number];
      tf_output_tensor = new char*[tf_output_number];
      for (int i=0;i<tf_output_number;i++){

	// read line on root proc 0 and then broadcast the line
	if (me == 0){
	  utils::sfgets(FLERR,line,MAXLINE,fp,filename,error);
	  n = strlen(line) + 1;
	}
	MPI_Bcast(&n,1,MPI_INT,0,world);
	MPI_Bcast(line,n,MPI_CHAR,0,world);

	// extract tf output tensor tag and names
	sscanf(line,"%s %*s",parameter);
	tf_output_tag[i] = new char[strlen(parameter)];
	strcpy(tf_output_tag[i],parameter);
	sscanf(line,"%*s %s",parameter);
	tf_output_tensor[i] = new char[strlen(parameter)];
	strcpy(tf_output_tensor[i],parameter);
      } 
    }

    else if (!strncmp(line,"cutoff",6))
      sscanf(line,"%s %lf",keyword,&cut_global);
    
    else if (!strncmp(line,"descriptor",10)){
      sscanf(line,"%s %s %d",keyword,parameter,&n_parameter);
      descriptor = new char[strlen(parameter)];
      strcpy(descriptor,parameter);
      if (!strncmp(descriptor,"acsf",4)){
	for (int ip=0;ip<n_parameter;ip++){

	  // read line on root proc 0 and then broadcast the line
	  if (me == 0){
	    utils::sfgets(FLERR,line,MAXLINE,fp,filename,error);
	    n = strlen(line) + 1;
	  }
	  MPI_Bcast(&n,1,MPI_INT,0,world);
	  MPI_Bcast(line,n,MPI_CHAR,0,world);

	  // extract descriptor parameters
	  if (!strncmp(line,"etaG2",5)){
	    n_etaG2 = getwords(line,args,MAXARGS) -1;
	    eta_G2 = new double[n_etaG2];
	    for (int i=0;i<n_etaG2;i++)
	      eta_G2[i] = strtod(args[i+1],NULL);
	  }
	  else if (!strncmp(line,"etaG4",5)){
	    n_etaG4 = getwords(line,args,MAXARGS) -1;
	    eta_G4 = new double[n_etaG4];
	    for (int i=0;i<n_etaG4;i++)
	      eta_G4[i] = strtod(args[i+1],NULL);
	  }
	  else if (!strncmp(line,"zeta",4)){
	    n_zeta = getwords(line,args,MAXARGS) -1;
	    zeta = new double[n_zeta];
	    for (int i=0;i<n_zeta;i++)
	      zeta[i] = strtod(args[i+1],NULL);
	  }
	  else if (!strncmp(line,"lambda",6)){
	    n_lambda = getwords(line,args,MAXARGS) -1;
	    lambda = new double[n_lambda];
	    for (int i=0;i<n_lambda;i++)
	      lambda[i] = strtod(args[i+1],NULL);
	  }
	}
      } 
    } // end of reading descriptor information      
  } // end of while
 
    // check reading error and set flag
  if (!strncmp(descriptor,"acsf",4)){
    if (n_etaG2 == 0)
      error->all(FLERR,"Need to set eta of G2 parameters of acsf");
    else
      g2_flag = 1;
    if (n_etaG4 == 0)
      error->all(FLERR,"Need to set eta of G4 parameters of acsf");
    else if (n_zeta == 0)
      error->all(FLERR,"Need to set zeta of G4 parameters of acsf");
    else if (n_lambda == 0)
      error->all(FLERR,"Need to set lambda of G4 parameters of acsf");
    else
      g4_flag = 1;
  }    
}

/* ---------------------------------------------------------------------- */

PairTFDNN::PairTFDNN(LAMMPS *lmp) : Pair(lmp), fingerprints(NULL)
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

PairTFDNN ::~PairTFDNN()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
     delete [] map;
  }
  
  memory->destroy(fingerprints);
  memory->destroy(eta_G2);
  memory->destroy(eta_G4);
  memory->destroy(zeta);
  memory->destroy(lambda);
  memory->destroy(tf_model_dir);
  memory->destroy(tf_input_tensor);
  memory->destroy(tf_output_tensor);
  memory->destroy(tf_element);
  memory->destroy(tf_output_tag);

  
  TF_DeleteGraph(Graph);
  TF_DeleteSession(Session, Status);
  TF_DeleteSessionOptions(SessionOpts);
  TF_DeleteStatus(Status);
}

/* ---------------------------------------------------------------------- */

void PairTFDNN::coeff(int narg, char **arg)
{
  int n = atom->ntypes;
  
  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
  
  read_file(arg[2]);

  int element_count = 1; // count the unique atom type
  int element_check = 0; // check if the element is set correctly 
  map[0] = -1; // first element of map is not used
  for (int i=0;i<n;i++){
    for (int j=0;j<tf_nelement;j++){
      if (strcasecmp(arg[3+i],tf_element[j]) == 0){
	map[i+1] = j+1;
	element_check=1;
      }
    }
    if (element_check==1)
      element_check=0;
    else
      error->all(FLERR,"Incorrect args for pair coefficients");
    if (i>1 & map[i]!=map[i+1])
      element_count++;
  }

  if (element_count!=tf_nelement)
    error->all(FLERR,"Incorrect args for pair coefficients");
  

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}
/* ---------------------------------------------------------------------- */
void PairTFDNN::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag,n+1,n+1,"pair:setflag");
  cutsq = memory->create(cutsq,n+1,n+1,"pair:cutsq");

  map = new int[n+1];
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

void PairTFDNN::settings(int narg, char ** /* arg */)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTFDNN::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style tfdnn requires atom IDs");
  
  if (force->newton_pair != 1)
    error->all(FLERR,"Pair style tfdnn requires newton pair on");


  // Initialise neighbor list, including neighbors of ghosts
  int irequest_full = neighbor->request(this);
  neighbor->requests[irequest_full]->id = 1;
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;
  neighbor->requests[irequest_full]->ghost = 1;

  
  // create tensorflow model
  create_tensorflow_model();
  
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTFDNN::init_one(int /*i*/, int /*j*/)
{
  return cut_global;
}


/*-----------------------------------------------------------------------*/

void PairTFDNN::NoOpDeallocator(void* data, size_t a, void* b) {}


void PairTFDNN::create_tensorflow_model()
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


/*-----------------------------------------------------------------------*/

void PairTFDNN::compute_fingerprints()
{
  const int inum = list->inum;
  const int* const ilist = list->ilist;
  const int* const numneigh = list->numneigh;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;
  int const ntypes = atom->ntypes;
  double** const x = atom->x;
  const int* const mask = atom->mask;
  double pi = 3.14159265358979323846;
  double cutoffsq = cut_global*cut_global;
  int ntypes_combinations = ntypes*(ntypes+1)/2;
  int n_fingerprints = n_etaG2*ntypes*g2_flag + n_lambda*n_zeta*n_etaG4*ntypes_combinations*g4_flag + ntypes;
  size_peratom_cols = n_fingerprints;
  size_rows = atom->nlocal;
  
  // Initialize fingerprnts vector per atom
  double fingerprints_atom[size_peratom_cols];
  for (int i=0;i<size_peratom_cols;i++)
    fingerprints_atom[i] = 0;

  // allocate fingerprints array, size of the array may change at different timestep
  if (size_rows > 0) {
    memory->destroy(fingerprints);
    fingerprints = (float *)malloc(size_rows * size_peratom_cols * sizeof(float));
    // memory->create(fingerpts,size_rows,size_peratom_cols,"tfdnn:fingerpts");
  }

  //
  int position[ntypes][ntypes];
  int pos = 0;
  for (int pos_1 = 0; pos_1 < ntypes; pos_1++)  {
    for (int pos_2 = pos_1; pos_2 < ntypes; pos_2++)  {
      position[pos_1][pos_2] = pos;
      position[pos_2][pos_1] = pos;
      pos++;  
    }
  }

  int j, jnum, jtype, type_comb, k;
  double Rx_ij, Ry_ij, Rz_ij, rsq, Rx_ik, Ry_ik, Rz_ik, rsq1;
  double Rx_jk, Ry_jk, Rz_jk, rsq2, cos_theta, aux, G4;
  double function, function1, function2;
  
  // The fingerprints are calculated for each atom i in the initial data
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (mask[i]) {

      // First neighborlist for atom i
      const int* const jlist = firstneigh[i];
      jnum = numneigh[i];

      /* ------------------------------------------------------------------------------------------------------------------------------- */
      for (int jj = 0; jj < jnum; jj++) {            // Loop for the first neighbor j
        j = jlist[jj];
        j &= NEIGHMASK;

        // Element type of atom j. Rij calculation.
        Rx_ij = x[j][0] - x[i][0];
        Ry_ij = x[j][1] - x[i][1];
        Rz_ij = x[j][2] - x[i][2];
        rsq = Rx_ij*Rx_ij + Ry_ij*Ry_ij + Rz_ij*Rz_ij;
        jtype = type[j];

        // Cutoff function Fc(Rij) calculation
        if (rsq < cutoffsq && rsq>1e-20) {    
          function = 0.5*(cos(sqrt(rsq/cutoffsq)*pi)+1);

          // G1 fingerprints calculation: sum Fc(Rij)
          // fingerprints_atom[(jtype-1)*(1+n_etaG2)] += function;
	  fingerprints_atom[jtype-1] += function;
	  
          if (g2_flag == 1) {
            // The number of G2 fingerprints depend on the number of given eta_G2 parameters
            for (int m = 0; m < n_etaG2; m++)  {
              //fingerprints_atom[1+m+(jtype-1)*(1+n_etaG2)] += exp(-eta_G2[m]*rsq)*function; // G2 fingerprints calculation
	      fingerprints_atom[ntypes+m*ntypes+jtype-1] += exp(-eta_G2[m]*rsq)*function;                                   // G2 fingerprints calculation
            }
          }

          /* ------------------------------------------------------------------------------------------------------------------------------- */
          if (g4_flag == 1) {
            for (int kk = 0; kk < jnum; kk++) {            // Loop for the second neighbor k
              k = jlist[kk];
              k &= NEIGHMASK;

              // Rik (rsq1) and Rjk (rsq2) calculation. G2 fingerprints and derivatives are only calculated if Rik<Rc and Rjk<Rc
              Rx_ik = x[k][0] - x[i][0];
              Ry_ik = x[k][1] - x[i][1];
              Rz_ik = x[k][2] - x[i][2];
              rsq1 = Rx_ik*Rx_ik + Ry_ik*Ry_ik + Rz_ik*Rz_ik;
              Rx_jk = x[k][0] - x[j][0];
              Ry_jk = x[k][1] - x[j][1];
              Rz_jk = x[k][2] - x[j][2];
              rsq2 = Rx_jk*Rx_jk + Ry_jk*Ry_jk + Rz_jk*Rz_jk;
              cos_theta = (rsq+rsq1-rsq2)/(2*sqrt(rsq*rsq1));        // cos(theta)
              type_comb = position[jtype-1][type[k]-1];

              if (rsq1 < cutoffsq && rsq1>1e-20 && rsq2 < cutoffsq && rsq2>1e-20) {
                function1 = 0.5*(cos(sqrt(rsq1/cutoffsq)*pi)+1);        // fc(Rik)
                function2 = 0.5*(cos(sqrt(rsq2/cutoffsq)*pi)+1);        // fc(Rjk)

                // The number of G4 fingerprints depend on the number of given parameters
                for (int h = 0; h < n_lambda; h++)  {
                  aux = 1+(lambda[h]*cos_theta);
		  if (aux > 0){
		    for (int l = 0; l < n_zeta; l++)  {
		      for (int q = 0; q < n_etaG4; q++) {
			G4 = pow(2,1-zeta[l])*pow(aux,zeta[l])*exp(-eta_G4[q]*(rsq+rsq1+rsq2))*function*function1*function2;
			if (kk > jj)   fingerprints_atom[ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))] += G4;
		      }
		    }
                  }
                }
              }          
            }
          }
        }
      }
      // Writing The fingerprnts_atom vector in the fingerprints matrix
      for(int n = 0; n < size_peratom_cols; n++) {
	fingerprints[i*size_peratom_cols+n]=fingerprints_atom[n];
        fingerprints_atom[n] = 0.0;
      }
    } 
  }
  
}

/*-------------------------------------------------------------*/
void PairTFDNN::compute_derivatives()
{
  const int inum = list->inum;
  const int* const numneigh = list->numneigh;

  // Get initial atoms data and neighborlists
  const int* const ilist = list->ilist;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;
  int const ntypes = atom->ntypes;
  double** const x = atom->x;
  const int* const mask = atom->mask;
  int * const tag = atom->tag;
  double pi = 3.14159265358979323846;
  int count = 0;
  double cutoffsq = cut_global*cut_global;

  int ntypes_combinations = ntypes*(ntypes+1)/2;
  int n_derivatives = n_etaG2*ntypes*g2_flag + n_lambda*n_zeta*n_etaG4*ntypes_combinations*g4_flag + ntypes;
  size_local_cols = 1 + n_derivatives;

  int total_list = accumulate(numneigh, numneigh+inum, 0);
  size_local_rows = 4*(inum+total_list);

 
  // allocate fingerprints derivative array, size of the array may change at different timestep
  if (size_local_rows > 0) {
    memory->destroy(fingerprints_der);
    fingerprints_der = (float *)malloc(size_local_rows * size_local_cols * sizeof(float));
  }  
  
  
  // define derivatives (w.r.t other atoms) and derivatives_i (w.r.t itself) array
  double derivatives[4][size_local_cols];
  double derivatives_i[4][size_local_cols];


  
  for(int n = 0; n < size_local_cols; n++) {
    derivatives_i[0][n] = 0.0;
    derivatives_i[1][n] = 0.0;
    derivatives_i[2][n] = 0.0;
    derivatives_i[3][n] = 0.0;
  }

  int position[ntypes][ntypes];
  int pos = 0;
  for (int pos_1 = 0; pos_1 < ntypes; pos_1++)  {
    for (int pos_2 = pos_1; pos_2 < ntypes; pos_2++)  {
      position[pos_1][pos_2] = pos;
      position[pos_2][pos_1] = pos;
      pos++;  
    }
  }

  int j, jnum, jtype, type_comb, k;
  double Rx_ij, Ry_ij, Rz_ij, rsq, Rx_ik, Ry_ik, Rz_ik, rsq1;
  double Rx_jk, Ry_jk, Rz_jk, rsq2, cos_theta, aux, G4;
  double function, function1, function2, dfc, dfc1, dfc2;
  double ij_factor, ik_factor, jk_factor;

  // The derivatives are calculated for each atom i in the initial data
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (mask[i]) {

      // First neighborlist for atom i
      const int* const jlist = firstneigh[i];
      jnum = numneigh[i];

      derivatives_i[0][0] = tag[i];
      derivatives_i[0][1] = type[i];
      derivatives_i[0][2] = tag[i];
      derivatives_i[0][3] = type[i];
      derivatives_i[0][4] = tag[i];
      derivatives_i[1][0] = x[i][0];
      derivatives_i[2][0] = x[i][1];
      derivatives_i[3][0] = x[i][2];

      for(int n = 1; n < size_local_cols; n++) {
	derivatives_i[1][n] = 0.0;
	derivatives_i[2][n] = 0.0;
	derivatives_i[3][n] = 0.0;
      }

      for(int n = 0; n < size_local_cols; n++) {
	derivatives[0][n] = 0.0;
	derivatives[1][n] = 0.0;
	derivatives[2][n] = 0.0;
	derivatives[3][n] = 0.0;
      }

      // Loop for the first neighbor j
      for (int jj = 0; jj < jnum; jj++) {            
	j = jlist[jj];
	j &= NEIGHMASK;

	// Element type of atom j. Rij calculation.
	Rx_ij = x[j][0] - x[i][0];
	Ry_ij = x[j][1] - x[i][1];
	Rz_ij = x[j][2] - x[i][2];
	rsq = Rx_ij*Rx_ij + Ry_ij*Ry_ij + Rz_ij*Rz_ij;
	jtype = type[j];

	// Cutoff function Fc(Rij) and dFc(Rij) calculation
	if (rsq < cutoffsq && rsq>1e-20) { 
	  function = 0.5*(cos(sqrt(rsq/cutoffsq)*pi)+1);
	  dfc = -pi*0.5*sin(pi*sqrt(rsq/cutoffsq))/(sqrt(cutoffsq));

	  derivatives[0][0] = tag[i];
	  derivatives[0][1] = type[i];
	  if (j>inum) {derivatives[0][2] = j;}
	  if (j<=inum)  {derivatives[0][2] = tag[j];}
	  derivatives[0][3] = jtype;
	  derivatives[0][4] = tag[j];
	  derivatives[1][0] = x[j][0];
	  derivatives[2][0] = x[j][1];
	  derivatives[3][0] = x[j][2];
	  
	  for(int n = 1; n < size_local_cols; n++) {
	    derivatives[1][n] = 0.0;
	    derivatives[2][n] = 0.0;
	    derivatives[3][n] = 0.0;
	  }

	  derivatives[1][jtype] = dfc*Rx_ij;
	  derivatives[2][jtype] = dfc*Ry_ij;
	  derivatives[3][jtype] = dfc*Rz_ij;
	  
	  derivatives_i[1][jtype] += -dfc*Rx_ij;
	  derivatives_i[2][jtype] += -dfc*Ry_ij;
	  derivatives_i[3][jtype] += -dfc*Rz_ij;
	  
	  if (g2_flag == 1) {
	    // The number of G2 derivatives depend on the number of given eta_G2 parameters
	    for (int m = 0; m < n_etaG2; m++)  {
	      derivatives[1][ntypes+m*ntypes+jtype] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rx_ij; // G2 derivatives in the x direction
	      derivatives[2][ntypes+m*ntypes+jtype] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Ry_ij; // G2 derivatives in the y direction
	      derivatives[3][ntypes+m*ntypes+jtype] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rz_ij; // G2 derivatives in the z direction

	      derivatives_i[1][ntypes+m*ntypes+jtype] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rx_ij; // G2 derivatives (w.r.t itself) in the x direction
	      derivatives_i[2][ntypes+m*ntypes+jtype] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Ry_ij; // G2 derivatives (w.r.t itself) in the y direction
	      derivatives_i[3][ntypes+m*ntypes+jtype] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rz_ij; // G2 derivatives (w.r.t itself) in the z direction 
	    }
	  }

	  /* ------------------------------------------------------------------------------------------------------------------------------- */
	  if (g4_flag == 1) {
	    for (int kk = 0; kk < jnum; kk++) {            // Loop for the second neighbor k
	      k = jlist[kk];
	      k &= NEIGHMASK;

	      // Rik (rsq1) and Rjk (rsq2) calculation. G4 derivatives are only calculated if Rik<Rc and Rjk<Rc
	      Rx_ik = x[k][0] - x[i][0];
	      Ry_ik = x[k][1] - x[i][1];
	      Rz_ik = x[k][2] - x[i][2];
	      rsq1 = Rx_ik*Rx_ik + Ry_ik*Ry_ik + Rz_ik*Rz_ik;
	      Rx_jk = x[k][0] - x[j][0];
	      Ry_jk = x[k][1] - x[j][1];
	      Rz_jk = x[k][2] - x[j][2];
	      rsq2 = Rx_jk*Rx_jk + Ry_jk*Ry_jk + Rz_jk*Rz_jk;
	      cos_theta = (rsq+rsq1-rsq2)/(2*sqrt(rsq*rsq1));               // cos(theta)
	      type_comb = position[jtype-1][type[k]-1];

	      if (cos_theta < -1)  cos_theta = -1;
	      if (cos_theta > 1)   cos_theta = 1;

	      if (rsq1 < cutoffsq && rsq1>1e-20 && rsq2 < cutoffsq && rsq2>1e-20) {
		function1 = 0.5*(cos(sqrt(rsq1/cutoffsq)*pi)+1);               // fc(Rik)
		function2 = 0.5*(cos(sqrt(rsq2/cutoffsq)*pi)+1);               // fc(Rjk)
		dfc2 = -pi*0.5*sin(pi*sqrt(rsq2/cutoffsq))/(sqrt(cutoffsq));      // dFc(Rjk)
		dfc1 = -pi*0.5*sin(pi*sqrt(rsq1/cutoffsq))/(sqrt(cutoffsq));      // dFc(Rik)
		
		// The number of G4 derivatives depend on the number of given parameters
		for (int h = 0; h < n_lambda; h++)  {
		  aux = 1+(lambda[h]*cos_theta);
		  if (aux > 0)  {
		    for (int l = 0; l < n_zeta; l++)  {
		      for (int q = 0; q < n_etaG4; q++) {
			G4 = pow(2,1-zeta[l])*pow(aux,zeta[l])*exp(-eta_G4[q]*(rsq+rsq1+rsq2))*function*function1*function2;

			// Calculation of factors necessary for the derivatives of G4 with respect to atom j
			ij_factor = (1/sqrt(rsq*rsq1)-cos_theta/rsq)*lambda[h]*zeta[l]/aux-2*eta_G4[q]+dfc/(sqrt(rsq)*function);
			ik_factor = (1/sqrt(rsq*rsq1)-cos_theta/rsq1)*lambda[h]*zeta[l]/aux-2*eta_G4[q]+dfc1/(sqrt(rsq1)*function1);
			jk_factor = -(1/sqrt(rsq*rsq1))*lambda[h]*zeta[l]/aux-2*eta_G4[q]+dfc2/(sqrt(rsq2)*function2);

			// G4 derivatives calculation
			if ( kk != jj ) {
			  // G4 derivatives with respect to x, y, z directions
			  derivatives[1][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+1]  += G4*(ij_factor*Rx_ij-jk_factor*Rx_jk);
			  derivatives[2][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+1]  += G4*(ij_factor*Ry_ij-jk_factor*Ry_jk);
			  derivatives[3][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+1]  += G4*(ij_factor*Rz_ij-jk_factor*Rz_jk);
			}
			if ( kk > jj ) {
			  derivatives_i[1][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+1]  += -G4*(ij_factor*Rx_ij+ik_factor*Rx_ik);
			  derivatives_i[2][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+1]  += -G4*(ij_factor*Ry_ij+ik_factor*Ry_ik);
			  derivatives_i[3][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+1]  += -G4*(ij_factor*Rz_ij+ik_factor*Rz_ik);
			}
		      }
		    }
		  }
		}
	      }          
	    } 
	  } // end loop of k

	  // Writing the derivatives array in the array_atom matrix, for derivative of atom i w.r.t. atom j
	  for (int m = 0; m < 3; m++) 
	    for(int n = 0; n < size_local_cols; n++)
	      fingerprints_der[(4*count+m)*size_local_cols+n] = derivatives[m][n];
	  count++;
	} 
      } // loop of j

      // Writing the derivatives_i array in the array_atom matrix, for derivative of atom i w.r.t. atom i
      for (int m = 0; m < 3; m++) 
	    for(int n = 0; n < size_local_cols; n++)
	      fingerprints_der[(4*count+m)*size_local_cols+n] = derivatives[m][n];
      count++;
    } 
  }

  
}



/* ---------------------------------------------------------------------- */

void PairTFDNN::compute(int eflag, int vflag)
{
  compute_fingerprints();

  //compute_derivatives();
  
  dims = new int64_t[ndims]; // allocate dims vector
    
  dims[0] = 1; // the first dimension is 1
  dims[1] = size_rows;
  dims[2] = size_peratom_cols;

  
  // length of the input tensors, in terms of bits
  int ndata = sizeof(float)*dims[1]*dims[2]; 
     
  //create tensorflow input and output tensors for inference   
  InputValues[0] = TF_NewTensor(TF_FLOAT, dims, ndims, fingerprints, ndata, &NoOpDeallocator, 0);
    
  if (InputValues[0] == NULL)
    error->all(FLERR,"Failed TF_NewTensor\n");
  
  // Run the tensorflow Session for prediction
  TF_SessionRun(Session, NULL, Input, InputValues, tf_input_number, Output, OutputValues, tf_output_number, NULL, 0,NULL , Status);
  
  if(TF_GetCode(Status) != TF_OK)
    error->all(FLERR,TF_Message(Status));

  void* buff = TF_TensorData(OutputValues[0]);
  float* model_output = (float*)buff;
  float eng_local;
  eng_local = model_output[0];
  float eng_total = 0;
  
  eng_vdwl = model_output[0];


  /*
  if(eflag_global){ 
    //eng_vdwl = eng_total;
  }

  
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
/*---------------------------------------------------------------*/
int PairTFDNN::getwords(char *line, char *words[], int maxwords)
{
  char *p = line;
  int nwords = 0; 
  while(1){
    while(isspace(*p))
      p++;
    if(*p == '\0')
      return nwords;
    words[nwords++] = p;
    while(!isspace(*p) && *p != '\0')
      p++;
    if(*p == '\0')
      return nwords;
    *p++ = '\0';
    if(nwords >= maxwords)
      return nwords;
  }
}

/*--------------------------------------------------------------*/
int strcasecmp(const char *s1, const char *s2) {
    const unsigned char *us1 = (const u_char *)s1,
                        *us2 = (const u_char *)s2;

    while (tolower(*us1) == tolower(*us2++))
        if (*us1++ == '\0')
            return (0);
    return (tolower(*us1) - tolower(*--us2));
}
