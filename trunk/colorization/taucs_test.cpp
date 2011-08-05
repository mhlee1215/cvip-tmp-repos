// The code from Alejandro with some slight modifications

#include <iostream>
#include <vector>
#include "Taucs_matrix.h"

typedef Taucs_matrix<taucs_double> TaucsMatrix;

using namespace std;

extern "C" {
#include <taucs.h>
}

int main4()
{
  vector<double> an(10);
  vector<int> jn(10);
  vector<int> ia(10);
  vector<double> f(10); // right-hand size vector object

  // create CCS matrix structure using vector class
  //value
  an[0] = 1.0;
  an[1] = 0.5;
  an[2] = 1.0;
  an[3] = 0.5;
  an[4] = 1.0;
  an[5] = 0.5;
  an[6] = 1.0;

  //row index
  jn[0] = 0;
  jn[1] = 1;
  jn[2] = 1;
  jn[3] = 2;
  jn[4] = 2;
  jn[5] = 3;
  jn[6] = 3;

  //col ptr
  ia[0] = 0;
  ia[1] = 2;
  ia[2] = 4;
  ia[3] = 6;

  ia[4] = 7;

  // create right-hand size vector object
  f[0] = 1.0;
  f[1] = 2.0;
  f[2] = 3.0;
  f[3] = 4.0;

  // resize vectors.
  an.resize(7);
  jn.resize(7);
  ia.resize(5);
  f.resize(4);
	int dim = 4;

  // create TAUCS matrix from vector objects an, jn and ia
  taucs_ccs_matrix  A; // a matrix to solve Ax=b in CCS format
  A.n = dim;
  A.m = dim;
  A.flags = (TAUCS_DOUBLE);
  A.colptr = &ia[0];
  A.rowind = &jn[0];
  A.values.d = &an[0];


  TaucsMatrix* Lmatrix1 = new TaucsMatrix(dim,dim,false);

  Lmatrix1->set_coef(0, 0, 1);
  Lmatrix1->set_coef(1, 0, .5);
  Lmatrix1->set_coef(1, 1, 1);
  Lmatrix1->set_coef(2, 1, .5);
  Lmatrix1->set_coef(2, 2, 1);
  Lmatrix1->set_coef(3, 2, .5);
  Lmatrix1->set_coef(3, 3, 1);
  


  taucs_ccs_matrix*  B = (taucs_ccs_matrix*)Lmatrix1->get_taucs_matrix(dim);


  // create TAUCS right-hand size
  taucs_double* b = &f[0]; // right hand side vector to solve Ax=b

  // allocate TAUCS solution vector
	vector<double> xv(dim);
  taucs_double* x = &xv[0]; // the unknown vector to solve Ax=b

  // solve the linear system
  void* F = NULL;
  char* options[] = {"taucs.factor.LU=true", NULL};
  void* opt_arg[] = { NULL };

	taucs_logfile("stdout");
  int i = taucs_linsolve(B, &F, 1, x, b, options, opt_arg);

  if (i != TAUCS_SUCCESS)
  {
	cout << "Solution error." << endl;
	if (i==TAUCS_ERROR)
	  cout << "Generic error." << endl;

	if (i==TAUCS_ERROR_NOMEM)
	  cout << "NOMEM error." << endl;

	if (i==TAUCS_ERROR_BADARGS)
	  cout << "BADARGS error." << endl;

	if (i==TAUCS_ERROR_MAXDEPTH)
	  cout << "MAXDEPTH error." << endl;

	if (i==TAUCS_ERROR_INDEFINITE)
	  cout << "NOT POSITIVE DEFINITE error." << endl;
  }
  else
  {
	cout << "Solution success." << endl;

	for (unsigned j = 0; j < f.size(); j++)
	  cout << x[j] << endl;
  }

  // deallocate the factorization
  taucs_linsolve(NULL, &F, 0, NULL, NULL, NULL, NULL);

  for(int i = 0 ; i < dim ; i++)
	  printf("%f\n", xv[i]);

  string outname = "test.mtx";
  char* str = (char*)malloc(sizeof(char)*20);
  sprintf(str, "test.mtx");
  taucs_ccs_write_ijv(B, str);

  int k;
  cin >> k;
  return 0;
}
