#include <iostream>
#include <vector>
#include "Taucs_matrix.h"

using namespace std;

extern "C" {
#include <taucs.h>
}

typedef Taucs_matrix<taucs_double> TaucsMatrix;

int main2()
{
	int rows = 3;
	int cols = 5;

	TaucsMatrix* Lmatrix = new TaucsMatrix(rows,cols,false);
	double *BUVvector = new double[cols];

	for(int i = 0 ; i < rows ; i++){
		for(int j = 0 ; j < cols ; j++)
			Lmatrix->set_coef(i, j, 1);
	}

	for( int i = 0 ; i < cols ; i++)
		BUVvector[i] = 2;

	taucs_ccs_matrix* tm = (taucs_ccs_matrix*)Lmatrix->get_taucs_matrix(rows);

	char* solve [] = {"taucs.factor.LU=true", NULL};
	int error_code;    
	double* tempResultUV = new double[2*cols];
	memset(tempResultUV, 0, cols*sizeof(double));
	error_code = taucs_linsolve( tm, NULL, 2, tempResultUV, BUVvector, solve, NULL);
	
	if(error_code != TAUCS_SUCCESS){
		printf("Solver Failed\n");
		getchar();
	}

	for(int i = 0 ; i < cols*2 ; i++)
		printf("%f\n", tempResultUV[i]);

	int k;
	cin >> k;
	return 0;
}