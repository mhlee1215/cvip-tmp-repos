#include "Taucs_fix.h"
#include "Taucs_matrix.h"
#include "Taucs_vector.h"
#include <stdio.h>
#include <cmath>


typedef Taucs_matrix<taucs_double> TaucsMatrix;
typedef Taucs_vector<taucs_double> TaucsVector;
typedef Taucs_symmetric_matrix<taucs_double> TaucsSymmetricMatrix;


#define mu 5 //local window size for calculating mean and variance
void getMeanVariance(int width, int height, int x, int y, float *input, float &mean, float &variance){
	int xMin = x-mu/2;
	int xMax = x+mu/2;
	int yMin = y-mu/2;
	int yMax = y+mu/2;
	int k,z;

	if(xMin < 0)	xMin = 0;
	if(yMin < 0)	yMin = 0;

	if(xMax >= width)	xMax = width-1;
	if(yMax >= height)  yMax = height-1;

	float sum=0.0f;
	int numOfElement = 0;

	for(k=xMin; k <= xMax ; k++)
	{
		for(z=yMin; z<= yMax; z++)
		{
			sum+=input[width*z+k];
			numOfElement++;
		}
	}

	mean = sum/(float)numOfElement;

	sum=0.0f;
	for(k=xMin; k <= xMax ; k++)
	{
		for(z=yMin; z<= yMax; z++)
		{
			float deviation = input[width*z+k]-mean;
			sum+=deviation*deviation;
		}
	}

	variance = sum/(float)numOfElement;
}

void colorization(int width,int height,float *input[3],float *markedImg[3],bool *isMarked,float *output[3])
{
	int x,y,index;
	float lamda = 1.0f;
	int MatrixDim = width*height;
	printf("MatrixDim: %d\n",MatrixDim);


	TaucsMatrix* Lmatrix = new TaucsMatrix(MatrixDim,MatrixDim,false);
	double *BUVvector = new double[2*MatrixDim];
	
	for(y=0, index = 0; y<height; y++)
	{


		int yMin = y-1;
		int yMax = y+1;
		if(yMin < 0)	yMin = 0;
		if(yMax >= height)  yMax = height-1;

		for(x=0; x<width; x++,index++)
		{
			float mean, var;
			getMeanVariance(width, height, x, y, input[0], mean, var);		
			if(var<0.000002f)	var=0.000002f;

			int xMin = x-1;
			int xMax = x+1;
			
			if(xMin < 0)	xMin = 0;
			if(xMax >= width)	xMax = width-1;
			
			float sum = 0;
			for(int k=xMin; k <= xMax ; k++)
			{
				for(int z=yMin; z<= yMax; z++)
				{
					if((x!=k || z!=y))
					{
						float weight = 0.05f+(input[0][index]-mean)*(input[0][width*z+k]-mean)/var ;
					//	float weight = exp((-1.0)/(2*var)*(input[0][index]-input[0][width*z+k])*(input[0][index]-input[0][width*z+k]));
						if(weight<0)	weight = -weight;
				
						sum+=weight;
						Lmatrix->set_coef(index, width*z+k, -weight);
					}
				}
			}
			if(isMarked[index])
			{
				Lmatrix->set_coef(index,index,sum+lamda);
				BUVvector[index]=lamda*markedImg[1][index];

				BUVvector[MatrixDim+index]=lamda*markedImg[2][index];		
			}else{
				Lmatrix->set_coef(index,index,sum);
				BUVvector[index]=0;
				BUVvector[MatrixDim+index]=0;	
			}
		}
	}

	char* solve [] = {"taucs.factor.LU=true", NULL};
	int error_code;    
	double* tempResultUV = new double[2*MatrixDim];
	memset(tempResultUV, 0, 2*MatrixDim*sizeof(double));

	taucs_ccs_matrix* tm = (taucs_ccs_matrix*)Lmatrix->get_taucs_matrix(MatrixDim);


  char* str = (char*)malloc(sizeof(char)*20);
  sprintf(str, "test.mtx");
  taucs_ccs_write_ijv(tm, str);

	
	error_code = taucs_linsolve( tm, NULL, 2, tempResultUV, BUVvector, solve, NULL);

	
	if(error_code != TAUCS_SUCCESS){
		printf("Solver Failed\n");
		getchar();
	}

	for(index =0; index<MatrixDim; index++)
	{
		output[0][index] = input[0][index];
		output[1][index] = (float)tempResultUV[index];
		output[2][index] = (float)tempResultUV[MatrixDim+index];
	}

	delete Lmatrix;
	delete [] BUVvector;
	delete [] tempResultUV;
}