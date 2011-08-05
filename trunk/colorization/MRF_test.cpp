#include "mrf.h"
#include "ICM.h"
#include "GCoptimization.h"
#include "MaxProdBP.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int sizeX = 10;
const int sizeY = 10;
const int K = 3;

MRF::CostVal D[sizeX*sizeY*K];
MRF::CostVal V[K*K];
MRF::CostVal hCue[sizeX*sizeY-50];
MRF::CostVal vCue[sizeX*sizeY];

MRF::CostVal smoothMax = 100,lambda=.3;

EnergyFunction* generate_MRF_smoothing()
{
	int i, j;
	
	// generate function
	for (i=0; i<K; i++)
	for (j=i; j<K; j++)
	{
		V[i*K+j] = V[j*K+i] = (i == j) ? 0 : 4;
	}
	MRF::CostVal* ptr;
	//for (ptr=&D[0]; ptr<&D[sizeX*sizeY*K]; ptr++) *ptr = rand() % 10;
	for(int k = 0 ; k < K ; k++){
		for(int i = 0 ; i < sizeY ; i++){
			for(int j = 0 ; j < sizeX ; j++){
				if(i < j){
					if(k == 2)
						D[K*(i*sizeX+j) + k] = 0;
					else if(k == 1)
						D[K*(i*sizeX+j) + k] = 1;
					else
						D[K*(i*sizeX+j) + k] = 1;
				}
				else if(i == j){
					if(k == 3)
						D[K*(i*sizeX+j) + k] = 0;
					else
						D[K*(i*sizeX+j) + k] = 1;
				}
				else{
					if(k == 2)

						D[K*(i*sizeX+j) + k] = 1;
					else if(k == 1)
						D[K*(i*sizeX+j) + k] = 0;
					else
						D[K*(i*sizeX+j) + k] = 1;
				}
				/*if(i < 5 && i > 0 && j > 0 && j < 5)
				{
					if(k == 2)
						D[K*(i*sizeX+j) + k] = 0;
					else if(k == 1)
						D[K*(i*sizeX+j) + k] = 1;
					else
						D[K*(i*sizeX+j) + k] = 1;
				}
				else if(i > 1 && i < 8 && j > 2 && j < 7)
				{
					if(k == 2)
						D[K*(i*sizeX+j) + k] = 1;
					else if(k == 1)
						D[K*(i*sizeX+j) + k] = 0;
					else
						D[K*(i*sizeX+j) + k] = 1;
				}

			}*/
			}
		}
	}
	//for (ptr=&hCue[0]; ptr<&hCue[sizeX*sizeY]; ptr++) *ptr = 1; // negative multiplier possible
	for(int i = 0 ; i < sizeY ; i++){
		for(int j = 0 ; j < sizeX ; j++){
			if(i < j){
				hCue[i*sizeX+j] = 1;
			}
			else{
				hCue[i*sizeX+j] = 1;
			}

		}
	}
	//for (ptr=&vCue[0]; ptr<&vCue[sizeX*sizeY]; ptr++) *ptr = 1;
	for(int i = 0 ; i < sizeY ; i++){
		for(int j = 0 ; j < sizeX ; j++){
			if(i < j){
				vCue[i*sizeX+j] = 1;
			}
			else{
				vCue[i*sizeX+j] = 1;
			}

		}
	}

	// allocate eng
	DataCost *data         = new DataCost(D);
	//SmoothnessCost *smooth = new SmoothnessCost(1, smoothMax,lambda, hCue,vCue);;//new SmoothnessCost(V,hCue,vCue);
	SmoothnessCost *smooth = new SmoothnessCost(V,hCue,vCue);
	EnergyFunction *eng    = new EnergyFunction(data,smooth);
	
	return eng;
}

int main_mrf(void){

	MRF* mrf;
	EnergyFunction *eng;
	MRF::EnergyVal E;
	float t,tot_t;
	int iter;


	// There are 4 sample energies below to play with. Uncomment 1 at a time 

	eng = generate_MRF_smoothing();

	////////////////////////////////////////////////
	//          Belief Propagation                //
	////////////////////////////////////////////////

	printf("\n*******  Started MaxProd Belief Propagation *****\n");
	mrf = new MaxProdBP(sizeX,sizeY,K,eng);
	mrf->initialize();
	mrf->clearAnswer();
	
	E = mrf->totalEnergy();
	printf("Energy at the Start= %d (%d,%d)\n", E,mrf->smoothnessEnergy(),mrf->dataEnergy());

	tot_t = 0;
	for (iter=0; iter < 10; iter++)
	{
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %d (%f secs)\n", E, tot_t);
	}
	
	for(int j = 0 ; j < sizeY ; j++){
		for(int i = 0 ; i < sizeX; i++){
			printf("%d ", mrf->getLabel(j*sizeX+i));
			
		}
		printf("\n");
	}

	
	delete mrf;

	getchar();

	return 0;
}