#include "colorization.h"
#include "Color.h"
#include "Bitmap.h"
#include <cmath>

void main5(int argv, char* argc){
	int index,width,height,size;
	float *grayImg[3];
	float *markedImg[3];
	bool *isMarked;

	float *result[3];

	readBMP("example.bmp",grayImg[0],grayImg[1],grayImg[2],width,height);
	readBMP("example_marked.bmp",markedImg[0],markedImg[1],markedImg[2],width,height);
	 
	//Fill isMarked array
	size = width *height;
	isMarked = new bool[size];
	for(index=0; index<size; index++){
		if(grayImg[0][index] != markedImg[0][index] || grayImg[1][index] != markedImg[1][index] || grayImg[2][index] != markedImg[2][index]){
			isMarked[index] = true;
		}
		else 
			isMarked[index]=false;


		//markedImg[2][index] = grayImg[2][index];
		//markedImg[1][index] = grayImg[1][index];
		printf("%f\n", grayImg[0][index]);

		RGBtoYUV(grayImg[0][index],grayImg[1][index],grayImg[2][index], grayImg[0][index],grayImg[1][index],grayImg[2][index]);
		RGBtoYUV(markedImg[0][index],markedImg[1][index],markedImg[2][index], markedImg[0][index],markedImg[1][index],markedImg[2][index]);
	}

	writeBMP("colorization_marked.bmp",width,height,markedImg[0],markedImg[1],markedImg[2]);

	//colorization
	result[0] = new float[size];
	result[1] = new float[size];
	result[2] = new float[size];

	colorization(width,height,grayImg,markedImg,isMarked,result);

	float *grayImg1[3]; 
	grayImg1[0] = new float[size];
	grayImg1[1] = new float[size];
	grayImg1[2] = new float[size];
	
	//YUV2RGB
	for(index=0; index<size; index++){
		YUVtoRGB(grayImg[0][index],grayImg[1][index],grayImg[2][index], result[0][index],result[1][index],result[2][index]);
	}

	readBMP("example.bmp",grayImg1[0],grayImg1[1],grayImg1[2],width,height);

	writeBMP("colorization_test.bmp",width,height,grayImg[0],grayImg1[1],grayImg1[2]);

	writeBMP("colorization.bmp",width,height,grayImg[0],grayImg[1],grayImg[2]);

	writeBMP("colorization_result.bmp",width,height,result[0],result[1],result[2]);

	delete [] grayImg[0];
	delete [] grayImg[1];
	delete [] grayImg[2];
	delete [] markedImg[0];
	delete [] markedImg[1];
	delete [] markedImg[2];
	delete [] isMarked;

	delete [] result[0];
	delete [] result[1];
	delete [] result[2];
}
