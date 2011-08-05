#include <stdio.h>
#include <opencv2/opencv.hpp>

double rootMeanSquareError(IplImage* src1, IplImage* src2);

int main_5(void){

	int imgRatio = 4;
	int num_of_spectrals = 8;
	IplImage** pan_parts = (IplImage**)malloc(sizeof(IplImage*)*num_of_spectrals);
	
	IplImage* pan = NULL;

	string readFileName = "";

	int pan_width, pan_height;
	//Load large but grayscaled pan image.
	readFileName = "C:/Users/user/Desktop/lab/2011.06.27/pan/PAN.tif";
	pan = cvLoadImage(readFileName.c_str(), CV_LOAD_IMAGE_UNCHANGED);
	pan_width = pan->width;
	pan_height = pan->height;
	printf("PAN depth, %d, %d, %d\n", pan->nChannels, pan->width, pan->height);

	//Load low resolution multi spectral responses image(s)
	for(int i = 0 ; i < num_of_spectrals ; i++){
		readFileName = "C:/Users/user/Desktop/lab/2011.06.27/pan/MS";
		ostringstream frameIndex;
		frameIndex << (i+1);
		readFileName += frameIndex.str() + "_one.tif";
		pan_parts[i] = cvLoadImage(readFileName.c_str(), CV_LOAD_IMAGE_UNCHANGED);
		printf("part[%d] depth, %d, %d, %d\n", i, pan_parts[i]->nChannels, pan_parts[i]->width, pan_parts[i]->height);
	}



	IplImage* originalMS;
	string originalMSFileName = "C:/Users/user/Desktop/Colorization/original.bmp";
	originalMS = cvLoadImage(originalMSFileName.c_str());
	IplImage* createdHS;
	string createdHSFileName = "C:/Users/user/Desktop/Colorization/colorization_0.bmp";
	//string createdHSFileName = "C:/Users/user/Desktop/Colorization/colorization_no_reloc.bmp";
	createdHS = cvLoadImage(createdHSFileName.c_str());

	IplImage* createdHS_resize = cvCreateImage(cvGetSize(originalMS), originalMS->depth, originalMS->nChannels);
	cvResize(createdHS, createdHS_resize);

	printf("original    %d, %d, \n", originalMS->width, originalMS->height);
	printf("synthesized %d, %d, \n", createdHS->width, createdHS->height);
	printf("synthesized %d, %d, \n", createdHS_resize->width, createdHS_resize->height);

	double resultDiff = rootMeanSquareError(originalMS, createdHS_resize);

	printf("Result Differ is %f\n", resultDiff);

	getchar();

	return 0;
}


//Performance evaluation function.
//Suppose each Input have three channel R, G, B
double rootMeanSquareError1(IplImage* src1, IplImage* src2){
	double error = 0.0;

	if(src1->width != src2->width){
		printf("ERROR: inputs have different width.\n");
		return -1;
	}
	if(src1->height != src2->height){
		printf("ERROR: inputs have different height.\n");
		return -1;
	}
	if(src1->depth != src2->depth){
		printf("ERROR: inputs have different depth.\n");
		return -1;
	}
	if(src1->nChannels != src2->nChannels){
		printf("ERROR: inputs have different number of channels.\n");
		return -1;
	}
	

	int w = src1->width;
	int h = src2->height;

	for(int i = 0 ; i < w ; i++){
		for(int j = 0 ; j < h ; j++){
			CvScalar sc1 = cvGet2D(src1, j, i);
			int R1 = sc1.val[0];
			int G1 = sc1.val[1];
			int B1 = sc1.val[2];

			CvScalar sc2 = cvGet2D(src2, j, i);
			int R2 = sc2.val[0];
			int G2 = sc2.val[1];
			int B2 = sc2.val[2];

			error += sqrt(pow(R1-R2, 2.0) + pow(G1-G2, 2.0) + pow(B1-B2, 2.0));
		}
	}

	return error;
}