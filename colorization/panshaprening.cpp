// The code from Alejandro with some slight modifications
// Also modified by M.Lee to implement colorization.

//Common
#include <cmath>
#include <iostream>
#include <vector>

//Basic
#include "colorization.h"
#include "Color.h"
#include "Bitmap.h"

//OPENCV
#include <opencv2/opencv.hpp>

//TAUCS linear equation solver
#include "Taucs_matrix.h"

//MRF 
#include "mrf.h"
#include "ICM.h"
#include "GCoptimization.h"
#include "MaxProdBP.h"

#define USE_DEBUG 0
#define DUMP_GRAY 1
#define DUMP_RGB 2
#define DUMP_FLOAT 1
#define DUMP_INT 2

typedef Taucs_matrix<taucs_double> TaucsMatrix;

using namespace std;

extern "C" {
#include <taucs.h>
}

//Colorization
void colorization_actual(int width, int height, float* grayImg[3], float *markedImg[3], float *result[3]);
double rootMeanSquareError(IplImage* src1, IplImage* src2);
void cvtCvArr2Array(const CvArr* src, float* dst, int width, int height, bool isUpsideDown = false);
void cvtArray2CvArr(const float* src, CvArr* dst, int width, int height, bool isUpsideDown = false);
void setValue(float* arr, int width, int height, int x, int y, float value);
float getValue(float* arr, int width, int height, int x, int y);
void swapColorSeed(float* grayImg, float* colorImg[3], int width, int height, int old_x, int old_y, int new_x, int new_y);
IplImage* cvtArrays2ColorImg(float* src1, float* src2, float* src3, int width, int height, bool normalize, bool upsidedown);

//Evaluation Part
double rootMeanSquareError(IplImage* src1, IplImage* src2);
int eval_main(void);

//MRF regulation part
EnergyFunction* generate_MRF_smoothing(float* imgMS_PC, float* imgPAN, int imgRatio, int width, int height, int numLabels, float* dataCost);
void MRF_main(float* imgMS_PC, float* imgPAN, int imgRatio, int width, int height, int numLabels, float* dataCost, int* output, int iter=10);

void stretchlim_cpp(IplImage* src, int width, int height, float* result_low_lim, float* result_high_lim, float* result_max, float low_lim=0.02, float high_lim=0.99);
void stretchlim_cpp(float* src, int size, float* result_low_lim, float* result_high_lim, float* result_max, float low_lim=0.02, float high_lim=0.99);
void imajust_cpp(float* src, float* dst, int size, float high_lim, float low_lim, float max);
void imajust_cpp(IplImage* src, IplImage* dst, int width, int height, float high_lim, float low_lim, float max);

void dumpCvArr(IplImage* src, int sx, int sy, int ex, int ey, int channel, int depth);

int main(int argc, char** argv)
{
	string saveFileExt = "bmp";
	int imgRatio = 4;
	int start_of_spectrals = 1;
	int end_of_spectrals = 7;
	int num_of_spectrals = end_of_spectrals - start_of_spectrals;

	//Cropping image from original to small sized one
	int startX = 1735*imgRatio;

	int startY = 1000*imgRatio;
	int sWidth = 300;
	int sHeight = 300;

	string output_prefix = "TEST11_";
	ostringstream image_info;
	image_info << startX;
	output_prefix += image_info.str()+"_";
	image_info.str("");
	image_info << startY;

	output_prefix += image_info.str()+"_";
	image_info.str("");
	image_info << sWidth;
	output_prefix += image_info.str()+"_";
	image_info.str("");
	image_info << sHeight;
	output_prefix += image_info.str()+"_";
	image_info.str("");
	printf("%s\n", output_prefix.c_str());

	int startXMS = startX/imgRatio;
	int startYMS = startY/imgRatio;
	int widthMS = sWidth/imgRatio;
	int heightMS = sHeight/imgRatio;
	string outName = "";
	

	IplImage** pan_parts = (IplImage**)malloc(sizeof(IplImage*)*num_of_spectrals);
	IplImage* pan = NULL;
	//string readFileName = "";
	string readPANImagePath = "C:/Users/user/Desktop/lab/2011.06.27/pan/PAN.tif";

	string readMSImagePrefix = "C:/Users/user/Desktop/lab/2011.06.27/pan/MS";
	string readMSImagePostfix = "_one.tif";
	string readMSImagePath = "";
	int pan_width, pan_height;

	//Load large but grayscaled pan image.
	readPANImagePath = "C:/Users/user/Desktop/lab/2011.06.27/pan/PAN.tif";

	pan = cvLoadImage(readPANImagePath.c_str(), CV_LOAD_IMAGE_UNCHANGED);
	pan_width = pan->width;
	pan_height = pan->height;
	printf("****Pan-Sharpening Start****\n");
	printf("PAN depth, %d, %d, %d\n", pan->nChannels, pan->width, pan->height);

	//Load low resolution multi spectral responses image(s)
	for(int i = 0 ; i <= num_of_spectrals ; i++){
		readMSImagePath = readMSImagePrefix;
		ostringstream frameIndex;
		frameIndex << (i+1+start_of_spectrals);
		readMSImagePath += frameIndex.str() + readMSImagePostfix;
		pan_parts[i] = cvLoadImage(readMSImagePath.c_str(), CV_LOAD_IMAGE_ANYDEPTH);
		printf("part[%d] depth, %d, %d, %d\n", i, pan_parts[i]->nChannels, pan_parts[i]->width, pan_parts[i]->height);
	}

	
	dumpCvArr(pan_parts[0], 0, 0, 10, 10, DUMP_GRAY, DUMP_INT);
	cvSaveImage("test_pan_parts.tif", pan_parts[0]);
	getchar();

	cvSetImageROI(pan, cvRect(startX, startY, sWidth, sHeight));
	IplImage* sPAN = cvCreateImage(cvSize(sWidth, sHeight), pan->depth, pan->nChannels);
	cvCopy(pan, sPAN);
	cvResetImageROI(pan);
	IplImage** span_parts = (IplImage**)malloc(sizeof(IplImage*)*num_of_spectrals);
	IplImage** span_parts_norm = (IplImage**)malloc(sizeof(IplImage*)*num_of_spectrals);
	
	for(int i = 0 ; i < num_of_spectrals ; i++){
		span_parts[i] = cvCreateImage(cvSize(widthMS, heightMS), pan_parts[i]->depth, pan_parts[i]->nChannels);
		span_parts_norm[i] = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 1);
		cvSetImageROI(pan_parts[i], cvRect(startXMS, startYMS, widthMS, heightMS));
		cvCopy(pan_parts[i], span_parts[i]);
		//cvEqualizeHist(pan_parts[i], span_parts[i]);
		/*int maxVal = 0;
		for(int x = 0 ; x < span_parts[i]->width ; x++){
			for(int y = 0 ; y < span_parts[i]->height ; y++){
				CvScalar val = cvGet2D(span_parts[i], y, x);
				float iVal = val.val[0];
				if(iVal > maxVal) maxVal = iVal;
			}
		}
		printf("max val at %d spec..: %d\n", i, maxVal);
		for(int x = 0 ; x < span_parts[i]->width ; x++){
			for(int y = 0 ; y < span_parts[i]->height ; y++){
				CvScalar val = cvGet2D(span_parts[i], y, x);
				float iVal = val.val[0];
				cvSetReal2D(span_parts_norm[i], y, x, iVal/maxVal);
			}
		}*/

		/*float low, high, max;
		stretchlim_cpp(span_parts[i], widthMS,heightMS, &low, &high, &max);
		printf("....%f, %f\n", low, high);*/
		//getchar();
		
		cvScale(span_parts[i], span_parts_norm[i]);
		//cvNormalize(span_parts[i], span_parts_norm[i], 0, 1, CV_MINMAX); 
		//imajust_cpp(span_parts[i], span_parts_norm[i], widthMS, heightMS, high, low, max);
		//cvNormalize(span_parts[i], span_parts_norm[i], low, high, CV_MINMAX); 
		//cvCopy(span_parts[i], span_parts_norm[i]);
		cvResetImageROI(pan_parts[i]);
	}

	float* original_img[3];
	original_img[0] = new float[widthMS*heightMS];
	original_img[1] = new float[widthMS*heightMS];
	original_img[2] = new float[widthMS*heightMS];
	cvtCvArr2Array(span_parts_norm[3], original_img[0], widthMS, heightMS, true);
	cvtCvArr2Array(span_parts_norm[1], original_img[1], widthMS, heightMS, true);
	cvtCvArr2Array(span_parts_norm[0], original_img[2], widthMS, heightMS, true);

	
	//dumpCvArr(span_parts[0], 0, 0, 10, 10, DUMP_GRAY, DUMP_INT);
	//getchar();
	

	outName = output_prefix+"original.tif";
	writeBMP(outName.c_str(),widthMS,heightMS,original_img[0],original_img[1],original_img[2]);

	if(USE_DEBUG){
		//Save original to BMP
		float* original_img[3];
		original_img[0] = new float[widthMS*heightMS];
		original_img[1] = new float[widthMS*heightMS];
		original_img[2] = new float[widthMS*heightMS];
		cvtCvArr2Array(span_parts_norm[3], original_img[0], widthMS, heightMS, true);
		cvtCvArr2Array(span_parts_norm[1], original_img[1], widthMS, heightMS, true);
		cvtCvArr2Array(span_parts_norm[0], original_img[2], widthMS, heightMS, true);

		outName = output_prefix+"original.png";
		IplImage* ori_img = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 3);
		writeBMP(outName.c_str(),widthMS,heightMS,original_img[0],original_img[1],original_img[2]);

		IplImage* out_ori_R = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 1);
		IplImage* out_ori_G = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 1);
		IplImage* out_ori_B = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 1);

		IplImage* out_ori_img = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 3);

		cvtArray2CvArr(original_img[0], out_ori_R, widthMS, heightMS, true);
		cvtArray2CvArr(original_img[1], out_ori_G, widthMS, heightMS, true);
		cvtArray2CvArr(original_img[2], out_ori_B, widthMS, heightMS, true);

		cvMerge(out_ori_B, out_ori_G, out_ori_R, NULL, out_ori_img);

		cvSaveImage(outName.c_str(), out_ori_img);

	}

	

	if(USE_DEBUG){
		IplImage* spec_RGB_8u = cvCreateImage(cvSize(widthMS, heightMS), IPL_DEPTH_32F, 3);
		cvMerge(span_parts_norm[0], span_parts_norm[1], span_parts_norm[3], NULL, spec_RGB_8u);
		cvShowImage("merge", spec_RGB_8u);
		cvWaitKey(0);
	}


	IplImage** span_parts_resize = (IplImage**)malloc(sizeof(IplImage*)*num_of_spectrals);
	printf("Upsampling..\n");
	//Upsampling.
	for(int i = 0 ; i < num_of_spectrals ; i++){
		span_parts_resize[i] = cvCreateImage(cvSize(sWidth, sHeight), IPL_DEPTH_16U, 1);
		cvResize(span_parts[i], span_parts_resize[i], CV_INTER_CUBIC);
	}

	int width = sWidth;//pan_parts[0]->width;
	int height = sHeight;//pan_parts[0]->height;
	
	CvMat* matA = cvCreateMat(width*height, num_of_spectrals, CV_32FC1);				//A
	CvMat* matA_transpose = cvCreateMat(num_of_spectrals, width*height, CV_32FC1);		//A'
	CvMat* matA_mul = cvCreateMat(num_of_spectrals, num_of_spectrals, CV_32FC1);		//A'*A

	CvMat* PC = cvCreateMat(sWidth*sHeight, 1, CV_32FC1);								//T(A)
	CvMat* PC_image = cvCreateMat(sHeight, sWidth, CV_32FC1);							//

	CvMat* PC_image_trans = cvCreateMat(PC_image->width, PC_image->height, CV_32FC1);
	IplImage* PC_image_trans2 = cvCreateImage(cvSize(PC_image->height, PC_image->width), IPL_DEPTH_32F, 1);
	IplImage* PC_image_trans2_norm = cvCreateImage(cvGetSize(PC_image_trans2), IPL_DEPTH_32F, 1);

	//To solbe Ax = b equation,

	//Step1) Set Matrix A
	//printf("////Matrix A////\n");
	for(int i = 0 ; i < num_of_spectrals ; i++){	
		for(int j = 0 ; j < width ; j++){
			for(int k = 0 ; k < height ; k++){
				//if(j >= 1673) printf("%d/%d\n", width*height, j*height+k);	
				CvScalar cs = cvGet2D(span_parts_resize[i], k, j);
				double value = cs.val[0];
				//printf("%d, %d, %d\n", cs.val[0], cs.val[1], cs.val[2]);
				cvSetReal2D(matA, j*height+k, i, value);
				//if(i==0 && j < 10 && k < 10)
				//	printf("%d\t", value);
			}
			//printf("\n");
		}
	}

	//Multiply transposed version of A to both sides.
	//Find Transposed version of A
	cvTranspose(matA, matA_transpose);

	//Multiply Transposed A and A
	cvMatMul(matA_transpose, matA, matA_mul);

	CvMat* RS = cvCreateMat(width*height, 1, CV_32FC1);
	for(int j = 0 ; j < width ; j++){
		for(int k = 0 ; k < height ; k++){
			//if(j >= 1673) printf("%d/%d\n", width*height, j*height+k);	
			CvScalar cs = cvGet2D(sPAN, k, j);
			double value = cs.val[0];
			//printf("%d\n", value);
			cvSetReal2D(RS, j*height+k, 0, value);
		}
	}

	CvMat* RS_mul = cvCreateMat(num_of_spectrals, 1, CV_32FC1);
	cvMatMul(matA_transpose, RS, RS_mul);

	int dim = num_of_spectrals;

	vector<double> f(dim); // right-hand size vector object

	// create right-hand size vector object
	for(int i = 0 ; i < dim ; i++){
		//CvScalar cs = cvGet2D(matA_mul, k, j);
		//int value = cs.val[0];
		double v = cvGetReal2D(RS_mul, i, 0);
		f[i] = v;

	}
  
	TaucsMatrix* Lmatrix1 = new TaucsMatrix(dim,dim,false);

	for(int i = 0 ; i < dim ; i++){
		for(int j = 0 ; j < dim ; j++){
			CvScalar cs = cvGet2D(matA_mul, j, i);
			double value = cs.val[0];
			Lmatrix1->set_coef(j, i, value);
		}
	}

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

		CvMat* xMat = cvCreateMat(dim, 1, CV_32FC1);
		//CvMat* xMat_test = cvCreateMat(dim/2, 2, CV_32FC1);
		for (unsigned j = 0; j < f.size(); j++){
			cout << x[j] << endl;
			cvSetReal2D(xMat, j, 0, x[j]);
		}
		//cvReshape(xMat, xMat_test, 0, dim/2);

		/*printf("TEST....\n");
		for(int i = 0 ; i < dim/2 ; i++){
			for(int j = 0 ; j < 2 ; j++){
				double val = cvGetReal2D(xMat_test, i, j);
				printf("%f\t", val);
			}
			printf("\n");
		}*/

		
		cvMatMul(matA, xMat, PC);
		cvReshape(PC, PC_image, 0, sWidth);

		PC_image_trans = cvCreateMat(PC_image->width, PC_image->height, CV_32FC1);
		PC_image_trans2 = cvCreateImage(cvSize(PC_image->height, PC_image->width), IPL_DEPTH_32F, 1);
		
		PC_image_trans2_norm = cvCreateImage(cvGetSize(PC_image_trans2), IPL_DEPTH_32F, 1);

		printf("%d/%d, %d/%d\n", PC_image->width, PC_image->height, PC_image_trans->width, PC_image_trans->height);
		cvTranspose(PC_image, PC_image_trans);
		cvGetImage(PC_image_trans, PC_image_trans2);
		printf("%d/%d, %d/%d\n", PC_image_trans2->width, PC_image_trans2->height, PC_image_trans2_norm->width, PC_image_trans2_norm->height);
		cvNormalize(PC_image_trans2, PC_image_trans2_norm, 0, 1, CV_MINMAX); 
		
	}


	// deallocate the factorization
	taucs_linsolve(NULL, &F, 0, NULL, NULL, NULL, NULL);

	if(USE_DEBUG){
		for(int i = 0 ; i < dim ; i++)
			printf("%f\n", xv[i]);

		string outname = "test.mtx";
		char* str = (char*)malloc(sizeof(char)*20);
		sprintf(str, "test.mtx");
		taucs_ccs_write_ijv(B, str);
	}


	//Colorization
	float* grayImg = (float*)malloc(sizeof(float)*sWidth*sHeight);
	float** markedImg = (float**)malloc(sizeof(float*)*num_of_spectrals);
	float** color_map = (float**)malloc(sizeof(float*)*num_of_spectrals);
	float** colorImg = (float**)malloc(sizeof(float*)*num_of_spectrals);
	float** resultImg = (float**)malloc(sizeof(float*)*num_of_spectrals);
	for(int i = 0 ; i < num_of_spectrals ; i++){
		markedImg[i] = (float*)malloc(sizeof(float)*sWidth*sHeight);
		color_map[i] = (float*)malloc(sizeof(float)*sWidth*sHeight);
		resultImg[i] = (float*)malloc(sizeof(float)*sWidth*sHeight);
		colorImg[i] = (float*)malloc(sizeof(float)*widthMS*heightMS);
	}
	float *grayImg_param[3];
	grayImg_param[0] = new float[sWidth*sHeight];
	grayImg_param[1] = new float[sWidth*sHeight];
	grayImg_param[2] = new float[sWidth*sHeight];

	float *markedImg_param[3];
	markedImg_param[0] = new float[sWidth*sHeight];
	markedImg_param[1] = new float[sWidth*sHeight];
	markedImg_param[2] = new float[sWidth*sHeight];

	float *color_map_param[3];
	color_map_param[0] = new float[sWidth*sHeight];
	color_map_param[1] = new float[sWidth*sHeight];
	color_map_param[2] = new float[sWidth*sHeight];

	float *result_param[3];
	result_param[0] = new float[sWidth*sHeight];
	result_param[1] = new float[sWidth*sHeight];
	result_param[2] = new float[sWidth*sHeight];

	int numLabels = 4;
	
	int* labelsMRF = new int[widthMS*heightMS];					//Result of MRF regulation
	float* pcArray = new float[widthMS*heightMS];					//MS input
	float* dataCost = new float[widthMS*heightMS*numLabels];		//Data cost input
	POINT* minLocations = new POINT[widthMS*heightMS*numLabels];	//Minimum location information

	//0: no re-loc
	//1: whole sim re-loc
	//2: MRF based re-loc
	for(int exp = 0 ; exp < 3 ; exp++){
		if(exp == 1) continue;

		IplImage* sPAN_float_normal = cvCreateImage(cvGetSize(sPAN), IPL_DEPTH_32F, 1);
		cvNormalize(sPAN, sPAN_float_normal, 0, 1, CV_MINMAX); 

		printf("Initilize gray scale image (PAN)...%d x %d\n", sWidth, sHeight);
		cvtCvArr2Array(sPAN_float_normal, grayImg, sWidth, sHeight, true);

		printf("Initilize multi spectral image (MS)...%d x %d\n", widthMS, heightMS);
		for(int spec = 0 ; spec < num_of_spectrals ; spec++){
			memcpy(markedImg[spec], grayImg, sizeof(float)*sWidth*sHeight);
		}

		printf("Set color seed from MS images to gray scaled image...%d x %d\n", widthMS, heightMS);
		for(int spec = 0 ; spec < num_of_spectrals ; spec++){
			//printf("%d th(st,nd,rd) spectral image processing now..\n", spec);
			for(int j = 0 ; j < heightMS ; j++){
				for(int i = 0 ; i < widthMS ; i++){
					double val = cvGetReal2D(span_parts_norm[spec], j, i);
					int index = ((heightMS-1-j)*imgRatio+imgRatio/2)*sWidth + i*imgRatio+imgRatio/2;
					markedImg[spec][index] = val;			
					color_map[spec][index] = val;			
				}
			}
			cvtCvArr2Array(span_parts_norm[spec], colorImg[spec], widthMS, heightMS, true);
		}

		printf("Colorization start...\n");
	
		memcpy(grayImg_param[0], grayImg, sizeof(float)*sWidth*sHeight);
		memcpy(grayImg_param[1], grayImg, sizeof(float)*sWidth*sHeight);
		memcpy(grayImg_param[2], grayImg, sizeof(float)*sWidth*sHeight);

		outName = output_prefix+"gray."+saveFileExt;
		/*IplImage* gray_img = cvtArrays2ColorImg(grayImg_param[0],grayImg_param[1],grayImg_param[2], width, height, true, true);
		cvSaveImage(outName.c_str(), gray_img);
		cvShowImage("ttt", gray_img);
		cvWaitKey(0);*/
		writeBMP(outName.c_str(),width,height,grayImg_param[0],grayImg_param[1],grayImg_param[2]);

		//writeBMP("test_1.bmp",width,height,grayImg,grayImg,grayImg);

		for(int spec = 0 ; spec < num_of_spectrals ; spec++){
			printf("===========================================\n");
			printf("SPECT NUM: %d\n", spec);
			memcpy(grayImg_param[0], grayImg, sizeof(float)*sWidth*sHeight);
			memcpy(grayImg_param[1], grayImg, sizeof(float)*sWidth*sHeight);
			memcpy(grayImg_param[2], grayImg, sizeof(float)*sWidth*sHeight);

			memcpy(markedImg_param[0], grayImg, sizeof(float)*sWidth*sHeight);
			memcpy(markedImg_param[1], markedImg[spec], sizeof(float)*sWidth*sHeight);
			memcpy(markedImg_param[2], grayImg, sizeof(float)*sWidth*sHeight);

			memcpy(color_map_param[0], color_map[3], sizeof(float)*sWidth*sHeight);
			memcpy(color_map_param[1], color_map[2], sizeof(float)*sWidth*sHeight);
			memcpy(color_map_param[2], color_map[1], sizeof(float)*sWidth*sHeight);

			//**********************************
			//**COLOR SEED RELOCATION PROCESS **
			//**********************************

			//grayImg (High Resolution Grayscale Image)
			//PC_image_trans2_norm (Low Resolution Grayscale Image);
			//markedImg_param[0,1,2] : gray + color seed image
			//color_map[0,1,2] : color seed image

			IplImage* PC_image_trans2_norm_MS = cvCreateImage(cvSize(widthMS, heightMS), PC_image_trans2_norm->depth, PC_image_trans2_norm->nChannels);
			cvResize(PC_image_trans2_norm, PC_image_trans2_norm_MS);

	

			cvtCvArr2Array(PC_image_trans2_norm_MS, pcArray, widthMS, heightMS, true);
	
			//Find minimum locations and costs per each qudrant.
			printf("Find minimum locations and costs per each qudrant.\n");
			for(int i = 0 ; i < widthMS ; i++){
				for(int j = 0 ; j < heightMS ; j++){
					//Find most similar pixel for the location of color seed.
					double pc_value = cvGetReal2D(PC_image_trans2_norm_MS, j, i);

					int start_x = 0;
					int start_y = 0;
					for(int k = 0 ; k < numLabels ; k++){

						//1st quadrant
						if(k == 0){
							start_x = i*imgRatio+imgRatio/2;
							start_y = j*imgRatio;
						}
						//2nd quadrant
						else if(k == 1){
							start_x = i*imgRatio;
							start_y = j*imgRatio;
						}
						//3rd quadrant
						else if(k == 2){
							start_x = i*imgRatio;
							start_y = j*imgRatio+imgRatio/2;
						}
						//4th quadrant
						else if(k == 3){
							start_x = i*imgRatio+imgRatio/2;
							start_y = j*imgRatio+imgRatio/2;
						}

						float minSimValue = 9999999;
						int min_x = -1;
						int min_y = -1;
						for(int smallX = 0 ; smallX < imgRatio/2 ; smallX++){
							for(int smallY = 0 ; smallY < imgRatio/2 ; smallY++){
								int x = start_x+smallX;
								int y = start_y+smallY;
								float curSimValue = abs(pc_value - getValue(grayImg, sWidth, sHeight, x, y));
								if( curSimValue < minSimValue){
									minSimValue = curSimValue;
									//printf("minVal:%f\n", minSimValue);
									min_x = x;
									min_y = y;
								}
							}
						}

						dataCost[numLabels*(j*widthMS+i) + k] = 1000*minSimValue;
						minLocations[numLabels*(j*widthMS+i) + k].x = min_x;
						minLocations[numLabels*(j*widthMS+i) + k].y = min_y;
					}

				}
			}

			printf("Find minimum locations and costs per each qudrant end.\n");
			printf("MRF regulation start.\n");
			MRF_main(pcArray, grayImg, imgRatio, widthMS, heightMS, numLabels, dataCost, labelsMRF, 10);
			printf("MRF regulation end\n");

			if(exp == 1){
				//Find direction
				for(int i = 0 ; i < widthMS ; i++){
					for(int j = 0 ; j < heightMS ; j++){
						//for(int direction = 0 ; i < direction ; direction++){
						//}

						//Find most similar pixel for the location of color seed.
						double pc_value = cvGetReal2D(PC_image_trans2_norm_MS, j, i);
						int new_x = -1;
						int new_y = -1;

						int old_x = i*imgRatio + imgRatio/2;
						int old_y = j*imgRatio + imgRatio/2;

						float minSimValue = 9999999;
						for(int x = i*imgRatio ; x < (i+1)*imgRatio ; x++){
							for(int y = j*imgRatio ; y < (j+1)*imgRatio ; y++){
								float curSimValue = abs(pc_value - getValue(grayImg, sWidth, sHeight, x, y));
								if( curSimValue < minSimValue){
									minSimValue = curSimValue;
									new_x = x;
									new_y = y;
								}
							}
						}

						//printf("change color seed location %d/%d to %d/%d\n", old_x, old_y, new_x, new_y);
						swapColorSeed(grayImg, markedImg_param, sWidth, sHeight, old_x, old_y, new_x, new_y);
						swapColorSeed(NULL, color_map_param, sWidth, sHeight, old_x, old_y, new_x, new_y);
					}
				}
			}
			if(exp == 2){
			//Find new location with MRF regulation
				for(int i = 0 ; i < widthMS ; i++){
					for(int j = 0 ; j < heightMS ; j++){
						int new_x = -1;
						int new_y = -1;

						int old_x = i*imgRatio + imgRatio/2;
						int old_y = j*imgRatio + imgRatio/2;

						int bestQuadrantIdx = labelsMRF[i*heightMS + j];
						//printf("%d\n", bestQuadrantIdx);
						new_x = minLocations[numLabels*(j*widthMS+i) + bestQuadrantIdx].x;
						new_y = minLocations[numLabels*(j*widthMS+i) + bestQuadrantIdx].y;

						//printf("change color seed location %d/%d to %d/%d\n", old_x, old_y, new_x, new_y);
						swapColorSeed(grayImg, markedImg_param, sWidth, sHeight, old_x, old_y, new_x, new_y);
						swapColorSeed(NULL, color_map_param, sWidth, sHeight, old_x, old_y, new_x, new_y);
					}
				}
			}

	

			string middle = "";
			if(exp == 0) middle = "_no_reloc";
			else if(exp == 1) middle = "_sim_reloc";
			else if(exp == 2) middle = "_mrf_reloc";

			ostringstream specIndex;
			specIndex << (spec);
			middle += "_"+specIndex.str();
			

			outName = output_prefix+"color_Seed_marked"+middle+"."+saveFileExt;
			writeBMP(outName.c_str(),width,height,markedImg[spec],grayImg, grayImg);
			outName = output_prefix+"color_Seed_marked_reloc"+middle+"."+saveFileExt;
			writeBMP(outName.c_str(),width,height,markedImg_param[0],markedImg_param[1],markedImg_param[2]);
			outName = output_prefix+"color_map"+middle+"."+saveFileExt;
			writeBMP(outName.c_str(),width,height,color_map_param[0],color_map_param[1],color_map_param[2]);

	


			colorization_actual(sWidth, sHeight, grayImg_param, markedImg_param, result_param);

			/*for(int ch = 0 ; ch < 3 ; ch++){
				float low, high, max;
				stretchlim_cpp(grayImg_param[ch], width*height, &low, &high, &max);
				printf("....%f, %f\n", low, high);
				imajust_cpp(grayImg_param[ch], grayImg_param[ch], width*height, high, low, max);
			}*/
		


		

			string outFileName = output_prefix+"color_result"+middle+"."+saveFileExt;
			//ostringstream frameIndex;
			//frameIndex << (0);
			//outFileName += frameIndex.str() + ".bmp";
			writeBMP(outFileName.c_str(),width,height,grayImg_param[0],grayImg_param[1],grayImg_param[2]);

			//for(int spec = 0 ; spec < 3 ; spec++){
			printf("store result..spec #%d\n", spec);
			for(int i = 0 ; i < sWidth*sHeight ; i++){
				resultImg[spec][i] = grayImg_param[0][i];
			}
			//}

		}

		printf("Colorization Finish...\n");
	
		

		for(int iter = 0 ; iter < 1 ; iter++){
			for(int spec = 0 ; spec < num_of_spectrals ; spec++){
				for(int i = 0 ; i < width*height ; i++){
					resultImg[spec][i] = resultImg[spec][i] + x[spec] * ((grayImg[i] - x[spec]*resultImg[spec][i]));
				}
			}
		}

		/*{
			float low, high, max;
			stretchlim_cpp(resultImg[3], width*height, &low, &high, &max);
			imajust_cpp(resultImg[3], resultImg[3], width*height, high, low, max);
		}
		{
			float low, high, max;
			stretchlim_cpp(resultImg[1], width*height, &low, &high, &max);
			imajust_cpp(resultImg[1], resultImg[1], width*height, high, low, max);
		}
		{
			float low, high, max;
			stretchlim_cpp(resultImg[0], width*height, &low, &high, &max);
			imajust_cpp(resultImg[0], resultImg[0], width*height, high, low, max);
		}*/


		string outFileName = output_prefix+"_final_result."+saveFileExt;
		writeBMP(outFileName.c_str(),width,height,resultImg[3],resultImg[1],resultImg[0]);

		if(USE_DEBUG){
			IplImage* test_R = cvCreateImage(cvSize(sWidth, sHeight), IPL_DEPTH_32F, 1);
			IplImage* test_G = cvCreateImage(cvSize(sWidth, sHeight), IPL_DEPTH_32F, 1);
			IplImage* test_B = cvCreateImage(cvSize(sWidth, sHeight), IPL_DEPTH_32F, 1);

			IplImage* test = cvCreateImage(cvSize(sWidth, sHeight), IPL_DEPTH_32F, 3);

			cvtArray2CvArr(grayImg_param[0], test_R, sWidth, sHeight, true);
			cvtArray2CvArr(grayImg_param[1], test_G, sWidth, sHeight, true);
			cvtArray2CvArr(grayImg_param[2], test_B, sWidth, sHeight, true);

			cvMerge(test_B, test_G, test_R, NULL, test);
			cvShowImage("R", test_R);
			cvShowImage("G", test_G);
			cvShowImage("B", test_B);
			cvShowImage("test", test);
			cvWaitKey(0);
		}

		//eval_main();

	}

	cvReleaseImage(&pan);
	cvReleaseImage(&sPAN);
	for(int i = 0 ; i < num_of_spectrals ; i++){
		cvReleaseImage(&pan_parts[i]);
		cvReleaseImage(&span_parts[i]);
		cvReleaseImage(&span_parts_norm[i]);
		delete [] resultImg[i];
		delete[] markedImg[i];
		delete[] color_map[i];
		delete[] colorImg[i];
	}

	delete [] grayImg_param[0];
	delete [] grayImg_param[1];
	delete [] grayImg_param[2];
	delete [] markedImg_param[0];
	delete [] markedImg_param[1];
	delete [] markedImg_param[2];

	return 0;
}

void swapColorSeed(float* grayImg, float* colorImg[3], int width, int height, int old_x, int old_y, int new_x, int new_y){
	float seed_ori = getValue(grayImg, width, height, old_x, old_y);
	float seed_R = getValue(colorImg[0], width, height, old_x, old_y);
	float seed_G = getValue(colorImg[1], width, height, old_x, old_y);
	float seed_B = getValue(colorImg[2], width, height, old_x, old_y);

	//Restore
	setValue(colorImg[0], width, height, old_x, old_y, seed_ori);
	setValue(colorImg[1], width, height, old_x, old_y, seed_ori);
	setValue(colorImg[2], width, height, old_x, old_y, seed_ori);
	
	//Put seed
	setValue(colorImg[0], width, height, new_x, new_y, seed_R);
	setValue(colorImg[1], width, height, new_x, new_y, seed_G);
	setValue(colorImg[2], width, height, new_x, new_y, seed_B);
}

//Modified from the original colorization main procedure.
void colorization_actual(int width, int height, float* grayImg[3], float *markedImg[3], float *result[3]){
	int index,/*width,height,*/size;
	bool *isMarked;

	//Fill isMarked array
	size = width *height;
	isMarked = new bool[size];
	for(index=0; index<size; index++){
		if(grayImg[0][index] != markedImg[0][index] || grayImg[1][index] != markedImg[1][index] || grayImg[2][index] != markedImg[2][index])
			isMarked[index] = true;
		else 
			isMarked[index]=false;

		RGBtoYUV(grayImg[0][index],grayImg[1][index],grayImg[2][index], grayImg[0][index],grayImg[1][index],grayImg[2][index]);
		RGBtoYUV(markedImg[0][index],markedImg[1][index],markedImg[2][index], markedImg[0][index],markedImg[1][index],markedImg[2][index]);
	}

	//colorization
	colorization(width,height,grayImg,markedImg,isMarked,result);

	//YUV2RGB
	for(index=0; index<size; index++){
		YUVtoRGB(grayImg[0][index],grayImg[1][index],grayImg[2][index], result[0][index],result[1][index],result[2][index]);
	}

	delete [] isMarked;
	
}



void setValue(float* arr, int width, int height, int x, int y, float value){
	int index = y*width+x;
	arr[index] = value;
}

float getValue(float* arr, int width, int height, int x, int y){
	if(arr == NULL) return 0;
	int index = y*width+x;
	return arr[index];
}



void cvtCvArr2Array(const CvArr* src, float* dst, int width, int height, bool isUpsideDown){
	for(int j = 0 ; j < height ; j++){
		for(int i = 0 ; i < width ; i++){
			int index = 0;
			if(isUpsideDown)
				index = (height-1-j)*width + i;
			else 
				index = (j)*width + i;
			double Rval = cvGetReal2D(src, j, i); 
			dst[index] = (float)Rval;
			
		}
	}
}

void cvtArray2CvArr(const float* src, CvArr* dst, int width, int height, bool isUpsideDown){
	for(int j = 0 ; j < height ; j++){
		for(int i = 0 ; i < width ; i++){
			int index = 0;
			if(isUpsideDown)
				index = (height-1-j)*width + i;
			else 
				index = (j)*width + i;
			float Rval = src[index];
			cvSetReal2D(dst, j, i, Rval); 
		}
	}
}

int eval_main(void){

	int imgRatio = 4;
	int num_of_spectrals = 6;
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
double rootMeanSquareError(IplImage* src1, IplImage* src2){
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
			double R1 = sc1.val[0];
			double G1 = sc1.val[1];
			double B1 = sc1.val[2];

			CvScalar sc2 = cvGet2D(src2, j, i);
			double R2 = sc2.val[0];
			double G2 = sc2.val[1];
			double B2 = sc2.val[2];

			error += sqrt(pow(R1-R2, 2.0) + pow(G1-G2, 2.0) + pow(B1-B2, 2.0));
		}
	}

	return error;
}

EnergyFunction* generate_MRF_smoothing(float* imgMS_PC, float* imgPAN, int imgRatio, int width, int height, int numLabels, float* dataCost){

	//int i, j;
	MRF::CostVal *V = new MRF::CostVal[numLabels*numLabels];
	for (int i=0; i<numLabels ; i++)
		for (int j=i; j<numLabels ; j++)
		{
			V[i*numLabels+j] = V[j*numLabels+i] = (i == j) ? 0 : 1;
		}
	MRF::CostVal *D = new MRF::CostVal[width*height*numLabels];
	MRF::CostVal *hCue  = new MRF::CostVal[width*height];
	MRF::CostVal *vCue  = new MRF::CostVal[width*height];
	MRF::CostVal smoothMax = 5,lambda=.3;

    // generate function
	//Just copy data cost
	for(int i = 0 ; i < height ; i++){
		for(int j = 0 ; j < width ; j++){
			for(int k = 0 ; k < numLabels ; k++){
				D[numLabels*(i*width+j) + k] = (int)dataCost[numLabels*(i*width+j) + k];
				//printf("%f/%d\t", dataCost[numLabels*(i*width+j) + k], numLabels*(i*width+j) + k);
			}
		}
	}

	//Get horizental smoothness cost
	for(int i = 0 ; i < height ; i++){
		for(int j = 0 ; j < width-1 ; j++){
			//printf("%d/%d\n", i, j);
			float smoothVal = 1/(pow(imgMS_PC[i*width+j], 2));// - imgMS_PC[i*width+j+1], 2)+1);
			hCue[i*width+j] = 500*smoothVal;
		}
	}

	//Get vertical smoothness cost
	for(int i = 0 ; i < height-1 ; i++){
		for(int j = 0 ; j < width ; j++){
			float smoothVal = 1/(pow(imgMS_PC[(i+1)*width+j] - imgMS_PC[i*width+j], 2)+1);
			vCue[i*width+j] = 500*smoothVal;

		}
	}

    // allocate eng
    DataCost *data         = new DataCost(D);
    //SmoothnessCost *smooth = new SmoothnessCost(2, smoothMax,lambda, hCue,vCue);
	SmoothnessCost *smooth = new SmoothnessCost(V, hCue,vCue);
    EnergyFunction *eng    = new EnergyFunction(data,smooth);

    return eng;
}

void MRF_main(float* imgMS_PC, float* imgPAN, int imgRatio, int width, int height, int numLabels, float* dataCost, int* output, int iter_max){
	MRF* mrf;

    EnergyFunction *eng;
    MRF::EnergyVal E;
    float t,tot_t;

    // There are 4 sample energies below to play with. Uncomment 1 at a time 

    eng = generate_MRF_smoothing(imgMS_PC, imgPAN, imgRatio, width, height, numLabels, dataCost);

	////////////////////////////////////////////////
    //          Belief Propagation                //
    ////////////////////////////////////////////////

    printf("\n*******  Started MaxProd Belief Propagation *****\n");
    mrf = new MaxProdBP(width,height,numLabels,eng);
    mrf->initialize();
    mrf->clearAnswer();
    
    E = mrf->totalEnergy();
    printf("Energy at the Start= %d (%d,%d)\n", E,mrf->smoothnessEnergy(),mrf->dataEnergy());

    tot_t = 0;
    for (int iter=0; iter < iter_max; iter++)
    {
        mrf->optimize(1, t);

        E = mrf->totalEnergy();
        tot_t = tot_t + t ;
        printf("energy = %d (%f secs)\n", E, tot_t);
    }
	
	for(int j = 0 ; j < height ; j++){
		for(int i = 0 ; i < width; i++){
			//printf("%d ", mrf->getLabel(j*width+i));
			output[j*width+i] = mrf->getLabel(j*width+i);
		}
		//printf("\n");
	}

    
    delete mrf;
}

IplImage* cvtArrays2ColorImg(float* src1, float* src2, float* src3, int width, int height, bool normalize, bool upsidedown){

	IplImage* comp_R = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	IplImage* comp_G = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	IplImage* comp_B = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);

	IplImage* comp_R_norm = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	IplImage* comp_G_norm = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	IplImage* comp_B_norm = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);

	IplImage* color_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 3);

	if(!normalize){
		cvtArray2CvArr(src1, comp_R_norm, width, height, upsidedown);
		cvtArray2CvArr(src2, comp_G_norm, width, height, upsidedown);
		cvtArray2CvArr(src3, comp_B_norm, width, height, upsidedown);
	}else{
		cvtArray2CvArr(src1, comp_R, width, height, upsidedown);
		cvtArray2CvArr(src2, comp_G, width, height, upsidedown);
		cvtArray2CvArr(src3, comp_B, width, height, upsidedown);

		cvNormalize(comp_R, comp_R_norm, 0, 1, CV_MINMAX);
		cvNormalize(comp_G, comp_G_norm, 0, 1, CV_MINMAX);
		cvNormalize(comp_B, comp_B_norm, 0, 1, CV_MINMAX);
	}

	cvMerge(comp_B_norm, comp_G_norm, comp_R_norm, NULL, color_img);
	
	return color_img;
}

void stretchlim_cpp(IplImage* src, int width, int height, float* result_low_lim, float* result_high_lim, float* result_max, float low_lim, float high_lim){
	float* tempContent = new float[width*height];
	cvtCvArr2Array(src, tempContent, width, height, true);
	stretchlim_cpp(tempContent, width*height, result_low_lim, result_high_lim, result_max, low_lim, high_lim);
	delete[] tempContent;
}

void stretchlim_cpp(float* src, int size, float* result_low_lim, float* result_high_lim, float* result_max, float low_lim, float high_lim){
	
	float* tempContent = new float[size];
	memcpy(tempContent, src, sizeof(float)*size);
	vector<float> contents (tempContent, tempContent+size); 

	sort(contents.begin(), contents.end());

	int low_idx = size*low_lim;
	int high_idx = size*high_lim;

	*result_max = contents[size-1];
	*result_low_lim = contents[low_idx]/contents[size-1];
	*result_high_lim = contents[high_idx]/contents[size-1];

	delete[] tempContent;
	contents.clear();
}

void imajust_cpp(float* src, float* dst, int size, float high_lim, float low_lim, float max){
	//Ajusted max : 1 , min : 0
	for(int i = 0 ; i < size ; i++)
	{
		float val = src[i]/max;
		if(val > high_lim)
			dst[i] = 1;
		else if(val < low_lim)
			dst[i] = 0;
		else{
			dst[i] = 1 - ( (high_lim - val) / (high_lim - low_lim));
		}
	}
}

void imajust_cpp(IplImage* src, IplImage* dst, int width, int height, float high_lim, float low_lim, float max){
	//Ajusted max : 1 , min : 0
	printf("high:%f, low:%f, max:%f\n", high_lim, low_lim, max);
	for(int i = 0 ; i < width ; i++)
	{
		for(int j = 0 ; j < height ; j++){
			CvScalar val = cvGet2D(src, j, i);
			float value = val.val[0]/max;
			if(value > high_lim)
				cvSetReal2D(dst, j, i, 1);
			else if(value < low_lim)
				cvSetReal2D(dst, j, i, 0);
			else{
				double tmpVal = 1 - ((high_lim - value) / (high_lim - low_lim));
				//printf("%f\n", tmpVal);
				cvSetReal2D(dst, j, i, tmpVal);
			}
		}
	}
}

void dumpCvArr(IplImage* src, int sx, int sy, int ex, int ey, int channel, int depth){
	for(int i = sx ; i < ex ; i++){
		for(int j = sy ; j < ey ; j++){
			CvScalar val = cvGet2D(src, j, i);
			int v0 = val.val[0];
			int v1 = val.val[1];
			int v2 = val.val[2];
			if(channel == DUMP_GRAY){
				if(depth == DUMP_FLOAT) printf("%f\t", v0);
				else if(depth == DUMP_INT) printf("%d\t", v0);
			}
			else if(channel == DUMP_RGB){
				if(depth == DUMP_FLOAT) printf("%f/%f/%f\t", v0, v1, v2);
				else if(depth == DUMP_INT) printf("%d/%d/%d\t", v0, v1, v2);
			}

		}
		printf("\n");
	}
}