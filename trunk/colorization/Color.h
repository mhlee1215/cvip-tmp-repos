#ifndef __COLOR_H__
#define __COLOR_H__

extern inline void RGBtoLAB( float red, float green, float blue, float &l, float &a, float &b );
extern inline void LABtoRGB( float &red, float &green, float &blue, float l, float a, float b );
extern inline void RGBtoCIELAB(float R, float G, float B, int& L, int& a, int& b);
extern inline void CIELABtoRGB(float& R, float& G, float& B, int L, int a, int b);
extern inline void RGBtoYUV( float R, float G, float B, float& Y, float& U, float& V);
extern inline void YUVtoRGB( float& R, float& G, float& B, float Y, float U, float V);
//RGB: [0,1], H[0,360), S[0,1], V[0,1]
extern inline void RGBtoHSV( float r, float g, float b, float &h, float &s, float &v );
extern inline void HSVtoRGB( float &r, float &g, float &b, float h, float s, float v );

//Orientation: [0, 360.0f), R[0,1],G[0,1],B[0,1]
extern inline void OrientationDegreetoRGB(float orientation, float& Red, float& Green, float& Blue);
//Orientation: [0, 2*PI), R[0,1],G[0,1],B[0,1]
extern inline void OrientationRadiantoRGB(float orientation, float& Red, float& Green, float& Blue);
extern inline void PositionToRGB(float x, float y, int width, int height, float& Red, float& Green, float& Blue);
extern inline void NormalDirectionToRGB(float nx, float ny, float nz, float& Red, float& Green, float& Blue);

extern inline void ColorNormalization(float value, float minc, float maxc, float& color);

#endif