#include "Color.h"
#include <cmath>

#ifndef MAX
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#endif

#ifndef MIN
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#endif

#ifndef FABS
#define FABS(x) ((x) > 0 ? (x) : (-x))
#endif

#ifndef PI
#define PI 3.141592653589793238462643383279502884197169399375105820974944592f
#endif

inline void RGBtoLAB( float red, float green, float blue, float &l, float &a, float &b )
{
	double X, Y, Z;
	double f1 = 1/sqrt(3.0),f2 = 1/sqrt(6.0),f3 = 1/sqrt(2.0);

	//shift up to avoid zero
	red *= 255.0f;	green *= 255.0f;	blue *= 255.0f;
	red += 1.0f; green += 1.0f; blue += 1.0f;

	X = 0.3811f*red + 0.5783f*green + 0.0402f*blue;
	Y = 0.1967f*red + 0.7244f*green + 0.0782f*blue;
	Z = 0.0241f*red + 0.1288f*green + 0.8444f*blue;

	X = log(X/0.9996); Y = log(Y/0.9993); Z = log(Z/0.9973);
  
	l = (float)(f1*X + f1*Y + f1*Z);
	a = (float)(f2*X + f2*Y - 2*f2*Z);
	b = (float)(f3*X - f3*Y);
}

inline void LABtoRGB( float &red, float &green, float &blue, float l, float a, float b )
{  
	double X, Y, Z;
	double f1 = sqrt(3.0)/3,f2 = sqrt(6.0)/6,f3 = sqrt(2.0)/2;

	X = f1*l + f2*a + f3*b;
	Y = f1*l + f2*a - f3*b;
	Z = f1*l - 2*f2*a;

	X = exp(X)*0.9996f; Y = exp(Y)*0.9993f; Z = exp(Z)*0.9973f;

	red   = (float)( 4.4679f*X - 3.5873f*Y + 0.1193f*Z );
	green = (float)(-1.2186f*X + 2.3809f*Y - 0.1624f*Z );
	blue  = (float)( 0.0497f*X - 0.2439f*Y + 1.2045f*Z );
  
	red -= 1.0;	green -= 1.0;	blue -= 1.0;
	red /= 255.0;	green /= 255.0;	blue /= 255.0;

	if( red < 0 ) red = 0;		if( red > 1 ) red = 1;
	if( green < 0 ) green = 0;	if( green > 1 ) green = 1;
	if( blue < 0 ) blue = 0;	if( blue > 1 ) blue = 1;
}

inline void RGBtoCIELAB(float R, float G, float B, int& L, int& a, int& b)
{
  // Convert between RGB and CIE-Lab color spaces
  // Uses ITU-R recommendation BT.709 with D65 as reference white.
  
  double X, Y, Z, fX, fY, fZ;
  R *= 255.0f;
  G *= 255.0f;
  B *= 255.0f;
  
  X = 0.412453*R + 0.357580*G + 0.180423*B;
  Y = 0.212671*R + 0.715160*G + 0.072169*B;
  Z = 0.019334*R + 0.119193*G + 0.950227*B;
  
  X /= (255 * 0.950456);
  Y /=  255;
  Z /= (255 * 1.088754);
  
  if (Y > 0.008856)
	{
	  fY = pow(Y, 1.0/3.0);
	  L = (int)(116.0*fY - 16.0 + 0.5);
	}
  else
	{
	  fY = 7.787*Y + 16.0/116.0;
	  L = (int)(903.3*Y + 0.5);
	}
  
  if (X > 0.008856)
	fX = pow(X, 1.0/3.0);
  else
	fX = 7.787*X + 16.0/116.0;
  
  if (Z > 0.008856)
	fZ = pow(Z, 1.0/3.0);
  else
	fZ = 7.787*Z + 16.0/116.0;
  
  a = (int)(500.0*(fX - fY) + 0.5);
  b = (int)(200.0*(fY - fZ) + 0.5);
}

inline void CIELABtoRGB(float& R, float& G, float& B, int L, int a, int b)
{
  // Convert between RGB and CIE-Lab color spaces
  // Uses ITU-R recommendation BT.709 with D65 as reference white.
  
  double X, Y, Z, fX, fY, fZ;
  float RR, GG, BB;
  
  fY = pow((L + 16.0) / 116.0, 3.0);
  if (fY < 0.008856)
	fY = L / 903.3;
  Y = fY;
  
  if (fY > 0.008856)
	fY = pow(fY, 1.0/3.0);
  else
	fY = 7.787 * fY + 16.0/116.0;
  
  fX = a / 500.0 + fY;          
  if (fX > 0.206893)
	X = pow(fX, 3.0);
  else
	X = (fX - 16.0/116.0) / 7.787;
  
  fZ = fY - b /200.0;          
  if (fZ > 0.206893)
	Z = pow(fZ, 3.0);
  else
	Z = (fZ - 16.0/116.0) / 7.787;
  
  X *= (0.950456 * 255);
  Y *= 255;
  Z *= (1.088754 * 255);
  
  RR = (float)((3.240479*X - 1.537150*Y - 0.498535*Z + 0.5) / 255.0f);
  GG = (float)((-0.969256*X + 1.875992*Y + 0.041556*Z + 0.5) / 255.0f);
  BB = (float)((0.055648*X - 0.204043*Y + 1.057311*Z + 0.5) / 255.0f);
  
  R = RR < 0 ? 0 : RR > 1.0f ? 1.0f : RR;
  G = GG < 0 ? 0 : GG > 1.0f ? 1.0f : GG;
  B = BB < 0 ? 0 : BB > 1.0f ? 1.0f : BB;
}

inline void RGBtoYUV( float R, float G, float B, float& Y, float& U, float& V){
	Y = R *  0.299f + G *  0.587f + B *  0.114f;
    U = R * -0.169f + G * -0.332f + B *  0.500f + 0.5f;
    V = R *  0.500f + G * -0.419f + B * -0.0813f + 0.5f;
}
inline void YUVtoRGB( float& R, float& G, float& B, float Y, float U, float V){
	R = Y + (1.4075f * (V - 0.5f));
    G = Y - (0.3455f * (U - 0.5f)) - (0.7169f * (V - 0.5f));
    B = Y + (1.7790f * (U - 0.5f));
}
//RGB: [0,1], H[0,360), S[0,1], V[0,1]
inline void RGBtoHSV( float r, float g, float b, float &h, float &s, float &v )
{
	float min, max, delta;
	min = MIN( r, g );
	min = MIN( min, b );
	max = MAX( r, g );
	max = MAX( max, b );
	v = max;				// v
	delta = max - min;
	if( max != 0 )
		s = delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		s = 0;
		h = -1;
		return;
	}
	if( r == max )
		h = ( g - b ) / delta;		// between yellow & magenta
	else if( g == max )
		h = 2 + ( b - r ) / delta;	// between cyan & yellow
	else
		h = 4 + ( r - g ) / delta;	// between magenta & cyan
	h *= 60.0f;				// degrees
	if( h < 0 )
		h += 360.0f;
}

inline void HSVtoRGB( float &r, float &g, float &b, float h, float s, float v )
{
	int i;
	float f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		r = g = b = v;
		return;
	}
	h /= 60.0f;			// sector 0 to 5
	i = (int)(floor( h ));
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			r = v;
			g = t;
			b = p;
			break;
		case 1:
			r = q;
			g = v;
			b = p;
			break;
		case 2:
			r = p;
			g = v;
			b = t;
			break;
		case 3:
			r = p;
			g = q;
			b = v;
			break;
		case 4:
			r = t;
			g = p;
			b = v;
			break;
		default:		// case 5:
			r = v;
			g = p;
			b = q;
			break;
	}
}

//Orientation: [0, 360.0f), R[0,1],G[0,1],B[0,1]
inline void OrientationDegreetoRGB(float orientation, float& Red, float& Green, float& Blue){
	float hue = orientation;
	float saturation = 1.0f;
	float value = 1.0f;
	HSVtoRGB(Red, Green, Blue, hue, saturation, value);
}

inline void OrientationRadiantoRGB(float orientation, float& Red, float& Green, float& Blue){
	float hue = orientation * 180.0f / PI;
	float saturation = 1.0f;
	float value = 1.0f;
	HSVtoRGB(Red, Green, Blue, hue, saturation, value);
}

inline void PositionToRGB(float x, float y, int width, int height, float& Red, float& Green, float& Blue){
	float cdist = (float)(x + y) * 180.0f / (width + height);	// use Mahatton distance
	float vdist = (float)(width - x + y) / (width + height);	// use Mahatton distance

	HSVtoRGB(Red,Green,Blue,cdist,vdist,1);
}

inline void NormalDirectionToRGB(float nx, float ny, float nz, float& Red, float& Green, float& Blue){
	Red = (nx+1.0f) * 0.5f;
	Green = (ny+1.0f) * 0.5f;
	Blue = nz;
}

inline void ColorNormalization(float value, float minc, float maxc, float& color){
	color = (value - minc) / (maxc - minc);
}
