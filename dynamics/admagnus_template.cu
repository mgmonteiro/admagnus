//-----------------------------------------------------------------------------//

/* 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			 |\__/,|   (`\	+++++++++++++++++++++++++++++++++++++++++
			 |_ _  |.--.) )	+++++++++++++++++++++++++++++++++++++++++
			 ( T   )     /	+++++++++++++++++++++++++++++++++++++++++
			(((^_(((/(((_/	+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++Ad Magnus - Precessional Dynamics for micro and nano systems+++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++Author: Maxwel Gama Monteiro Junior++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++Contact: maxweljr@gmail.com++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


*/
/*

DISCLAIMER: This Version Implements a Predictor-Corrector Spherical Midpoint Method
adapted from Modin and others (2015): https://arxiv.org/pdf/1402.3334.pdf

Code is verbose and lack abstractions in order to be accessible to beginners as well,
but it can be used at production level! Just keep in mind magnetostatic energy uses
the N-Body method: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch31.html

If you want this is pretty easy to take into OOP, most of the code is actually 
reusable past the first steps

*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <string.h>
#include <curand.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//GPU CONSTANTS

static double gyro = (2.0/1.05)*9.27e+10; // Gyromagnetic ratio in [rad/(seconds*Tesla)]
static double mu = 3.1415926535897932*4.0e-07; //Vacuum permeability
static double kB = 1.38064852*(1.0e-23); //Boltzmann Constant
static double e_charge = 1.6021765*1.0e-19; //Charge of the electron
static double muB = 9.2740096820*(1.0e-24); //Bohr Magneton

#define piCPU 3.1415926535897932





/*--------------------------------------------------DEVICE KERNELS--------------------------------*/
/*--------------------------------------------------DEVICE KERNELS--------------------------------*/
/*--------------------------------------------------DEVICE KERNELS--------------------------------*/

//rsqrt with double point precision rounding - substantially faster than sqrt() at a small precision cost,
//if systems blows up, try and print values given by this function

    __device__ __forceinline__ double drsqrt (double a)
    {
      double y, h, l, e;
      unsigned int ilo, ihi, g, f;
      int d;

      ihi = __double2hiint(a);
      ilo = __double2loint(a);
      if (((unsigned int)ihi) - 0x00100000U < 0x7fe00000U){
        f = ihi | 0x3fe00000;
        g = f & 0x3fffffff;
        d = g - ihi;
        a = __hiloint2double(g, ilo); 
        y = rsqrt (a);
        h = __dmul_rn (y, y);
        l = __fma_rn (y, y, -h);
        e = __fma_rn (l, -a, __fma_rn (h, -a, 1.0));
        /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
        y = __fma_rn (__fma_rn (0.375, e, 0.5), e * y, y);
        d = d >> 1;
        a = __hiloint2double(__double2hiint(y) + d, __double2loint(y));
      } else if (a == 0.0) {
        a = __hiloint2double ((ihi & 0x80000000) | 0x7ff00000, 0x00000000);
      } else if (a < 0.0) {
        a = __hiloint2double (0xfff80000, 0x00000000);
      } else if (isinf (a)) {
        a = __hiloint2double (ihi & 0x80000000, 0x00000000);
      } else if (isnan (a)) {
        a = a + a;
      } else {
        a = a * __hiloint2double (0x7fd00000, 0);
        y = rsqrt (a);
        h = __dmul_rn (y, y);
        l = __fma_rn (y, y, -h);
        e = __fma_rn (l, -a, __fma_rn (h, -a, 1.0));
        /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
        y = __fma_rn (__fma_rn (0.375, e, 0.5), e * y, y);
        a = __hiloint2double(__double2hiint(y) + 0x1ff00000,__double2loint(y));
      }
      return a;
    }



__global__ void renormalize(int atoms, double *mx, double *my, double *mz)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

double M = 0.0;
double xx = 0.0;
double yy = 0.0;
double zz = 0.0;

xx = mx[n]*mx[n];
yy = my[n]*my[n];
zz = mz[n]*mz[n];


M = drsqrt(xx + yy + zz);

mx[n] = mx[n]*M;
my[n] = my[n]*M;
mz[n] = mz[n]*M;

n+= blockDim.x * gridDim.x;

}

}

//Measures total magnetization
__global__ void magnetum_opus(int atoms, double *mx, double *my, double *mz, double *Mblockx, double *Mblocky, double *Mblockz, double *mm_)
{

int n = threadIdx.x + blockDim.x * blockIdx.x;

__shared__ double3 cache[256];
__shared__ double  r[256];

double tempx = 0.0, tempy = 0.0, tempz = 0.0, tempr = 0.0;

while(n < atoms)
{

Mblockx[blockIdx.x] = 0.0;
Mblocky[blockIdx.x] = 0.0;
Mblockz[blockIdx.x] = 0.0;
mm_[blockIdx.x] = 0.0;

tempx += mx[n];
tempy += my[n];
tempz += mz[n];

tempr += 1.0/drsqrt(mx[n]*mx[n] + my[n]*my[n] + mz[n]*mz[n]);

n+= blockDim.x * gridDim.x;

}


cache[threadIdx.x].x = tempx;
cache[threadIdx.x].y = tempy;
cache[threadIdx.x].z = tempz;
r[threadIdx.x] = tempr;

__syncthreads();


int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x].x += cache[threadIdx.x + u].x;
			cache[threadIdx.x].y += cache[threadIdx.x + u].y;
			cache[threadIdx.x].z += cache[threadIdx.x + u].z;
			r[threadIdx.x] += r[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0){
 Mblockx[blockIdx.x] = cache[0].x;
 Mblocky[blockIdx.x] = cache[0].y;
 Mblockz[blockIdx.x] = cache[0].z;
 mm_[blockIdx.x] = r[0];
}

}

//If cylindrical radius anisotropy
__global__ void Polar_AxisYZ(int atoms, double *ry, double *rz, double *y, double *z) //Easy axis is normal to tangent planes of a cylindrical surface
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

double theta = atan2(z[n], y[n]);

ry[n] = cos(theta);
rz[n] = sin(theta);

n+= blockDim.x * gridDim.x;
}

}

//If radial anisotropy
__global__ void SphereAxis(int atoms, double *ry, double *rz, double *y, double *z, double *rx, double *x) //Easy axis is normal to tangent planes of a spherical surface
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

double rphi = drsqrt(x[n]*x[n] + y[n]*y[n] + z[n]*z[n]);
double theta = atan2(y[n],x[n]);
double phiz = z[n]*rphi;
double phi = acos(phiz);

rx[n] = cos(theta)*sin(phi);
ry[n] = sin(theta)*sin(phi);
rz[n] = cos(phi);

n+= blockDim.x * gridDim.x;
}

}

__device__ __forceinline__ void Neighbor_Field(int nviz, int *viz, int threshold, double *mnx, double *mny, double *mnz, double *Hex, double* Hey, double* Hez, int *lbl, int lbl1, double*  HDMx, double* HDMy, double* HDMz, double *x, double *y, double *z, double x1, double y1, double z1, double Jl, int curveflag, double ry, double rz)
{

int w = 0;
double exx = 0.0, exy = 0.0, exz = 0.0;
double edx = 0.0, edy = 0.0, edz = 0.0;

	while( w < nviz )
	{

	//Exchange Field
	int j1 = viz[w + threshold];
	int lbl2 = lbl[j1];
	
	if(lbl1 == lbl2)
	{
	exx += mnx[j1];
	exy += mny[j1];
	exz += mnz[j1];
	}
	else if(lbl1 != lbl2)
	{
	exx += Jl*mnx[j1];
	exy += Jl*mny[j1];
	exz += Jl*mnz[j1];
	}

	//Antisymmetric Superficial Exchange Field
	//Non-coplanar interactions are trivially zero
	//The rij vector must be unitary but it is trivially so in a square lattice and first neighbors

	if(curveflag == 1) //Dij = (rij X z) X Sj
	{

	if( abs(z1 - z[j1]) < 0.01)
	{
	double xx = x1 - x[j1];
	double yy = y1 - y[j1];

	edx -= xx*mnz[j1];
	edy -= yy*mnz[j1];
	edz += mnx[j1]*xx + mny[j1]*yy;

	}

	}
	else if(curveflag == 2) //Dij = (rij X n) X Sj, n = (0, cos(theta), sen(theta) ) for the yz plane cylinder!
	{

	double xx = x1 - x[j1];
	double yy = y1 - y[j1];
	double zz = z1 - z[j1];

	edx -= xx*(mnz[j1]*rz + mny[j1]*ry);
	edy += ry*(mnx[j1]*xx + mnz[j1]*zz) - mnz[j1]*yy*rz;
	edz += rz*(mnx[j1]*xx + mny[j1]*yy) - mny[j1]*zz*ry;

	}

	//P.S no DMI implemented for the sphere here (a closed surface has skyrmions at 0 DMI!), but not hard to add :)
	
	w++;
	}

	*Hex = exx;
	*Hey = exy;
	*Hez = exz;
	*HDMx = edx;
	*HDMy = edy;
	*HDMz = edz;

	return;

}

__device__ __forceinline__ void Demag_Field(int blocknatom, double *x, double *y, double *z, double *mx, double *my, double *mz, double3 *r, double3 *m, double* Hdipx, double* Hdipy, double* Hdipz, double x1, double y1, double z1)
{

int p = blockDim.x;

double rij;
double xx;
double yy;
double zz;
double uppx;
double uppy;
double uppz;
double down;
double dipx = 0.0;
double dipy = 0.0;
double dipz = 0.0;
double dip;
double mrjx, mrjy, mrjz;

	//Demagnetizing Field due to dipole-dipole interactions among magnetic moments
	for(int w = 0; w < blocknatom; w += p)
	{

	//Filling the shared memory arrays with the block-to-thread interacting moment pairs

	r[threadIdx.x].x = x[threadIdx.x + w];
	r[threadIdx.x].y = y[threadIdx.x + w];
	r[threadIdx.x].z = z[threadIdx.x + w];

	m[threadIdx.x].x = mx[threadIdx.x + w];
	m[threadIdx.x].y = my[threadIdx.x + w];
	m[threadIdx.x].z = mz[threadIdx.x + w];

	__syncthreads();

		#pragma unroll
		for(int l = 0; l < p; l++)
		{

		xx = x1 - r[l].x;
		yy = y1 - r[l].y;
		zz = z1 - r[l].z;

		rij = drsqrt(xx*xx + yy*yy + zz*zz);

				//no self interaction
			if(isinf(rij) == 0)
			{

			xx = xx*rij;
			yy = yy*rij;
			zz = zz*rij;
		
			mrjx = m[l].x * xx;
			mrjy = m[l].y * yy;
			mrjz = m[l].z * zz;

			down = rij*rij*rij;
	
			dip = mrjx + mrjy + mrjz;

			uppx = (3.0 * dip * xx - m[l].x);
			uppy = (3.0 * dip * yy - m[l].y);
			uppz = (3.0 * dip * zz - m[l].z);

			dipx += uppx*down;
			dipy += uppy*down;
			dipz += uppz*down;

			}


		}

	__syncthreads();

	}


	*Hdipx = dipx;
	*Hdipy = dipy;
	*Hdipz = dipz;

	return;

}

__device__ __forceinline__ void Local_Fields(double ex, double ey, double ez, double mx, double my, double mz, double* Hax, double* Hay, double* Haz, int curveflag, double ry, double rz, double rx)
{

if(curveflag == 1) //n = z
{
*Hax = (mx*ex + my*ey + mz*ez)*ex;
*Hay = (mx*ex + my*ey + mz*ez)*ey;
*Haz = (mx*ex + my*ey + mz*ez)*ez;
}

if(curveflag == 2) //n = (0, ry, rz) = (0, cos t, sin t)
{
*Hax = (my*ry + mz*rz)*ry;
*Hay = (my*ry + mz*rz)*rz;
}

if(curveflag == 3) //n = (rx, ry, rz) = (cos t sin w, sin t sin w, cos w)
{
*Hax = (mx*rx + my*ry + mz*rz)*rx;
*Hay = (mx*rx + my*ry + mz*rz)*ry;
*Haz = (mx*rx + my*ry + mz*rz)*rz;

}

return;

}


//Torque+damp like precession terms of the LLG eq.

__global__ void precessional3D(int atoms, double *x_, double *y_, double *z_,int *viz_, int *n_viz_, double *mx_, double *my_, double *mz_, double D_J, double alpha, double *fx_, double *fy_, double *fz_, double Bx_, double By_, double Bz_, double Alph_g, int *threshold, double pi, int blockround, double ex, double ey, double ez, double lamb, double *D, int *lbl, double Jl, double *Hdipx, double *Hdipy, double *Hdipz, double *Hexx, double *Hexy, double *Hexz, double *HDMx, double *HDMy, double *HDMz, double *Hanisx, double *Hanisy, double *Hanisz, int curveflag, double *ry, double *rz, double *rx)

{

int n = threadIdx.x + blockIdx.x * blockDim.x;	 

//shared memory vectors, of size = p; enforce p = size via extern if needed, but this is the most readable.


__shared__ double3 r[256];
__shared__ double3 m[256];

while(n < blockround)
{



double DM = D[n];
double dot_mm = 0.0;
double dot_mb = 0.0;
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];
double x = x_[n];
double y = y_[n];
double z = z_[n];
double ry_ = ry[n];
double rz_ = rz[n];
double rx_ = rx[n];
double deltax, deltay, deltaz;
int lbl1 = lbl[n];
int offset, nviz;
if(n < atoms) 
{
offset = threshold[n], nviz = n_viz_[n];
}
else 
{
offset = 0, nviz = 0;
}


Neighbor_Field(nviz, viz_, offset, mx_, my_, mz_, &Hexx[n], &Hexy[n], &Hexz[n], lbl, lbl1, &HDMx[n], &HDMy[n], &HDMz[n], x_, y_, z_, x, y, z, Jl, curveflag, ry_, rz_);


		
Demag_Field(blockround, x_, y_, z_, mx_, my_, mz_, r, m, &Hdipx[n], &Hdipy[n], &Hdipz[n], x, y, z);


Local_Fields(ex, ey, ez, mx, my, mz, &Hanisx[n], &Hanisy[n], &Hanisz[n], curveflag, ry_, rz_, rx_);

	deltax = Hexx[n] + DM*HDMx[n] + 0.25*(D_J/pi)*Hdipx[n] + 2.0*lamb*Hanisx[n] + D_J*Bx_;
	deltay = Hexy[n] + DM*HDMy[n] + 0.25*(D_J/pi)*Hdipy[n] + 2.0*lamb*Hanisy[n] + D_J*By_;
	deltaz = Hexz[n] + DM*HDMz[n] + 0.25*(D_J/pi)*Hdipz[n] + 2.0*lamb*Hanisz[n] + D_J*Bz_;

	dot_mb = mx * deltax + my * deltay + mz * deltaz;
	dot_mm = mx * mx + my * my + mz * mz;

	fx_[n] = - Alph_g * ((my * deltaz - mz * deltay) + alpha * ( dot_mb * mx - dot_mm * deltax ) );
	fy_[n] = - Alph_g * ((mz * deltax - mx * deltaz) + alpha * ( dot_mb * my - dot_mm * deltay ) );
	fz_[n] = - Alph_g * ((mx * deltay - my * deltax) + alpha * ( dot_mb * mz - dot_mm * deltaz ) );



n += blockDim.x * gridDim.x;

}


}

//Same as before but for Langevin eq. (yes, can also just reuse code but did not want to call another __device__ let alone extra kernel)

__global__ void precessional3DRand(int atoms, double *x_, double *y_, double *z_,int *viz_, int *n_viz_, double *mx_, double *my_, double *mz_, double D_J, double alpha, double *fx_, double *fy_, double *fz_, double Bx_, double By_, double Bz_, double Alph_g, int *threshold, double pi, int blockround, double fluctu, double *xran, double *yran, double *zran, double ex, double ey, double ez, double lamb, double *D, int *lbl, double Jl, double *Hdipx, double *Hdipy, double *Hdipz, double *Hexx, double *Hexy, double *Hexz, double *HDMx, double *HDMy, double *HDMz, double *Hanisx, double *Hanisy, double *Hanisz, int curveflag, double *ry, double *rz, double *rx)

{

int n = threadIdx.x + blockIdx.x * blockDim.x;	 


__shared__ double3 r[256];
__shared__ double3 m[256];

while(n < blockround)
{


int lbl1 = lbl[n];
double DM = D[n];
double dot_mm = 0.0;
double dot_mb = 0.0;
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];
double ry_ = ry[n];
double rz_ = rz[n];
double rx_ = rx[n];
double x = x_[n];
double y = y_[n];
double z = z_[n];
double deltax, deltay, deltaz;

int offset, nviz;
if(n < atoms) 
{
offset = threshold[n], nviz = n_viz_[n];
}
else 
{
offset = 0, nviz = 0;
}

Neighbor_Field(nviz, viz_, offset, mx_, my_, mz_, &Hexx[n], &Hexy[n], &Hexz[n], lbl, lbl1, &HDMx[n], &HDMy[n], &HDMz[n], x_, y_, z_, x, y, z, Jl, curveflag, ry_, rz_);

//Tile calculation for dipolar interactions - uses the nearest integer blockround >= atoms that is a multiple of p; guaranteed sucessful access to shared memory.


		
Demag_Field(blockround, x_, y_, z_, mx_, my_, mz_, r, m, &Hdipx[n], &Hdipy[n], &Hdipz[n], x, y, z);


//Zeeman External Field Contribution and (Uniaxial, uniform) Anisotropy Field Contribution

Local_Fields(ex, ey, ez, mx, my, mz, &Hanisx[n], &Hanisy[n], &Hanisz[n], curveflag, ry_, rz_, rx_);

	deltax = Hexx[n] + DM*HDMx[n] + 0.25*(D_J/pi)*Hdipx[n] + 2.0*lamb*Hanisx[n] + D_J*Bx_ + fluctu*xran[n];
	deltay = Hexy[n] + DM*HDMy[n] + 0.25*(D_J/pi)*Hdipy[n] + 2.0*lamb*Hanisy[n] + D_J*By_ + fluctu*yran[n];
	deltaz = Hexz[n] + DM*HDMz[n] + 0.25*(D_J/pi)*Hdipz[n] + 2.0*lamb*Hanisz[n] + D_J*Bz_ + fluctu*zran[n];

	deltax += fluctu*xran[n];
	deltay += fluctu*yran[n];
	deltaz += fluctu*zran[n];

	dot_mb = mx * deltax + my * deltay + mz * deltaz;
	dot_mm = mx * mx + my * my + mz * mz;

	fx_[n] = - Alph_g * ((my * deltaz - mz * deltay) + alpha * ( dot_mb * mx - dot_mm * deltax ) );
	fy_[n] = - Alph_g * ((mz * deltax - mx * deltaz) + alpha * ( dot_mb * my - dot_mm * deltay ) );
	fz_[n] = - Alph_g * ((mx * deltay - my * deltax) + alpha * ( dot_mb * mz - dot_mm * deltaz ) );

n += blockDim.x * gridDim.x;

}


}



//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____Time_stepping___XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

__global__ void TimeSkip(int atoms, double *fx, double *fy, double *fz, double *fx_1, double *fy_1, double *fz_1, double *fx_2, double *fy_2, double *fz_2, double *fx_3, double *fy_3, double *fz_3)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n<atoms)
{

fx_3[n] = fx_2[n];
fx_2[n] = fx_1[n];
fx_1[n] = fx[n];

fy_3[n] = fy_2[n];
fy_2[n] = fy_1[n];
fy_1[n] = fy[n];

fz_3[n] = fz_2[n];
fz_2[n] = fz_1[n];
fz_1[n] = fz[n];

n += blockDim.x * gridDim.x;


}

}


__global__ void PC_Adams_Moulton_step(int atoms, double *mx, double *my, double *mz, double *mtempx, double *mtempy, double *mtempz, double *fx, double *fy, double *fz, double *fx_1, double *fy_1, double *fz_1, double *fx_2, double *fy_2, double *fz_2, double *fx_3, double *fy_3, double *fz_3, double dt, int steps, int *lbl)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;


while(n < atoms)
{

double h = dt/24.0;

		if(lbl[n]!=0)
		{

		mtempx[n] = mx[n];
		mtempy[n] = my[n];
		mtempz[n] = mz[n];

		mx[n] = mx[n] + (55.0*fx[n] - 59.0*fx_1[n] + 37.0*fx_2[n] - 9.0*fx_3[n])*h;
		my[n] = my[n] + (55.0*fy[n] - 59.0*fy_1[n] + 37.0*fy_2[n] - 9.0*fy_3[n])*h;
		mz[n] = mz[n] + (55.0*fz[n] - 59.0*fz_1[n] + 37.0*fz_2[n] - 9.0*fz_3[n])*h;

		}
		

n += blockDim.x * gridDim.x;


}

}

//=============================================================================================================

__global__ void PC_Adams_Bashforth_step(int atoms, double *mx, double *my, double *mz, double *mtempx, double *mtempy, double *mtempz, double *fx, double *fy, double *fz, double *fx_1, double *fy_1, double *fz_1, double *fx_2, double *fy_2, double *fz_2, double *fx_3, double *fy_3, double *fz_3, double dt, int *lbl)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;


while(n < atoms)
{

double h = dt/24.0;
		if(lbl[n]!=0)
		{
		mx[n] = mtempx[n] + (9.0*fx[n] + 19.0*fx_1[n] - 5.0*fx_2[n] + fx_3[n])*h;
		my[n] = mtempy[n] + (9.0*fy[n] + 19.0*fy_1[n] - 5.0*fy_2[n] + fy_3[n])*h;
		mz[n] = mtempz[n] + (9.0*fz[n] + 19.0*fz_1[n] - 5.0*fz_2[n] + fz_3[n])*h;
		}


n += blockDim.x * gridDim.x;

}

}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_________Spherical Euler Method_________XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


__global__ void PC_Euler_step(int atoms, double *mx, double *my, double *mz, double *fx, double *fy, double *fz, double *mtempx, double *mtempy, double *mtempz, double dt, int *lbl)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;



while(n < atoms)
{

	double rx;
	double ry;
	double rz;
	double r;

	if(lbl[n]!=0)
	{
	mtempx[n] = mx[n];
	mtempy[n] = my[n];
	mtempz[n] = mz[n];

	mx[n] = mx[n] + dt*fx[n];
	my[n] = my[n] + dt*fy[n];
	mz[n] = mz[n] + dt*fz[n];  //m[n+1] intermediary values


	rx = mtempx[n] + mx[n];
	ry = mtempy[n] + my[n];
	rz = mtempz[n] + mz[n];

	r = drsqrt(rx*rx + ry*ry + rz*rz);


	mx[n] = rx*r;
	my[n] = ry*r;
	mz[n] = rz*r;
	}

	n += blockDim.x * gridDim.x;

}

}


//===================================================================================================================================================================

//This Subroutine uses the explicit form of the LLG equation


__global__ void PC_Spherical_Midpoint_Step(int atoms, double *mx, double *my, double *mz, double *fx, double *fy, double *fz, double *mtempx, double *mtempy, double *mtempz, double dt, int *lbl)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

double r;
	if(lbl[n]!=0)
	{
	mx[n] = mtempx[n] + dt*fx[n];
	my[n] = mtempy[n] + dt*fy[n];
	mz[n] = mtempz[n] + dt*fz[n];

	r = drsqrt(mx[n]*mx[n] + my[n]*my[n] + mz[n]*mz[n]);

	mx[n] = mx[n]*r;
	my[n] = my[n]*r;
	mz[n] = mz[n]*r;
	}

n += blockDim.x * gridDim.x;

}


}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____Heun's Method______XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

//Heun's method integral stochastically converges on the Stratonovich sense likewise for the explicit spherical midpoint scheme


__global__ void Heuns_First(int atoms, double *mx, double *my, double *mz, double *fx, double *fy, double *fz, double *fx_1, double *fy_1, double *fz_1, double *mtempx, double *mtempy, double *mtempz, double dt, int *lbl)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

	double r;

	if(lbl[n]!=0)
	{
	mtempx[n] = mx[n];
	mtempy[n] = my[n];
	mtempz[n] = mz[n];

	mx[n] = mx[n] + dt*fx[n];
	my[n] = my[n] + dt*fy[n];
	mz[n] = mz[n] + dt*fz[n];  //Everything must be projected into S2

	r = drsqrt(mx[n]*mx[n] + my[n]*my[n] + mz[n]*mz[n]);

	mx[n] = mx[n]*r;
	my[n] = my[n]*r;
	mz[n] = mz[n]*r;


	fx_1[n] = fx[n];
	fy_1[n] = fy[n];
	fz_1[n] = fz[n];
	}

	n += blockDim.x * gridDim.x;
	


}

}

__global__ void Heuns_Second(int atoms, double *mx, double *my, double *mz, double *fx, double *fy, double *fz, double *fx_1, double *fy_1, double *fz_1, double *mtempx, double *mtempy, double *mtempz, double dt, int *lbl)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

double r;

	if(lbl[n]!=0)
	{
	mx[n] = mtempx[n] + (dt/2.0)*(fx[n] + fx_1[n]);
	my[n] = mtempy[n] + (dt/2.0)*(fy[n] + fy_1[n]);
	mz[n] = mtempz[n] + (dt/2.0)*(fz[n] + fz_1[n]);

	r = drsqrt(mx[n]*mx[n] + my[n]*my[n] + mz[n]*mz[n]);

	mx[n] = mx[n]*r;
	my[n] = my[n]*r;
	mz[n] = mz[n]*r;
	}

n += blockDim.x * gridDim.x;

}

}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____Exchange Energy____XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


__global__ void potential_energy(int atoms, double *x_, double *y_, double *z_, double *epot_,int *viz_, int *n_viz_, double *mx_, double *my_, double *mz_, int *threshold, int *lbl, double Jl)
{
int n = threadIdx.x + blockIdx.x * blockDim.x; 

double temp_ = 0.0;


__shared__ double cache[256]; //Whole block for shared memory all-prefix sum

while(n < atoms)
{


int j1 = 0, w = 0;
int lbl1 = lbl[n];
int lbl2;
double mxx_ = 0.0;
double myy_ = 0.0;
double mzz_ = 0.0;
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];

epot_[blockIdx.x] = 0.0;
int w_ = n_viz_[n];
int offset = threshold[n];


			
			for(w = 0; w < w_; w++)
			{

			j1 = viz_[offset + w];
			lbl2 = lbl[j1];
			mxx_ = mx * mx_[j1];
			myy_ = my * my_[j1];
			mzz_ = mz * mz_[j1];
			

			if(lbl1 == lbl2) temp_ += mxx_ + myy_ + mzz_;
			else if(lbl1 != lbl2) temp_ += Jl*(mxx_ + myy_ + mzz_);
			}

	

n += blockDim.x * gridDim.x;
}



//Reduction/scan within a whole block using shared memory - code can be reused in a __device__ function, try it out :)


cache[threadIdx.x] = temp_;

__syncthreads();

int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x] += cache[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0) epot_[blockIdx.x] = cache[0];


}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____Zeeman Energy____XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

__global__ void potential_energy2(int atoms, double *x_, double *y_, double *z_, double *epot_, double *mx_, double *my_, double *mz_, double Bx_, double By_, double Bz_)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;	 

__shared__ double cache[256]; 


double temp_ = 0.0;


while(n < atoms)
{

double mBx_ = 0.0;
double mBy_ = 0.0;
double mBz_ = 0.0;
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];

epot_[blockIdx.x] = 0.0;

mBx_ = mx * Bx_;
mBy_ = my * By_;
mBz_ = mz * Bz_;

temp_ += mBx_ + mBy_ + mBz_;

n += blockDim.x * gridDim.x;

}



cache[threadIdx.x] = temp_;
__syncthreads();


int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x] += cache[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0) epot_[blockIdx.x] = cache[0];


}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____Dipolar Energy____XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



__global__ void potential_energy3(int atoms, double *x_, double *y_, double *z_, double *epot_, double *mx_, double *my_, double *mz_, int blockround)

{

int n = threadIdx.x + blockIdx.x * blockDim.x;	


double temp_ = 0.0;

__shared__ double cache[256];



while(n < atoms - 1)
{

int w = 0;
double mxx_ = 0.0;
double myy_ = 0.0;
double mzz_ = 0.0;
double mrjx_ = 0.0;
double mrjy_ = 0.0;
double mrjz_ = 0.0;
double mrix_ = 0.0;
double mriy_ = 0.0;
double mriz_ = 0.0;
double xx_ = 0.0;
double yy_ = 0.0;
double zz_ = 0.0;
double rij = 0.0;
double upp = 0.0;
double down = 0.0;
epot_[blockIdx.x] = 0.0;
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];
double x = x_[n];
double y = y_[n];
double z = z_[n];


	#pragma unroll
	for(w = n + 1; w < atoms; w++)
	{
		//(m_i, m_j)

		mxx_ = mx * mx_[w];
		myy_ = my * my_[w];
		mzz_ = mz * mz_[w];

		//rij (both vector and its norm)

		xx_ = x - x_[w];
		yy_ = y - y_[w];
		zz_ = z - z_[w];

		rij = drsqrt(xx_*xx_ + yy_*yy_ + zz_*zz_);

		//(m_i, r_ij) * (m_j, r_ij)

		xx_ = xx_*rij;
		yy_ = yy_*rij;
		zz_ = zz_*rij;		

		mrix_ = mx * xx_;
		mriy_ = my * yy_;
		mriz_ = mz * zz_;
		
		mrjx_ = mx_[w] * xx_;
		mrjy_ = my_[w] * yy_;
		mrjz_ = mz_[w] * zz_;

		upp = (mxx_ + myy_ + mzz_) - 3.0*(mrix_ + mriy_ + mriz_)*(mrjx_ + mrjy_ + mrjz_);
		down = (rij*rij*rij);
	
		temp_ += upp * down;
		

	}


n += blockDim.x * gridDim.x;

}


cache[threadIdx.x] = temp_;
__syncthreads();



int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x] += cache[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0) epot_[blockIdx.x] = cache[0];


}

//==========================================================Anisotropic Energy=====================================================*/

__global__ void potential_energy4(int atoms, double *mx_, double *my_, double *mz_, double ex, double ey, double ez, double lamb, double *epot_, double *ry, double *rz, double *rx, int curveflag)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

__shared__ double cache[256]; 


double temp_ = 0.0;


while(n < atoms)
{

double mex_ = 0.0;
double mey_ = 0.0;
double mez_ = 0.0;
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];

epot_[blockIdx.x] = 0.0;

if(curveflag == 1)
{
mex_ = mx * ex;
mey_ = my * ey;
mez_ = mz * ez;
}

if(curveflag == 2)
{

mex_ = 0.0;
mey_ = my*ry[n];
mez_ = mz*rz[n];
}

if(curveflag == 3)
{
mex_ = mx*rx[n];
mey_ = my*ry[n];
mez_ = mz*rz[n];
}

temp_ += lamb * (mex_ + mey_ + mez_) * (mex_ + mey_ + mez_);


n += blockDim.x * gridDim.x;

}



cache[threadIdx.x] = temp_;
__syncthreads();


int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x] += cache[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0) epot_[blockIdx.x] = cache[0];



}

//================================================Dzyaloshinskii-Moriya Energy=======================================================

__global__ void potential_energy5(int atoms, double *x, double *y, double *z, double *epot, int *viz, int *nviz, double *mx_, double *my_, double *mz_, int *threshold, double *ry, double *rz, int curveflag)
{

int n = threadIdx.x + blockDim.x * blockIdx.x;

double temp_ = 0.0;
__shared__ double cache[256];

while(n < atoms)
{

int j1, w;
double x_ = x[n];
double y_ = y[n];
double z_ = z[n];
double ry_ = ry[n];
double rz_ = rz[n];
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];
double deltax = 0.0;
double deltay = 0.0;
double deltaz = 0.0;
double xx_;
double yy_;
double zz_;
epot[blockIdx.x] = 0.0;
int w_ = nviz[n];
int offset = threshold[n];

			if(curveflag == 1)
			{

                        #pragma unroll
                        for(w = 0; w < w_; w++)
                        {

                        j1 = viz[offset + w];

                        if(z_ == z[j1])
                        {
                        xx_ = x_ - x[j1];
                        yy_ = y_ - y[j1];

                        deltax -= xx_ * mz_[j1];
                        deltay -= yy_ * mz_[j1];
                        deltaz += mx_[j1] * xx_ + my_[j1] * yy_;
                        }



			}

                        }
			
			else if(curveflag == 2)
			{

			for(w = 0; w < w_; w++)
			{

			j1 = viz[offset + w];

			xx_ = x_ - x[j1];
			yy_ = y_ - y[j1];
			zz_ = z_ - z[j1];

			deltax -= xx_*(mz_[j1]*rz_ + my_[j1]*ry_);
			deltay += ry_*(mx_[j1]*xx_ + mz_[j1]*zz_) - mz_[j1]*yy_*rz_;
			deltaz += rz_*(mx_[j1]*xx_ + my_[j1]*yy_) - my_[j1]*zz_*ry_;
			}

			}

                        temp_ = -1.0*(mx*deltax + my*deltay + mz*deltaz);

n += blockDim.x * gridDim.x;


}

cache[threadIdx.x] = temp_;
__syncthreads();


int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x] += cache[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0) epot[blockIdx.x] = cache[0];


}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX______Spin-Transfer Torque_________XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

__global__ void flumen_electrica(int atoms, double *fx, double *fy, double *fz, double *mx_, double *my_, double *mz_, double *x, double betavj, double *vj_aw, double ab1, double *x_min, double *x_max, double alpha, double Alph_g, double tcurr, int j_flag, int steps, double dt, int *nviz, int *viz, int *threshold, double pie)
{

int n = threadIdx.x + blockIdx.x * blockDim.x;

while(n < atoms)
{

//Spin polarized current terms - components of sucessive vector products among m and its "gradient" in the x-direction
double del_X;
double del_Y;
double del_Z;
double f_pX = 0.0;
double f_pY = 0.0;
double f_pZ = 0.0;
double s_pX = 0.0;
double s_pY = 0.0;
double s_pZ = 0.0;
double t_pX = 0.0;
double t_pY = 0.0;
double t_pZ = 0.0;
double vj;
int offset = threshold[n];
int n_viz = nviz[n];
//Registers
double mx = mx_[n];
double my = my_[n];
double mz = mz_[n];
double x_ = x[n];
double Mleftx, Mrightx, Mlefty, Mrighty, Mleftz, Mrightz;
int w = 0, j1;
//Current density velocity

if(j_flag == 0)
{
vj = vj_aw[n];
}
else
{
double vj = vj_aw[n]*sin(2.0*pie*tcurr*(double)steps*dt);
}



	//Points at both ends must be excluded
	if( (x[n] != *x_min) || (x[n] != *x_max) )
	{

		while(w < n_viz)
		{
			//Use the neighbor list to find the backward and frontward moments in the x-direction
			
			j1 = viz[w + offset];
		
			//X is either equal, less than or higher than the original position - and in this sense we define left and right!
			if(x[j1] < x_)
			{
			Mleftx = mx_[j1], Mlefty = my_[j1], Mleftz = mz_[j1];
			}
			else if(x[j1] > x_)
			{
			Mrightx = mx_[j1], Mrighty = my_[j1], Mrightz = mz_[j1];	
			}

		w++;
		}
		
		//Derivatives of each component towards the x direction
		//In the neighbor list, for a simple cubic cell, x-neighbors are those where r[n] = (xn, yn, zn) and r_neigh = (+- xneigh, yn, zn)

		del_X = (Mrightx - Mleftx)*0.5; 
		del_Y = (Mrighty - Mlefty)*0.5; 
		del_Z = (Mrightz - Mleftz)*0.5;
		
 	}

	if( (x[n] == *x_min) )
	{
		//Forward difference

	while(w < n_viz)
		{
			//Use the neighbor list to find the backward and frontward moments in the x-direction

			//Future optimization - Sort the neighbor list so that first elements are always x neighbors
			
			j1 = viz[w + offset];
		
			//X is either equal, less than or higher than the original position - and in this sense we define left and right!
		
			if(x[j1] > x_)
			{
			Mrightx = mx_[j1], Mrighty = my_[j1], Mrightz = mz_[j1];	
			}

		w++;
		}

		del_X = (Mrightx - mx);
		del_Y = (Mrighty - my);
		del_Z = (Mrightz - mz);

	
	}

	if( (x[n] == *x_max) )
	{

		//Backward difference

		while(w < n_viz)
		{
			//Use the neighbor list to find the backward and frontward moments in the x-direction
			
			j1 = viz[w + offset];
		
			//X is either equal, less than or higher than the original position - and in this sense we define left and right!
			if(x[j1] < x_)
			{
			Mleftx = mx_[j1], Mlefty = my_[j1], Mleftz = mz_[j1];
			}
			
		w++;
		}

	
		del_X = (mx - Mleftx);
		del_Y = (my - Mlefty);
		del_Z = (mz - Mleftz);


	}

	//Sucessive products

	//m X del	
	f_pX = my*del_Z - mz*del_Y;
	f_pY = mz*del_X - mx*del_Z;
	f_pZ = mx*del_Y - my*del_X;

	//m X (m X del)
	s_pX = my*f_pZ - mz*f_pY;
	s_pY = mz*f_pX - mx*f_pZ;
	s_pZ = mx*f_pY - my*f_pX;

	//m X (m X (m X del) )
	t_pX = my*s_pZ - mz*s_pY;
	t_pY = mz*s_pX - mx*s_pZ;
	t_pZ = mx*s_pY - my*s_pX;


	

	

	//Adding STT term to numerical F[t] approximation of precession
	fx[n] = fx[n] - Alph_g*(betavj*vj*f_pX + ab1*vj*s_pX + alpha*vj*t_pX);
	fy[n] = fy[n] - Alph_g*(betavj*vj*f_pY + ab1*vj*s_pY + alpha*vj*t_pY);
	fz[n] = fz[n] - Alph_g*(betavj*vj*f_pZ + ab1*vj*s_pY + alpha*vj*t_pZ);

n += blockDim.x * gridDim.x;	

}

}

//===================================================================================================================================

/*-----------GLOBAL VARIABLES, FUNCTIONS AND MAIN PROGRAM--------------*/

int m = 0; 

unsigned int natom, natom2; 
int sum2 = 0; //counter for dynamic allocation of neighbor list

int blocknatom = 0; //nearest integer higher than or equal to natom, which is also a multiple of block;

int lang; //Parameter file for Langevin dynamics

double *xran, *yran, *zran; //Thermal field arrays
double fluctu = 0.0; //Fluctuation-dissipation of thermal field

double mF = 0.0; //"cubic cell magnetization" i.e finite difference equivalent of localized magnetic moment


//comment in xyz input file
char cmt[100];
//variables for the simulation
double deltat;
double nran = 0.0;
int steps_dyn;
long semente;
double temp_dyn;
int n_write_trj, n_write_data, n_end_zeeman, n_end_curr;
double alpha, A, Ms, latt, exlen, D_J, Jcell;
double lamb, ex, ey, ez; //Direction and intensity of uniform and uniaxial anisotropic fields 
double D1,D2,*DM, *hDM; //Intensity of the Dzyaloshinskii-Moriya  field
double Bx, By, Bz; //External magnetic field for Zeeman interactions
int steps_restart, restart, integrator; //neighbor list parameters

double omega = 0.0; //Fundamental frequency of gyromagnetic precession.
double delta_tau; //Computational increment
double Alph_g; //Gilbert correction to Brown's LL equation

double Mag = 0.0;

double M = 0.0, Mx = 0.0, My = 0.0, Mz = 0.0;

double *ry, *rz, *rx;
int curveflag;

int *viz = NULL, *new_viz = NULL;

int *n_viz, *threshold;

int *dev_nviz, *dev_threshold;
int *dev_viz;

//Spin current density terms
double polaris;
double J_curr;
double betavj;
double etha;
double ab1;
double *vj_aw;
double *host_vj;
double *xmin, *xmax, *dxmin, *dxmax;

double *epot1, *epot2, *epot3, *epot4, *epot5;

double *dev_epot1, *dev_epot2, *dev_epot3, *dev_epot4, *dev_epot5;

double *Mblockx, *Mblocky, *Mblockz, *mm_;

double *d_Mblockx, *d_Mblocky, *d_Mblockz, *d_mm_;

//Skyrmion limit dimensions and atomic layer size
double L1, L2;

curandGenerator_t langerand;

double total_energy = 0.0,  total_M = 0.0, total_exchange = 0.0, total_dipolar = 0.0, total_zeeman = 0.0, total_anisotropy = 0.0, total_moryia = 0.0;
double Bxe, Bye, Bze;

int thread;
int block; //Can be set on runtime but this version fixes it for simplicity


double Jl;
int *label, *dlabel;
static double *x, *y, *z;

/* magnetic momenta */
static double *mx, *my, *mz, *mx_1, *my_1, *mz_1, *fx, *fy, *fz, *fx_1, *fy_1, *fz_1, *fx_2, *fy_2, *fz_2, *fx_3, *fy_3, *fz_3;

double *dev_x, *dev_y, *dev_z, *dev_mx, *dev_my, *dev_mz, *dev_fx, *dev_fy, *dev_fz, *dev_fx_1, *dev_fy_1, *dev_fz_1, *dev_fx_2, *dev_fy_2, *dev_fz_2, *dev_fx_3, *dev_fy_3, *dev_fz_3, *dev_mtempx, *dev_mtempy, *dev_mtempz;

double *Hdipx, *Hdipy, *Hdipz, *Hanisx, *Hanisy, *Hanisz, *Hexx, *Hexy, *Hexz, *HDMx, *HDMy, *HDMz;

/* Runge Kutta temporary Coefficients */
double *Cx, *Cy, *Cz, *mtempx, *mtempy, *mtempz;

/*Langevin Temperature */
double tempinst;

/* controlling simulation */
int 	steps_control, c_zeemanfield, c_write_trj, c_write_data, c_current;
int	c_steps_restart;

/* unit convertions if somehow needed */
/*
static double a2m=1.0e-10;
static double ev2j=1.602176e-19;
static double j2ev=6.24150974e+18;
static double uam=1.660540e-27;
static double kb=1.3800e-23; // kb = 1.3800D-23 J/K = 8.6200D-5 eV/K
*/

//timekeeping
clock_t start_t, end_t;
double cpu_time_used;

int n_passos;

int ZeePeriod, CurrPeriod;
double tx, ty, tz; //Frequencies for variable magnetic field
double tcurr; // variable current - one dimensional currents only for now

int size_coord, size_lbl;
int size_list;
int size_block;

/* READ XYZ INPUTFILE */
FILE	*inp_file;
/* READ SIMULATION SETTINGS */
FILE	*inp_set_file;
/* WRITE XYZ TRAJECTORY OUTPUT FILE */
FILE	*out_file;
/* WRITE COORDINATES TO RESTART SIMULATION (coord_restart.xyz) */
FILE	*wcoordrestart;
/* WRITE MOMENTS TO RESTART SIMULATION (mom.dat) */
FILE	*wvel;
/* READ XYZ FILE TO RESTART */
FILE	*rxyz_restart;
/* WRITE ENERGY AND TEMPERATURE */
FILE	*wenergy;
/* TMP DATA FILE -- http://www.gnu.org/s/hello/manual/libc/CPU-Time.html */
FILE	*tmp_file;
/* EXPECTED VALUES OF COMPONENTS OF M */
FILE 	*MFile;
/* LAST FRAME OF TRAJECTORY - READY FOR NEXT SIMULATION */
FILE	*NewTime;
/* TEST NUMERICAL INTEGRATION ROUTINES AGAINST DETERMINISTIC SOLUTION TO LLG EQUATION */
FILE	*test_file;

//Subroutines for the CPU: Not necessary neither efficient to have these on GPU


void w_trj(void);
void w_restart(void);
void w_energy(void);
void read_restart(void);
void readset(void);
void readxyz(void);
void alocate_vec(void);
void reallocable_list(void);
double aleatorio(void);
void w_trj_end(void);


/* ============================================================================

	MAIN PROGRAM
=============================================================================*/

int main(void)
{

curandCreateGenerator(&langerand, CURAND_RNG_PSEUDO_MTGP32);
    

start_t = clock();
		
printf
("=============================================================================\n");
printf(">>> Start Simulation\n");

//Input file opening

wenergy = fopen("data.dat","w");
MFile = fopen("Avg_mags.dat","w");

//File with initial data for simulation

tmp_file = fopen("tmp.dat", "w");

readset(); //Read control parameters from input file


fprintf(tmp_file, "Simulation Parameters\n");
fprintf(tmp_file, "=================================================================\n");
fprintf(tmp_file, "deltat=%le, steps_dyn = %d\n", deltat, steps_dyn);
fprintf(tmp_file, "semente = %ld, temp_dyn = %lf\n", semente, temp_dyn);
fprintf(tmp_file, "n_write_trj = %d, n_write_data= %d, firstneighbor = %le\n", n_write_trj, n_write_data, latt);
fprintf(tmp_file, "Saturation Magnetization Ms = %le\n", Ms);
fprintf(tmp_file, "Lattice parameter a = %le\n", latt);
fprintf(tmp_file, "Zeeman Field: Bx = %le, By = %le, Bz = %le\n", Bx, By, Bz);
fprintf(tmp_file, "Gilbert Damping parameter alpha = %lf\n", alpha);
fprintf(tmp_file, "steps_restart=%d, restart=%d\n", steps_restart, restart);
fprintf(tmp_file, "Integration method = %d, D/J = %le\n", integrator, D_J/(4.0f*piCPU));
fprintf(tmp_file, "Computational time step = %le\n", delta_tau);
fprintf(tmp_file, "Exchange length = %le\n", exlen);
fprintf(tmp_file, "Langevin = %d\n", lang);
fprintf(tmp_file, "Percentage of polarized angular moments in elecric current = %lf\n", polaris);
fprintf(tmp_file, "Current Density = %le\n", J_curr);
fprintf(tmp_file, "Adiabatic Torque Transfer parameter beta = %le\n", betavj);
fprintf(tmp_file, "Magnetocrystall. Axis = %lf %lf %lf\n", ex, ey, ez);
fprintf(tmp_file, "Surface Skyrmion interaction intensity for first and second layer sets respectively: %lf %lf\n", D1, D2);
fprintf(tmp_file, "Skyrmion limit range of interactions = %le %le\n", L1, L2);
fprintf(tmp_file, "====================================================================\n");




    /* Set seed */
curandSetPseudoRandomGeneratorSeed( langerand, semente);


//Condition for restart of simulation - this is for very long runs for those in third world countries with downtimes
//To run several simulations, e.g hysteresis curves, just use coord_last.xyz instead of the restart flag

if(restart == 1)
{
	

	//positions previously calculated
	read_restart();

	size_coord = sizeof(double)*blocknatom;
	size_lbl = sizeof(int)*natom;
	size_list = sizeof(int)*natom*natom;
 	size_block = block*sizeof(double);
		


	//Building the neighbor list
	reallocable_list();
}

if(restart == 0)
{	
	
	//read coordinates
	readxyz();
      	
	//building the neighbor lists
	reallocable_list();
}


//Energy of initial state


	cudaMemcpy(dev_x, x, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z, z, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mx, mx, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_my, my, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mz, mz, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(DM, hDM, size_coord, cudaMemcpyHostToDevice);
	cudaMemcpy(dlabel, label, size_lbl, cudaMemcpyHostToDevice);



if(curveflag == 2)
{
Polar_AxisYZ<<<block,thread>>>(natom, ry, rz, dev_y, dev_z); 
}

if(curveflag == 3)
{
SphereAxis<<<block,thread>>>(natom, ry, rz, dev_y, dev_z, rx, dev_x);
}

//Energy due to exchange interaction

potential_energy<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot1, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, dev_threshold, dlabel, Jl);


//Energy due to zeeman interaction

potential_energy2<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot2, dev_mx, dev_my, dev_mz, Bx, By, Bz);


//Energy due to dipolar interaction


potential_energy3<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot3, dev_mx, dev_my, dev_mz, blocknatom);

//Energy of magnetocrystalline anisotropy

potential_energy4<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, ex, ey, ez, lamb, dev_epot4, ry, rz, rx, curveflag);

//Energy of surface DM interaction

potential_energy5<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot5, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, dev_threshold, ry, rz, curveflag);

cudaMemcpy(epot1, dev_epot1, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot2, dev_epot2, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot3, dev_epot3, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot4, dev_epot4, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot5, dev_epot5, size_block, cudaMemcpyDeviceToHost);

			for( m = 0; m < block; m++)
			{
				total_exchange  += epot1[m];
				total_zeeman    += epot2[m];
				total_dipolar   += epot3[m];
				total_anisotropy+= epot4[m];
				total_moryia	+= epot5[m];
			}

total_exchange   = -1.0 * (total_exchange * Jcell * 1.0e+15)/2.0;
total_zeeman     = -1.0 * (Jcell * 1.0e+15 * D_J * total_zeeman);
total_dipolar    =  (0.25 * D_J/(piCPU)) * total_dipolar * Jcell * 1.0e+15;
total_anisotropy = -1.0*Jcell*total_anisotropy * 1.0e+15;
total_moryia	 = 0.5*Jcell * 1.0e+15 * ((D1+D2)/2.0) * total_moryia;

//Total energy in units of exchange coefficient Jcell

total_energy = total_exchange + total_dipolar + total_zeeman + total_anisotropy + total_moryia;

//First frame data

fprintf(tmp_file, "Zeeman Energy(fJ)      	%le\n", total_zeeman);
fprintf(tmp_file, "Exchange Energy(fJ)      	%le\n", total_exchange);
fprintf(tmp_file, "Dipolar Energy (fJ)      	%le\n", total_dipolar);
fprintf(tmp_file, "Anisotropy Energy (fJ)	%le\n", total_anisotropy);
fprintf(tmp_file, "DM Surface Energy (fJ)	%le\n", total_moryia);
fprintf(tmp_file, "Total Energy (fJ)      	%le\n", total_energy);

fflush(tmp_file);

//variables that control the dyn.
steps_control   = 0;
c_write_trj     = 0;
c_write_data    = 0;
c_steps_restart = 0;
c_zeemanfield 	= 0;

//Data from first frame
w_energy();
w_restart();
w_trj();

//Test the integration schemes against single spin-solution



/*
=================================================================================
MICROMAGNETIC SIMULATION WITH NO TEMPERATURE
=================================================================================
Lots of code in here could actually be reused for temperature, but this is streamlined for non-coder reading ;)
*/


//Starting simulation

	n_passos = 0;
	int i = 0;
	for(i = 0; i < steps_dyn; i++)
	{		
			

		
		if(c_write_data == n_write_data)
		{


//Energy due to exchange interaction

potential_energy<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot1, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, dev_threshold, dlabel, Jl);



//Energy due to zeeman interaction

potential_energy2<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot2, dev_mx, dev_my, dev_mz, Bx, By, Bz);


//Energy due to dipolar interaction


potential_energy3<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot3, dev_mx, dev_my, dev_mz, blocknatom);

//Energy of magnetocrystalline anisotropy

potential_energy4<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, ex, ey, ez, lamb, dev_epot4, ry, rz, rx, curveflag);

//Energy of surface DM interaction

potential_energy5<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot5, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, dev_threshold, ry, rz, curveflag);

cudaMemcpy(epot1, dev_epot1, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot2, dev_epot2, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot3, dev_epot3, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot4, dev_epot4, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot5, dev_epot5, size_block, cudaMemcpyDeviceToHost);

			for( m = 0; m < block; m++)
			{
				total_exchange  += epot1[m];
				total_zeeman    += epot2[m];
				total_dipolar   += epot3[m];
				total_anisotropy+= epot4[m];
				total_moryia	+= epot5[m];
			}

total_exchange   = -1.0 * (total_exchange * Jcell * 1.0e+15)/2.0;
total_zeeman     = -1.0 * (Jcell * 1.0e+15 * D_J * total_zeeman);
total_dipolar    =  (0.25 * D_J/(piCPU)) * total_dipolar * Jcell * 1.0e+15;
total_anisotropy = -1.0*Jcell*total_anisotropy * 1.0e+15;
total_moryia	 = 0.5*Jcell * 1.0e+15 * ((D1+D2)/2.0) * total_moryia;

//Total energy in units of exchange coefficient Jcell

total_energy = total_exchange + total_dipolar + total_zeeman + total_anisotropy + total_moryia;

			
			//Writing simulation energy data
			w_energy();
			c_write_data = 0;

			
		
			
		}
		
		//writing coordinates required for a restart

		if(c_steps_restart == steps_restart)
		{
			cudaMemcpy(mx, dev_mx, size_coord, cudaMemcpyDeviceToHost);
			cudaMemcpy(my, dev_my, size_coord, cudaMemcpyDeviceToHost);
			cudaMemcpy(mz, dev_mz, size_coord, cudaMemcpyDeviceToHost);
			
			w_restart();
			c_steps_restart = 0;

		}

//Turn off the constant applied magnetic field B

		if(c_zeemanfield == n_end_zeeman)
		{
		Bxe = 0.0, Bye = 0.0, Bze = 0.0;
		ZeePeriod = 0;
		c_zeemanfield = 0;
		}

		if(ZeePeriod == 1)
		{

		Bx = Bxe*cos(2.0*piCPU*tx*steps_control*deltat);
		By = Bye*sin(2.0*piCPU*ty*steps_control*deltat);
		Bz = Bze*sin(2.0*piCPU*tz*steps_control*deltat);

		}
		else if (ZeePeriod == 0)
		{
	
		Bx = Bxe, By = Bye, Bz = Bze;
		

		}

//Turn off the applied current density J

		if(c_current == n_end_curr)
		{
			for(int yi = 0; yi < natom; yi++)
			{
			host_vj[yi] = 0.0;
			}
			CurrPeriod = 0;
	
			cudaMemcpy(vj_aw, host_vj, size_coord, cudaMemcpyHostToDevice);
			c_current = 0;
		}

		//Writing the trajectory of the system
		if(c_write_trj == n_write_trj)
		{
		


			cudaMemcpy(mx, dev_mx, size_coord, cudaMemcpyDeviceToHost);
			cudaMemcpy(my, dev_my, size_coord, cudaMemcpyDeviceToHost);
			cudaMemcpy(mz, dev_mz, size_coord, cudaMemcpyDeviceToHost);
			

			
			c_write_trj = 0;
			w_trj();
		}


//Deterministic dynamics

	if(lang == 0)
	{

		//Runge-Kutta steps


		if(integrator == 0)
		{

		if(i < 3)
		{

		
		// Runge-Kutta Coefficients


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);


		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		if(i == 0)
		{		
		
		//C1 quarter-step
	

		cudaMemcpy(fx_3, dev_fx, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fy_3, dev_fy, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fz_3, dev_fz, size_coord, cudaMemcpyDeviceToHost);
		
		

		for( m = 0; m < natom; m++)
		{

			Cx[m] = fx_3[m];
			Cy[m] = fy_3[m];
			Cz[m] = fz_3[m];

			mx_1[m] = mx[m] + 0.5*delta_tau*Cx[m];
			my_1[m] = my[m] + 0.5*delta_tau*Cy[m];
			mz_1[m] = mz[m] + 0.5*delta_tau*Cz[m];


			mtempx[m] = mx[m] + ((delta_tau)/6.0)*Cx[m];
			mtempy[m] = my[m] + ((delta_tau)/6.0)*Cy[m];
			mtempz[m] = mz[m] + ((delta_tau)/6.0)*Cz[m];


		}

		}

		if(i == 1)
		{
		//C1 quarter-step
	

		
		cudaMemcpy(fx_2, dev_fx, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fy_2, dev_fy, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fz_2, dev_fz, size_coord, cudaMemcpyDeviceToHost);
		

		for( m = 0; m < natom; m++)
		{
					
			
			Cx[m] = fx_2[m];
			Cy[m] = fy_2[m];
			Cz[m] = fz_2[m];

			mx_1[m] = mx[m] + 0.5*delta_tau*Cx[m];
			my_1[m] = my[m] + 0.5*delta_tau*Cy[m];
			mz_1[m] = mz[m] + 0.5*delta_tau*Cz[m];

			mtempx[m] = mx[m] + ((delta_tau)/6.0)*Cx[m];
			mtempy[m] = my[m] + ((delta_tau)/6.0)*Cy[m];
			mtempz[m] = mz[m] + ((delta_tau)/6.0)*Cz[m];
		}

		}

		if(i == 2)
		{		
		//C1 quarter-step
	
		
		cudaMemcpy(fx_1, dev_fx, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fy_1, dev_fy, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fz_1, dev_fz, size_coord, cudaMemcpyDeviceToHost);
		
		
		for( m = 0; m < natom; m++)
		{

		
			Cx[m] = fx_1[m];
			Cy[m] = fy_1[m];
			Cz[m] = fz_1[m];

			mx_1[m] = mx[m] + 0.5*delta_tau*Cx[m];
			my_1[m] = my[m] + 0.5*delta_tau*Cy[m];
			mz_1[m] = mz[m] + 0.5*delta_tau*Cz[m];

			mtempx[m] = mx[m] + ((delta_tau)/6.0)*Cx[m];
			mtempy[m] = my[m] + ((delta_tau)/6.0)*Cy[m];
			mtempz[m] = mz[m] + ((delta_tau)/6.0)*Cz[m];
		}


		}



		//C2 quarter_step

		
		
		cudaMemcpy(dev_mx, mx_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_my, my_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mz, mz_1, size_coord, cudaMemcpyHostToDevice);
		
//renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz);	//Allowing system to actually blow up if it goes too far from equilibrium
//remember renormalizing m with this integrator is a non-linear modification of the original eq.



		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);
	

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		cudaMemcpy(fx, dev_fx, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fy, dev_fy, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fz, dev_fz, size_coord, cudaMemcpyDeviceToHost);

		
		for( m = 0; m < natom; m++)
		{

			Cx[m] = fx[m];
			Cy[m] = fy[m];
			Cz[m] = fz[m];
			
			mx_1[m] = mx[m] + 0.5*delta_tau*Cx[m];
			my_1[m] = my[m] + 0.5*delta_tau*Cy[m];
			mz_1[m] = mz[m] + 0.5*delta_tau*Cz[m];

		
			mtempx[m] +=  ((delta_tau)/3.0)*Cx[m];
			mtempy[m] +=  ((delta_tau)/3.0)*Cy[m];
			mtempz[m] +=  ((delta_tau)/3.0)*Cz[m];

		}


		
		//C3 quarter_step

		cudaMemcpy(dev_mx, mx_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_my, my_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mz, mz_1, size_coord, cudaMemcpyHostToDevice);
				
//renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz);


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);
		 

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		cudaMemcpy(fx, dev_fx, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fy, dev_fy, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fz, dev_fz, size_coord, cudaMemcpyDeviceToHost);


		
		for( m = 0; m < natom; m++)
		{

			Cx[m] = fx[m];
			Cy[m] = fy[m];
			Cz[m] = fz[m];			

			mx_1[m] = mx[m] + delta_tau*Cx[m];
			my_1[m] = my[m] + delta_tau*Cy[m];
			mz_1[m] = mz[m] + delta_tau*Cz[m];

		
			mtempx[m] +=  ((delta_tau)/3.0)*Cx[m];
			mtempy[m] +=  ((delta_tau)/3.0)*Cy[m];
			mtempz[m] +=  ((delta_tau)/3.0)*Cz[m];
		}	

		//C4 quarter_step

	
		cudaMemcpy(dev_mx, mx_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_my, my_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mz, mz_1, size_coord, cudaMemcpyHostToDevice);

		//renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz);				



		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	


		cudaMemcpy(fx, dev_fx, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fy, dev_fy, size_coord, cudaMemcpyDeviceToHost);
		cudaMemcpy(fz, dev_fz, size_coord, cudaMemcpyDeviceToHost);



		
		for( m = 0; m < natom; m++)
		{
			Cx[m] = fx[m];
			Cy[m] = fy[m];
			Cz[m] = fz[m];
			
			mtempx[m] +=  ((delta_tau)/6.0)*Cx[m];
			mtempy[m] +=  ((delta_tau)/6.0)*Cy[m];
			mtempz[m] +=  ((delta_tau)/6.0)*Cz[m];
		
			mx[m] = mtempx[m];
			my[m] = mtempy[m];
			mz[m] = mtempz[m];
		}	

		cudaMemcpy(dev_mx, mx, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_my, my, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mz, mz, size_coord, cudaMemcpyHostToDevice);
		
		//renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz);
		
		}//End of Runge-Kutta integration

		//Predictor-Corrector integrator

		if(i >= 3)
		{


		if(i == 3)
		{



		cudaMemcpy(dev_fx_2, fx_2, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fy_2, fy_2, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fz_2, fz_2, size_coord, cudaMemcpyHostToDevice);
		
		cudaMemcpy(dev_fx_1, fx_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fy_1, fy_1, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fz_1, fz_1, size_coord, cudaMemcpyHostToDevice);
		

		cudaMemcpy(dev_fx_3, fx_3, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fy_3, fy_3, size_coord, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fz_3, fz_3, size_coord, cudaMemcpyHostToDevice);

free(mx_1);
free(my_1);
free(mz_1);
free(Cx);
free(Cy);
free(Cz);	
cudaFreeHost(mtempx);
cudaFreeHost(mtempy);
cudaFreeHost(mtempz);
cudaFreeHost(fx_1);
cudaFreeHost(fx_2);
cudaFreeHost(fx_3);
cudaFreeHost(fy_1);
cudaFreeHost(fy_2);
cudaFreeHost(fy_3);
cudaFreeHost(fz_1);
cudaFreeHost(fz_2);
cudaFreeHost(fz_3);


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);
	
		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	


		}
		
		
		PC_Adams_Moulton_step<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_mtempx, dev_mtempy, dev_mtempz, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_fx_2, dev_fy_2, dev_fz_2, dev_fx_3, dev_fy_3, dev_fz_3, delta_tau, i, dlabel);

		//renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz);

		TimeSkip<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_fx_2, dev_fy_2, dev_fz_2, dev_fx_3, dev_fy_3, dev_fz_3);		
					


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	


		PC_Adams_Bashforth_step<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_mtempx, dev_mtempy, dev_mtempz, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_fx_2, dev_fy_2, dev_fz_2, dev_fx_3, dev_fy_3, dev_fz_3, delta_tau, dlabel);

		//renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz);


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);
	
		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	


		}

		}// End of 4th order integrators


//Spherical Midpoint Rule PC steps

	if(integrator == 1)
	{

	//Euler Method - approximates the derivative with simple slope approximation in each dimension - could be improved to exponential integration or even machine learning later!


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);


		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	
	

		PC_Euler_step<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau, dlabel);



		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		//Spherical Midpoint Rule - Symplectic, discrete precession over the phase space S2

		PC_Spherical_Midpoint_Step<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau, dlabel);

	}//End of Predicting-Correcting integration


//PC Heun's Method Step (Euler + Trapezoidal, equivalent to 2nd order Runge-Kutta, converges the stochastic LLG integral in Stratonovich sense)

	if(integrator == 2)
	{


		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		Heuns_First<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau, dlabel);



		precessional3D<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		Heuns_Second<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau, dlabel);


	}

	}//End of deterministic dynamics

//Langevin dynamics - Verify Stratonovich convergence!

if(lang == 1)
	{

	if( (i==0)&&(integrator!=1)&&(integrator!=2) )
	{
	printf("\nPlease choose only integrators with order less or equal to 2 for stochastical dynamics!\n");
	printf("Program Exitting========================================================================>\n");
	exit(1);
	}


//Filling random number arrays for Thermal Field

curandGenerateNormalDouble(langerand, xran, natom, 0.0, 1.0);
curandGenerateNormalDouble(langerand, yran, natom, 0.0, 1.0);
curandGenerateNormalDouble(langerand, zran, natom, 0.0, 1.0);



//Spherical Midpoint Rule PC steps

	if(integrator == 1)
	{

	//Euler Method - approximates the derivative with simple slope approximation in each dimension - could be improved to gradient descent or machine learning later!


		precessional3DRand<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, fluctu, xran, yran, zran, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	


		PC_Euler_step<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau,dlabel);


		precessional3DRand<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, fluctu, xran, yran, zran, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		//Spherical Midpoint Rule - Symplectic, discrete precession over the phase space S2

		PC_Spherical_Midpoint_Step<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau,dlabel);

	}//End of Predicting-Correcting integration


//PC Heun's Method Step (Euler + Trapezoidal, equivalent to 2nd order Runge-Kutta, converges the stochastic LLG integral in Stratonovich sense)

	if(integrator == 2)
	{


		precessional3DRand<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, fluctu, xran, yran, zran, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		Heuns_First<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau,dlabel);

		precessional3DRand<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, D_J, alpha, dev_fx, dev_fy, dev_fz, Bx, By, Bz, Alph_g, dev_threshold,piCPU,blocknatom, fluctu, xran, yran, zran, ex, ey, ez, lamb, DM, dlabel, Jl, Hdipx, Hdipy, Hdipz, Hexx, Hexy, Hexz, HDMx, HDMy, HDMz, Hanisx, Hanisy, Hanisz, curveflag, ry, rz, rx);

		flumen_electrica<<<block,thread>>>(natom, dev_fx, dev_fy, dev_fz, dev_mx, dev_my, dev_mz, dev_x, betavj, vj_aw, ab1, dxmin, dxmax, alpha, Alph_g, tcurr, CurrPeriod, steps_control, deltat, dev_nviz, dev_viz, dev_threshold, piCPU);	

		Heuns_Second<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, dev_fx, dev_fy, dev_fz, dev_fx_1, dev_fy_1, dev_fz_1, dev_mtempx, dev_mtempy, dev_mtempz, delta_tau,dlabel);


	}



	}//End of Langevin dynamics





/*
if( (steps_control % 10 == 0) && (steps_control > 0) )
{
magnetization();
if(abs(total_M - 1.0) > 1.0e-7) renormalize<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz); //Prevents divergence but, if this is active, nonlinear errors!!
											      //Should never be needed, if system blows up see timestep or drsqrt()

}
*/
		/* counting steps */
		
		c_write_data += 1;
		c_steps_restart += 1;
		c_write_trj += 1;
		n_passos += 1;
		steps_control += 1;
		c_zeemanfield += 1;
		c_current += 1;		

	}//End of time loop - dynamics is contained here!

/*000000000000000000000000000000000000000000000000000000000000000000000000000000
END OF SIMULATION FOR THE CASE OF NO TEMPERATURE
00000000000000000000000000000000000000000000000000000000000000000000000000000000*/

//data from last frame
//Energy due to exchange interaction

potential_energy<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot1, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, dev_threshold, dlabel, Jl);



//Energy due to zeeman interaction

potential_energy2<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot2, dev_mx, dev_my, dev_mz, Bx, By, Bz);


//Energy due to dipolar interaction


potential_energy3<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot3, dev_mx, dev_my, dev_mz, blocknatom);

//Energy of magnetocrystalline anisotropy

potential_energy4<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, ex, ey, ez, lamb, dev_epot4, ry, rz, rx, curveflag);

//Energy of surface DM interaction

potential_energy5<<<block,thread>>>(natom, dev_x, dev_y, dev_z, dev_epot5, dev_viz, dev_nviz, dev_mx, dev_my, dev_mz, dev_threshold, ry, rz, curveflag);

cudaMemcpy(epot1, dev_epot1, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot2, dev_epot2, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot3, dev_epot3, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot4, dev_epot4, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(epot5, dev_epot5, size_block, cudaMemcpyDeviceToHost);

			for( m = 0; m < block; m++)
			{
				total_exchange  += epot1[m];
				total_zeeman    += epot2[m];
				total_dipolar   += epot3[m];
				total_anisotropy+= epot4[m];
				total_moryia	+= epot5[m];
			}

total_exchange   = -1.0 * (total_exchange * Jcell * 1.0e+15)/2.0;
total_zeeman     = -1.0 * (Jcell * 1.0e+15 * D_J * total_zeeman);
total_dipolar    =  (0.25 * D_J/(piCPU)) * total_dipolar * Jcell * 1.0e+15;
total_anisotropy = -1.0*Jcell*total_anisotropy * 1.0e+15;
total_moryia	 = 0.5*Jcell * 1.0e+15 * ((D1+D2)/2.0) * total_moryia;

//Total energy in units of exchange coefficient Jcell

total_energy = total_exchange + total_dipolar + total_zeeman + total_anisotropy + total_moryia;

w_energy();
w_restart();
w_trj_end();

//close files
fclose (inp_file);
fclose (out_file);
fclose (wenergy);
fclose (tmp_file);
fclose (MFile);
fclose (NewTime);

free(label);
cudaFreeHost(x);
cudaFreeHost(y);
cudaFreeHost(z);
cudaFreeHost(mx);
cudaFreeHost(my);
cudaFreeHost(mz);
if(integrator==0)
{
cudaFreeHost(fx);
cudaFreeHost(fy);
cudaFreeHost(fz);
}

free(viz);
cudaFreeHost(n_viz);
cudaFreeHost(threshold);
cudaFreeHost(epot1);
cudaFreeHost(epot2);
cudaFreeHost(epot3);
cudaFreeHost(epot4);
cudaFreeHost(epot5);
cudaFreeHost(Mblockx);
cudaFreeHost(Mblocky);
cudaFreeHost(Mblockz);
cudaFreeHost(mm_);



cudaFree(dev_mtempx);
cudaFree(dev_mtempy);
cudaFree(dev_mtempz);
cudaFree(dev_fx);
cudaFree(dev_fy);
cudaFree(dev_fz);


if(integrator==0)
{
cudaFree(dev_fx_1);
cudaFree(dev_fy_1);
cudaFree(dev_fz_1);

cudaFree(dev_fx_2);
cudaFree(dev_fy_2);
cudaFree(dev_fz_2);

cudaFree(dev_fx_3);
cudaFree(dev_fy_3);
cudaFree(dev_fz_3);
}

if(integrator==2)
{

cudaFree(dev_fx_1);
cudaFree(dev_fy_1);
cudaFree(dev_fz_1);


}

cudaFree(Hdipx);
cudaFree(Hdipy);
cudaFree(Hdipz);
cudaFree(Hexx);
cudaFree(Hexy);
cudaFree(Hexz);
cudaFree(Hanisx);
cudaFree(Hanisy);
cudaFree(Hanisz);
cudaFree(HDMx);
cudaFree(HDMy);
cudaFree(HDMz);

cudaFree(dlabel);
cudaFree(DM);
cudaFreeHost(hDM);
cudaFree(dev_nviz);
cudaFree(dev_threshold);
cudaFree(dev_x);
cudaFree(dev_y);
cudaFree(dev_z);
cudaFree(dev_viz);
cudaFree(dev_mx);
cudaFree(dev_my);
cudaFree(dev_mz);
cudaFree(dev_epot1);
cudaFree(dev_epot2);
cudaFree(dev_epot3);
cudaFree(dev_epot4);
cudaFree(dev_epot5);
cudaFree(d_Mblockx);
cudaFree(d_Mblocky);
cudaFree(d_Mblockz);
cudaFree(d_mm_);
cudaFree(dxmin);
cudaFree(dxmax);
cudaFree(ry);
cudaFree(rz);
cudaFree(rx);
cudaFree(vj_aw);
cudaFreeHost(host_vj);


if(lang == 1)
{

cudaFree(xran);
cudaFree(yran);
cudaFree(zran);

}

curandDestroyGenerator(langerand);
end_t = clock();
cpu_time_used = ((double) (end_t - start_t)) / CLOCKS_PER_SEC;

printf("\n\n>>> Ta-daaa! Simulation finished at:\n");
printf(">>> Total Time Elapsed (s) %13.10lf\n", cpu_time_used);
printf(">>> Real Time Elapsed (ns) %13.10lf\n", (steps_control*delta_tau*1.0e+9)/omega);
printf("=========================================================================\n");




/*>>>>>>>>>>>>>>>>>> Ending Main Program>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/

	return 0;

}


/*----------------------------------------------------------------------*/
/*-----------------Writing trajectory file------------------------------*/

void w_trj(void)
{
int i;

fprintf(out_file, "%lu\n",natom);
fprintf(out_file, "%d\n", steps_control);

for(i = 0; i < natom; i++)
	{

	if(label[i] == 1)fprintf(out_file, "H  %16.8f %16.8f %16.8f atom_vector %16.8f %16.8f %16.8f \n", x[i], y[i], z[i], mx[i], my[i], mz[i]);
	else		 fprintf(out_file, "Ag %16.8f %16.8f %16.8f atom_vector %16.8f %16.8f %16.8f \n", x[i], y[i], z[i], mx[i], my[i], mz[i]);
	}



return;
}		

void w_trj_end(void) //This file is separate from the rest and contains the last frame - makes it much easier to script successive runs such as hysteresis
{
int i;


fprintf(NewTime, "%lu\n",natom);
fprintf(NewTime, "%d\n", steps_control);

for(i = 0; i < natom; i++)
	{


	if(label[i] == 1)fprintf(NewTime, "H  %16.15lf %16.15lf %16.15lf atom_vector %16.15lf %16.15lf %16.15lf \n", x[i], y[i], z[i], mx[i], my[i], mz[i]);
	else		 fprintf(NewTime, "Ag %16.15lf %16.15lf %16.15lf atom_vector %16.15lf %16.15lf %16.15lf \n", x[i], y[i], z[i], mx[i], my[i], mz[i]);
	}


return;
}		


/*-----------------------------------------------------------------------------*/
/*---------------------Restarting coordinates----------------------------------*/

void w_restart(void)
{

int l;

wcoordrestart = fopen("coord_restart.xyz","w");

fprintf(wcoordrestart, "%lu\n",natom);
fprintf(wcoordrestart, "%s\n",cmt);

for(l=0; l < natom; l++)
	{
		fprintf(wcoordrestart, "%-6u %16.15lf %16.15lf %16.15lf atom_vector %16.15lf %16.15lf %16.15lf\n", label[l], x[l], y[l], z[l], mx[l], my[l], mz[l]);
	}

fclose(wcoordrestart);

return;
}

/*-----------------------------------------------------------------------------*/
/*------------------------Writing simulation data------------------------------*/

void w_energy(void)
{




if(steps_control == 0)
{
	fprintf(wenergy,"#===========================++++AD_MAGNUS CUDA Version 3.0====++++++========================================================================\n");
	fprintf(wenergy,"#=================================== DATA OF SIMULATION =====================================================================================\n");
	fprintf(wenergy,"#==============================================================================================================--------------------------\n");
	fprintf(wenergy,"#   Step	Real_time            E_Zee(fJ)   		 E_Anis(fJ)		E_DM(fJ)	   	E_Dip(fJ)         	E_Ex(fJ)            E_Total(fJ)          Average Mag. \n");

	fprintf(MFile,"#=========================== AD_MAGNUS CUDA Version 3.0====++++++========================================================================\n");
	fprintf(MFile,"#=================================== EXPECTED VALUES OF THE COMPONENTS OF MAGNETIZATION VECTOR FIELD M=====================================\n");
	fprintf(MFile,"#==============================================================================================================----------------------------\n");
	fprintf(MFile,"#Step			Real_time   			Mx 				My		      	Mz			  \n");
}



magnetum_opus<<<block,thread>>>(natom, dev_mx, dev_my, dev_mz, d_Mblockx, d_Mblocky, d_Mblockz, d_mm_);

cudaMemcpy(Mblockx, d_Mblockx, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(Mblocky, d_Mblocky, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(Mblockz, d_Mblockz, size_block, cudaMemcpyDeviceToHost);
cudaMemcpy(mm_, d_mm_, size_block, cudaMemcpyDeviceToHost);

Mx = 0.0;
My = 0.0;
Mz = 0.0;
total_M = 0.0;

for( m = 0; m < block; m++)
			{
				total_M += mm_[m];
				Mx += Mblockx[m];
				My += Mblocky[m];
				Mz += Mblockz[m];
			}

Mx /= natom;
My /= natom;
Mz /= natom;
total_M /= natom;

fprintf(wenergy, "%8d %14.15e %14.15e %14.15e %14.15e %14.15e %14.15e %14.15e %14.15e\n", steps_control, steps_control*deltat, total_zeeman, total_anisotropy, total_moryia, total_dipolar, total_exchange, total_energy, total_M);
fprintf(MFile, "%8d \t \t %14.15e \t  %14.15e  \t %14.15e \t %14.15e\n", steps_control, steps_control*deltat, Mx, My, Mz);


fflush(wenergy);
fflush(MFile);

if(steps_control < steps_dyn - n_write_data)
{
total_zeeman = 0.0;
total_dipolar = 0.0;
total_exchange = 0.0;
total_anisotropy = 0.0;
total_moryia = 0.0;
total_energy = 0.0;
Mx = 0.0, My = 0.0, Mz = 0.0;
}

return;
}

/*-----------------------------------------------------------------------------*/
/*-----------------------Reading restart coordinates---------------------------*/

void read_restart(void)
{

int w = 0;
rxyz_restart = fopen("coord_restart.dat", "r");

fscanf(rxyz_restart, "%lu\n", &natom);
fscanf(rxyz_restart, "%s\n", &cmt);


alocate_vec();

for(w = 0; w<natom; w++)
	{

	fscanf(inp_file,"%d %lf %lf %lf %*s %lf %lf %lf\n", &label[w], &x[w], &y[w], &z[w], &mx[w], &my[w], &mz[w]);

	}


fclose(rxyz_restart);

//Obs: Assuming that m has already been previously projected into S2

return;
}

/*-----------------------------------------------------------------------------*/
/*-----------------------Reading control parameters----------------------------*/

void readset(void)
{
	if((inp_set_file=fopen("settings.inp","r")) == NULL) 
	{
		printf("The control parameter file could not be opened; please don't panic, see if the file is really there, and try again.");
		exit(1);
	}

//	[1] time step for integrating the movement equations [double]
fscanf(inp_set_file, "%le\n", &deltat);
//	[2] number of steps for the dynamics [int]
fscanf(inp_set_file, "%d\n", &steps_dyn);
//	[3] seed for the distribution of moments [int] 
fscanf(inp_set_file, "%ld\n", &semente); 
//	[4] Temperature for stochastical Langevin dynamics[double]
fscanf(inp_set_file, "%lf\n", &temp_dyn);
//	[5] number of steps between each write to the trajectory files [int]
fscanf(inp_set_file, "%d\n", &n_write_trj);
//	[6] number of steps between each writing of actual simulation data [int]
fscanf(inp_set_file, "%d\n", &n_write_data);
//	[7] Gilbert Damping constant alpha
fscanf(inp_set_file, "%lf\n", &alpha);
//	[8] Micromagnetic exchange tensor (constant if diagonal!) A
fscanf(inp_set_file,"%le\n", &A);
//	[9] External field (choose amplitude if variable in time field)
fscanf(inp_set_file, "%le %le %le\n", &Bxe, &Bye, &Bze);
//	[10] Lattice parameter of finite difference node (only SC lattices for now!)
fscanf(inp_set_file, "%le\n", &latt);
//	[11] Saturation Magnetization
fscanf(inp_set_file, "%le\n", &Ms); 
//	[12] steps between writings of force, acceleration, velocity and coordinate [int]
fscanf(inp_set_file, "%d\n", &steps_restart);
//	[13] choose 0 for a fresh started simulation, choose 1 to restart a simulation
fscanf(inp_set_file, "%d\n", &restart);
/*	[14] LIST OF INTEGRATORS IMPLEMENTED: 
0 - 4th order Runge-Kutta starter + Adams-Moulton-Bashfort Predictor-corrector;
1 - Spherical Euler Predictor-Corrector (explicit Spherical);
2 - Heun's Method (Euler Predictor + Trapezoidal Corrector) - Equivalent to 2nd order Runge-Kutta;
*/
fscanf(inp_set_file, "%d\n", &integrator);
//	[15] number of steps to turn off the Zeeman field B (field start at intermediary dynamics not applied yet!) relaxation = ([2] - [15])
fscanf(inp_set_file, "%d\n", &n_end_zeeman);
//	[16] dimensionality of model (deprecated in this version, use strong negative anisotropy if need a XY model)
fscanf(inp_set_file, "%d\n", &dimension);
//	[17] block and thread sizes; fixed for this version but easy to implement @runtime via extern[]
fscanf(inp_set_file, "%d %d\n", &block, &thread);
//	[18] Choose whether or not field is periodic in time (0 for no, 1 for yes)
fscanf(inp_set_file, "%d\n", &ZeePeriod);
// 	[19] Choose frequency of period field
fscanf(inp_set_file, "%lf %lf %lf\n", &tx, &ty, &tz);
//	[20] Choose 0 for deterministic dynamics at 0 Kelvin and 1 for Langevin dynamics with stochastical equations!
fscanf(inp_set_file, "%d\n", &lang);
//	[21] Anisotropic field direction (uniform and uniaxial)
fscanf(inp_set_file, "%lf %lf %lf\n", &ex, &ey, &ez);
//	[22] Anisotropic field intensity
fscanf(inp_set_file, "%le\n", &lamb);
//	[23] Dzyaloshinskii-Moriya Superficial field intensity (only hedgehogs in this version) - Choose up to two values to be split up to two layers
fscanf(inp_set_file, "%le %le\n", &D1, &D2);
//	[24] Layer separation and layer limit of Dzyaloshinskii-Moriya interactions (in meters)
fscanf(inp_set_file, "%le %le\n", &L1, &L2);
//	[25] J' / Jcell ratio of pointlike impurity exchange interactions
fscanf(inp_set_file, "%lf\n", &Jl);
//	[26] Density of current generating the Spin-Transfer Torque (STT) of the system
fscanf(inp_set_file, "%le\n", &J_curr);
//	[27] Choose whether or not current density is periodic in time (0 for not, 1 for yes)
fscanf(inp_set_file, "%d\n", &CurrPeriod);
//	[28] Choose frequency of periodic current (only one direction for now) in s^-1 
fscanf(inp_set_file, "%lf\n", &tcurr);
//	[29] Adiabatic Torque Transfer parameter Beta
fscanf(inp_set_file, "%lf\n", &betavj);
//	[30] Number of steps to turn off the current density
fscanf(inp_set_file, "%d\n", &n_end_curr);
//	[31] Mean percentage of polarization of angular moments in the current density - ( P )
fscanf(inp_set_file, "%lf\n", &polaris);
//	[32] Flag for curvature of structure - 1 for plane perpendicular to z, 2 for cylinder rotationally symmetric to x, 3 for spherical shell
fscanf(inp_set_file, "%d\n", &curveflag);

//Current density polarization interaction parameter

etha = (muB * polaris)/(e_charge * Ms * (betavj*betavj + 1.0));

//Mixed parameters  parameter

ab1 = alpha*betavj + 1.0;

//acquire exchange length:

exlen = sqrt((2.0 * A)/(mu * Ms * Ms));

//get 4*Pi*Dipolar/Exchange term from it:

D_J = pow((latt/exlen),2.0);


Jcell = 2.0*latt*A; //First order equivalent Hamiltonian - by expanding (\nabla m)^2

//Dimensionless Zeeman field

Bxe = Bxe/(mu*Ms);
Bye = Bye/(mu*Ms);
Bze = Bze/(mu*Ms);

//Adimensionalization of time

omega = (1.0/D_J) * mu * Ms * gyro;

//Dimensionless peak frequencies of external fields (current densities or magnetic fields)

tx /= omega;
ty /= omega;
tz /= omega;
tcurr /= omega;


//Magnetization of micromagnetic cell if relevant

mF = Ms * pow(latt,3);

// \hat{n}

double eR = sqrt(ex*ex + ey*ey + ez*ez);

if(eR != 0.0)
{
ex /= eR;
ey /= eR;
ex /= eR;
}

else
{
ex = 0.0;
ey = 0.0;
ez = 0.0;
}

//Dimensionless anisotropic field intensity

lamb *= (D_J / (mu*Ms*Ms) );

delta_tau = omega*deltat;



Alph_g = 1.0/(1.0 + pow(alpha,2.0)); //Gilbert damping constant

//Thermal field (fluctuation-dissipation theorem) intensity already made dimensionless

fluctu = sqrt( (2.0*Alph_g*kB*temp_dyn) / (mu*Ms*Ms*latt*latt*latt) );

printf("%.15le - mu \n%.15le - D_J \n%.15le - latt \n%.15le - exlen \n%.15le - delta_tau \n%.15le - omega \n%.15le - gyro\n%.15le - alpha\n%.15le - lamb\n%.15le %.15le - D1 D2\n %.15le - fluctu\n", mu, D_J, latt, exlen, delta_tau, omega, gyro, alpha, lamb, D1, D2, fluctu);


return;
}


/*-----------------------------------------------------------------------------*/
/*---------------------------Reading of input files----------------------------*/

void readxyz(void)
{

int w=0;

if((inp_file=fopen("coord_z.xyz", "r")) == NULL)
{
	printf("Could not open the file.\n");
	printf("The program is going to exit now.");
	exit(1);
}

//Trajectory file

out_file=fopen("trj.xyz", "w");

NewTime = fopen("coord_last.xyz", "w");

//Reading input

fscanf(inp_file,"%d\n", &natom);
fscanf(inp_file,"%s\n", &cmt);

//Picking up nearest multiple of block size for ideal CUDA shared memory interactions

if (block == 0)
        blocknatom = natom;

    int remainder = abs(int(natom)) % block;
    if (remainder == 0)
        blocknatom = natom;
    else
	blocknatom = natom + block - remainder;
	printf("\nnext multiple of %d is %d!\n", block, blocknatom);




alocate_vec();


for(w = 0; w < natom ; w++)
{

	fscanf(inp_file,"%d %lf %lf %lf %*s %lf %lf %lf\n", &label[w], &x[w], &y[w], &z[w], &mx[w], &my[w], &mz[w]);


}

//Dimensionless DM field intensity

if( ( (D1 != 0.0)||(D2 != 0.0) ) )//WARNING: THIS ASSUMES THE COORDINATES ARE Z-ORDERED!!!!
{


D1 *= latt*latt/Jcell;
D2 *= latt*latt/Jcell;


printf("\nNormalized DM -> %le %le\n", D1, D2);

if(curveflag == 1)
{
   //Attributing values on a per-layer basis - bottom layer is closest to Platinum alloy
   for(int l = 1; l < natom-1; l++)
   {
	double limit_break = abs(z[l] - z[0])*latt;

	if( (limit_break <= L1) && (limit_break < L2) ) hDM[l] = D1;
	else if( (limit_break > L1) && (limit_break < L2) ) hDM[l] = D2;
	else if( limit_break >= L2) hDM[l] = 0.0;
  
   }
}

else if(curveflag == 2)
{

   for(int l = 1; l < natom-1; l++) hDM[l] = D1;

}

else{
printf("\n No valid value on the [32] settings input has been detected. Verify and re-attempt please. Progam exitting now.\n");
exit(1);
}

}
else
{
for(int l = 0; l < natom-1; l++) hDM[l] = 0.0;
}


//Current density propagation velocity term

for(int l = 0; l < natom; l++)
{

host_vj[l] = (etha*J_curr)/(latt*omega);

}

//Finding x-range boundary of system

	
	//Finding max and minimum values of x
	//Needed only for STT term
	//Because in first order we use the equivalent Hamiltonian
	xmin = thrust::min_element(x, x + natom);
	xmax = thrust::max_element(x, x + natom);

	cudaMemcpyAsync(dxmin, xmin, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dxmax, xmax, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vj_aw, host_vj, size_coord, cudaMemcpyHostToDevice);


printf("\n system boundaries: x_min = %lf, x_max = %lf\n", *xmin, *xmax);


return;

}

/*-----------------------------------------------------------------------------*/
/*---------------------------Allocation function-------------------------------*/


/*--------------CPU VECTORS----------------*/

void alocate_vec(void)
{

int f;


size_coord = sizeof(double)*blocknatom;
size_lbl = sizeof(int) * blocknatom;
size_list = sizeof(int) * natom * natom;

size_block = block*sizeof(double);


cudaMallocHost((void**)&x, size_coord);
cudaMallocHost((void**)&y, size_coord);
cudaMallocHost((void**)&z, size_coord);

cudaMallocHost((void**)&mx, size_coord);
cudaMallocHost((void**)&my, size_coord);
cudaMallocHost((void**)&mz, size_coord);

cudaMallocHost((void**)&hDM, size_coord);

label = (int *)malloc(size_lbl);
cudaMalloc((void**)&dlabel, size_lbl);
cudaMallocHost((void**)&xmin, sizeof(double));
cudaMallocHost((void**)&xmax, sizeof(double));
cudaMalloc((void**)&dxmin, sizeof(double));
cudaMalloc((void**)&dxmax, sizeof(double));




if(integrator==0)
{

cudaMallocHost((void**)&mtempx, size_coord);
cudaMallocHost((void**)&mtempy, size_coord);
cudaMallocHost((void**)&mtempz, size_coord);



cudaMallocHost((void**)&fx, size_coord);
cudaMallocHost((void**)&fy, size_coord);
cudaMallocHost((void**)&fz, size_coord);

cudaMallocHost((void**)&fx_1, size_coord);
cudaMallocHost((void**)&fx_2, size_coord);
cudaMallocHost((void**)&fx_3, size_coord);

cudaMallocHost((void**)&fy_1, size_coord);
cudaMallocHost((void**)&fy_2, size_coord);
cudaMallocHost((void**)&fy_3, size_coord);

cudaMallocHost((void**)&fz_1, size_coord);
cudaMallocHost((void**)&fz_2, size_coord);
cudaMallocHost((void**)&fz_3, size_coord);
}


if(integrator==0)
{
Cx = (double *)malloc(size_coord);
Cy = (double *)malloc(size_coord);
Cz = (double *)malloc(size_coord);
mx_1 = (double *)malloc(size_coord);
my_1 = (double *)malloc(size_coord);
mz_1 = (double *)malloc(size_coord);
}

gpuErrchk( cudaMalloc((void**)&ry, size_coord) );
gpuErrchk( cudaMalloc((void**)&rz, size_coord) );
gpuErrchk( cudaMalloc((void**)&rx, size_coord) );


gpuErrchk( cudaMalloc((void**)&dev_mtempx, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_mtempy, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_mtempz, size_coord) );

gpuErrchk( cudaMalloc((void**)&dev_nviz, size_lbl) );
gpuErrchk( cudaMalloc((void**)&dev_threshold, size_lbl) );
gpuErrchk( cudaMalloc((void**)&dev_x, size_coord) );	
gpuErrchk( cudaMalloc((void**)&dev_y, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_z, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_mx, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_my, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_mz, size_coord) );
gpuErrchk( cudaMalloc((void**)&DM, size_coord) );

gpuErrchk( cudaMalloc((void**)&Hdipx, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hdipy, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hdipz, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hexx, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hexy, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hexz, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hanisx, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hanisy, size_coord) );
gpuErrchk( cudaMalloc((void**)&Hanisz, size_coord) );
gpuErrchk( cudaMalloc((void**)&HDMx, size_coord) );
gpuErrchk( cudaMalloc((void**)&HDMy, size_coord) );
gpuErrchk( cudaMalloc((void**)&HDMz, size_coord) );

gpuErrchk( cudaMalloc((void**)&vj_aw, size_coord) );
cudaMallocHost((void**)&host_vj, size_coord);

cudaMallocHost((void**)&n_viz, size_lbl);
cudaMallocHost((void**)&threshold, size_lbl);

gpuErrchk( cudaMalloc((void**)&dev_fx, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fy, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fz, size_coord) );
if(integrator==0)
{
gpuErrchk( cudaMalloc((void**)&dev_fx_1, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fy_1, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fz_1, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fx_2, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fy_2, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fz_2, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fx_3, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fy_3, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fz_3, size_coord) );
}

if(integrator==2)
{
gpuErrchk( cudaMalloc((void**)&dev_fx_1, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fy_1, size_coord) );
gpuErrchk( cudaMalloc((void**)&dev_fz_1, size_coord) );
}


if(lang==1)
{

gpuErrchk( cudaMalloc((void**)&xran, size_coord) );
gpuErrchk( cudaMalloc((void**)&yran, size_coord) );
gpuErrchk( cudaMalloc((void**)&zran, size_coord) );

curandCreateGenerator(&langerand, CURAND_RNG_PSEUDO_MTGP32);

curandSetPseudoRandomGeneratorSeed(langerand, semente);


}

gpuErrchk( cudaMalloc((void**)&dev_epot1, size_block) );

gpuErrchk( cudaMalloc((void**)&dev_epot2, size_block) );

gpuErrchk( cudaMalloc((void**)&dev_epot3, size_block) );

gpuErrchk( cudaMalloc((void**)&dev_epot4, size_block) );

gpuErrchk( cudaMalloc((void**)&dev_epot5, size_block) );

gpuErrchk( cudaMalloc((void**)&d_Mblockx, size_block) );

gpuErrchk( cudaMalloc((void**)&d_Mblocky, size_block) );

gpuErrchk( cudaMalloc((void**)&d_Mblockz, size_block) );

gpuErrchk( cudaMalloc((void**)&d_mm_, size_block) );

cudaMallocHost((void**)&epot1, size_block);

cudaMallocHost((void**)&epot2, size_block);

cudaMallocHost((void**)&epot3, size_block);

cudaMallocHost((void**)&epot4, size_block);

cudaMallocHost((void**)&epot5, size_block);

cudaMallocHost((void**)&mm_, size_block);

cudaMallocHost((void**)&Mblockx, size_block);

cudaMallocHost((void**)&Mblocky, size_block);

cudaMallocHost((void**)&Mblockz, size_block);


/*zeroing vectors from the get go*/

for(f = 0; f < blocknatom; f++)
{

	mx[f] = 0.0;
	my[f] = 0.0;
	mz[f] = 0.0;

	

	if(integrator==0)
	{

	mtempx[f] = 0.0;
	mtempy[f] = 0.0;
	mtempz[f] = 0.0;

	mx_1[f] = 0.0;
	my_1[f] = 0.0;
	mz_1[f] = 0.0;

	Cx[f] = 0.0;
	Cy[f] = 0.0;
	Cz[f] = 0.0;

	}

	x[f] = 0.0;
	y[f] = 0.0;
	z[f] = 0.0;




}

return;

}

/*-----------------------------------------------------------------------------*/
/*----------------------Simple random number generator-------------------------*/

//Use for test only - too slow, too clumsy, too small of a period

double aleatorio(void)
{

double semente_tmp = 0.0;
semente_tmp++;
/*
n_semente = (8127*n_semente + 28417) % 134453;
semente_tmp = (double)n_semente;
nran = semente_tmp / 134453.0;
n_semente = semente;

return (nran);
*/
return 0;
}


/* --------------------------------------------------------------------- */
/* -------------------Neighbor list  functions-------------------------- */

void reallocable_list(void)
{

//The neighbor list is built only once - in CPU - as it has a reliable realloc() function

int sum = 0;
int s = 0, w = 0;
double x2 = 0, y2 = 0, z2 = 0, rij2 = 0;

	for(s = 0; s < natom; s++)
	{
		threshold[s] = sum2;

		#pragma unroll
		for(w = 0; w < natom; w++)
		{

		
		if(w != s)
		{

		x2 = x[s] - x[w];
		y2 = y[s] - y[w];
		z2 = z[s] - z[w];

		rij2 = sqrt(x2*x2 + y2*y2 + z2*z2);
		
		if(rij2 <= 1.01)
		{

			sum++;
			sum2++;

			new_viz = (int*) realloc (viz, sizeof(int)*sum2);

			if(new_viz != NULL)
			{
			
			viz = new_viz;
			viz[sum2 - 1] = w;
			}
			else
			{
			puts("Error allocating new memory block");
			exit (1);
			}


		}
			
			rij2 = 0.0;
		}

		}
	
	n_viz[s] = sum;
	//printf("\n neighbors of %d: %d\n",s, sum);
	sum = 0;

	}

size_list = sum2 * sizeof(int);

cudaMalloc((void**)&dev_viz, size_list);

cudaMemcpy(dev_viz, viz, size_list, cudaMemcpyHostToDevice);
cudaMemcpy(dev_nviz, n_viz, size_lbl, cudaMemcpyHostToDevice);
cudaMemcpy(dev_threshold, threshold, size_lbl, cudaMemcpyHostToDevice);


printf("\nSize of neighbor list:%d \n", sum2);

}


