 /*
 
 /\-/\    /\-/\    /\-/\    /\-/\    /\-/\    /\-/\    /\-/\    /\-/\
(=^Y^=)  (=^Y^=)  (=^Y^=)  (=^Y^=)  (=^Y^=)  (=^Y^=)  (=^Y^=)  (=^Y^=)
 (>o<)    (>o<)    (>o<)    (>o<)    (>o<)    (>o<)    (>o<)    (>o<)
 
 ==============================================================================
 								SPIN-LEGO Version 1.0
 ==============================================================================
 AUTHOR: 
		Maxwel Gama Monteiro Junior
		

 ================================================================================================================
 ================================================================================================================
 DESCRIPTION:


	Creates a cubic cell/finite difference grid equivalent of structures with varying magnetization fields.
	You can combine magnetization fields with geometries at will with the risk of creating inconsistent
	and/or impossible configurations.
	
 =================================================================================================================
 =================================================================================================================

  INSTRUCTIONS:

	A blueprint.inp file containing the relevant input parameters must be present in the directory where this
	code is going to run. Code is verbose and orderly to facilitate the addition of either new fields or new
	geometries.
	


		MAIN PROGRAM STARTS HERE
		
 ================================================================================
 								SPIN-LEGO
 ================================================================================

*/

//
//============================================================================================================================================
//											VARIABLES
//============================================================================================================================================
//

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>


double pi = 3.1415926535897932; 

int Lx, Ly, Lz; 

double nran; 

int n_semente, semente; 

int Nx, Ny, Nz, N; 

double *x, *y, *z;
double *mx, *my, *mz; 

int *in; 

int field; 

int i, j, k, n, NewN = 0; 


int shape; 

FILE *inp_file; 

FILE *out_file; 

double aleatorio(void); 

double unit_vec(void); 




//
//============================================================================================================================================
//											SHAPE PARAMETERS
//============================================================================================================================================
//Devemos fazer um header file parameters.h que declara todas essas variáveis separadamente, para limpar o código quando tivermos mais estruturas;

//++++SHAPE=3++++
double theta; 
int PacDimension; 
//++++SHAPE=4++++
double r_hole; 
//++++SHAPE=5++++
double dphi; 
//++++SHAPE=6++++
int r2; 
int h1, h2; 


//
//============================================================================================================================================
//											FIELD VARIABLES
//============================================================================================================================================

//+++FIELD=2,3
double skyr_size; 

//+++FIELD=4
double ex, ey, ez;

//============================================================================================================================================

int main()
{

printf("Starting program =======================================================================>\n");


if((inp_file=fopen("blueprint.inp", "r")) == NULL)
{
	
	printf("Could not open the file. Are you sure \"blueprint.inp\" is inside this folder?\n");
	printf("The program is going to exit now.\n");
	exit(1);

}

fscanf(inp_file, "%d\n", &shape);			
fscanf(inp_file, "%d %d %d\n", &Lx, &Ly, &Lz);
fscanf(inp_file, "%d\n", &semente);
fscanf(inp_file, "%lf\n", &theta);
fscanf(inp_file, "%d\n", &PacDimension);
fscanf(inp_file, "%lf\n", &r_hole);
fscanf(inp_file, "%d\n", &field);
fscanf(inp_file, "%lf\n", &skyr_size);
fscanf(inp_file, "%d\n", &r2);
fscanf(inp_file, "%d %d\n", &h1, &h2);
fscanf(inp_file, "%lf %lf %lf\n", &ex, &ey, &ez);
fclose(inp_file);

Nz = Lz + 1; 

Nx = Lx + 1;
Ny = Ly + 1; 

N = Nx*Ny*Nz; 

//
//============================================================================================================================================
//											SIMULATION BOX
//============================================================================================================================================
//

out_file = fopen("coord_z.xyz", "w"); 

double size_coord = sizeof(double) * N;


x = (double *)malloc(size_coord);
y = (double *)malloc(size_coord);
z = (double *)malloc(size_coord);
mx = (double *)malloc(size_coord);
my = (double *)malloc(size_coord);
mz = (double *)malloc(size_coord);

in = (int *)malloc(sizeof(int)*N);

n = 0; 

for(k = 0; k < Nz; k++)
{
	for(j = 0; j < Ny; j++)
	{
		for(i = 0; i < Nx; i++)
		{
			in[n] = 0;
			x[n] = (double)(i+1) - ((double)(Lx)/2.0 + 1.0);
			y[n] = (double)(j+1) - ((double)(Ly)/2.0 + 1.0);
			z[n] = (double)(k+1) - ((double)(Lz)/2.0 + 1.0);  
			n += 1;
		}
	}
}



//
//============================================================================================================================================
//					FIELD GENERATION 
//============================================================================================================================================
//Generates Field across "all" of space/simulation box

switch(field)
{

case 0://Random
{
	n_semente = semente;

	for(n = 0; n < N; n++) //Formato do número aleatório: (-1)^(int(n)) * n (se int(n) par, o número sorteado é positivo, se n ímpar, é negativo)
	{
		aleatorio();
		mx[n] =  nran*(1 - (-1) + 0.9999) - 1;
		aleatorio();
		my[n] =  nran*(1 - (-1) + 0.9999) - 1;
		aleatorio();
		mz[n] =  nran*(1 - (-1) + 0.9999) - 1;


	}

		unit_vec();
		break;
}

case 1: //Simple planar vortex with singularity case
{

	for(n = 0; n < N; n++)
	{
	if(x[n] != 0.0 || y[n] != 0.0)
	{
	double theta = atan2(y[n], x[n]);
	
	mx[n] = -sin(theta);
	my[n] = cos(theta);
	mz[n] = 0.0;
	}
	else
	{
	mx[n] = 0.0;
	my[n] = 0.0;
	mz[n] = 1.0;
	}

	}
unit_vec();
break;
}


case 2: //Skyrmion on cylinder plane XZ
{

	for(n = 0; n < N; n++)
	{
	double rho = sqrt(x[n]*x[n] + z[n]*z[n]);
	double Mr = (rho*rho - skyr_size*skyr_size)/(rho*rho + skyr_size*skyr_size);
	double theta = atan2(x[n],z[n]);
	if(y[n] > (Lx/2)-3 )
	{

	mx[n] = sqrt(1 - Mr*Mr)*sin(theta);
	my[n] = Mr;
	mz[n] = sqrt(1 - Mr*Mr)*cos(theta);

	}
	else
	{
	mx[n] = cos(atan2(y[n],x[n]));
	my[n] = sin(atan2(y[n],x[n]));
	mz[n] = 0.0;
	}
	}
unit_vec();
break;
}

case 3: //Simple Skyrmion with breaking inversion in perpendicular-to-XY plane
{

	for(n = 0; n < N; n++)
	{

	double rho = sqrt(x[n]*x[n] + y[n]*y[n]);
	double Mr = (rho*rho - skyr_size*skyr_size)/(rho*rho + skyr_size*skyr_size);
	double theta = atan2(y[n],x[n]);

	mx[n] = sqrt(1 - Mr*Mr)*sin(theta);
	my[n] = sqrt(1 - Mr*Mr)*cos(theta);
	mz[n] = Mr;	

	}
unit_vec();

break;
}

case 4: //Saturation towards direction (ex,ey,ez)
{

	for(n = 0; n < N; n++)
	{

	mx[n] = ex;
	my[n] = ey;
	mz[n] = ez;

	}

unit_vec();
break;
}


}//End switch
//
//============================================================================================================================================
//													RECTANGLE
//============================================================================================================================================
//

switch(shape)
{

case 0: 
{
fprintf(out_file, "%d\n", N);
fprintf(out_file, "Squarelike\n");

	for(k = 0; k < N; k++)
	{

	fprintf(out_file, "1 %16.15lf %16.15lf %16.15lf atom_vector %16.15lf %16.15lf %16.15lf\n", x[k], y[k], z[k], mx[k], my[k], mz[k]);

	}

break;
}
//
//============================================================================================================================================
//													SPHERE
//============================================================================================================================================
//

case 1: 
{

if((Lx!=Ly)||(Lx!=Lz)||(Ly!=Lz))
{
	printf("You asked for a sphere but these sizes are unequal! Please refer to your nearest linear algebra book.\n");
	printf("Program exitting now. Sorry!\n");
	exit(1);
}

double r;
double Rs = Lx/2.0; 
	
for(k = 0; k < N; k++)
{
	
	r = sqrt(x[k]*x[k] + y[k]*y[k] + z[k]*z[k]);
	
	if(r < Rs)
	{
	
	NewN++;
	
	in[k] = 1;
	
	}
	
}

fprintf(out_file, "%d\n", NewN);
fprintf(out_file, "Spherelike\n");

	for(k = 0; k < N; k++)
	{
	

	if(in[k] == 1)fprintf(out_file, "1 %16.15lf %16.15lf %16.15lf atom_vector  %16.15lf %16.15lf %16.15lf\n", x[k], y[k], z[k], mx[k], my[k], mz[k]);

	}

break;
}
//
//============================================================================================================================================
//													CYLINDER
//============================================================================================================================================
//

case 2: 
{

if(Lx == Ly) 
{

double r;
double Rs = (Lx+1)/2.0; 


for(k = 0; k < N; k++)
{
	
	r = sqrt(x[k]*x[k] + y[k]*y[k]); 
	
	if(r < Rs)
	{
	
	NewN++;
	
	in[k] = 1;
	
	}
	
}

}

if(Lx != Ly) 
			 //http://math.stackexchange.com/questions/41940/is-there-an-equation-to-describe-regular-polygons
{
	
	double a = (Lx+1)/2.0; 
	double b = (Ly+1)/2.0; 
	double Rs;
	
	for(k = 0; k < N; k++)
	{
		
		Rs = ( (x[k]*x[k])/(a*a) ) + ( (y[k]*y[k])/(b*b) );
		
		if(Rs < 1.01) 
		{
			
			NewN++;
			
			in[k] = 1;
			
		}
		
	}
	
}


fprintf(out_file, "%d\n", NewN);
fprintf(out_file, "Cylinderlike\n");

	for(k = 0; k < N; k++)
	{
	

	if(in[k] == 1)fprintf(out_file, "1 %16.15lf %16.15lf %16.15lf atom_vector  %16.15lf %16.15lf %16.15lf\n", x[k], y[k], z[k], mx[k], my[k], mz[k]);

	}
break;
}

//
//============================================================================================================================================
//													PAC_MAN
//============================================================================================================================================
//

case 3: 
{
	
	if(PacDimension!=1)
	{
	printf("Unavailable PacMan shape for version. Please insert only 1 (plane) in the blueprint!\n");
	printf("Program exitting now. Sorry!\n");
	exit(1);
	}
	
		

			if(Lx != Ly)
			{
			printf("Pacman must be circular or he won't be able to properly digest ghosts. Please make sure Lx = Ly!\n");
			printf("Program exitting now. Sorry!\n");
			exit(1);
			}

			double r;
			double beta; 
			double Rs = Lx/2.0;
			theta = theta * (pi/180); //Convertendo para radianos
			
						if( theta>=2.0*pi)// >= !!!
						{
						printf("Please choose less than a full revolution (2*pi) as the angle. \n");
						printf("Program exitting now. Sorry!\n");
						exit(1);
						}
			
						if(theta==0.0)
						{
						printf("Warning: the option you have chosen is equivalent to a full cylinder!\n");
						}
			
			
			for(k = 0; k<N; k++) 
			{
				
				if( (x[k]==0.0)&&(y[k]==0.0) )
				{
					
					NewN++;
					in[k] = 1;
					continue; 
					
				}
				
				r = sqrt(x[k]*x[k] + y[k]*y[k]);
				
				if(r < Rs) 
				{
					
					if(theta == 0.0)
					{
							NewN++;
							in[k] = 1;
					}
					
					else if(theta < pi/2.0) 
					{
						
						beta = fabs( atan(y[k]/x[k]) );
						if( (beta > theta)||(x[k]<=0.0)||(y[k]<=0.0) )
						
						{
							NewN++;
							in[k] = 1;
						}
						
					}

					else if( (theta > pi/2.0)&&(theta < pi) )
					{
						
						beta = fabs( atan(y[k]/x[k]) );
						if( ( (beta < pi - theta)&&(x[k]<0.0) )||( (x[k]<=0.0)&&(y[k]<=0.0) )||( (x[k]>=0.0)&&(y[k]<=0.0)) )
						{
							NewN++;
							in[k] = 1;
						}
						
					}
					
					else if( (theta > pi)&&(theta < (3*pi)/2.0) ) 
					{
						
						beta = fabs( atan(y[k]/x[k]) );
						if( ( (beta > theta - pi)&&(y[k]<0.0) )||( (x[k]>=0.0)&&(y[k]<=0.0) ) )
						{
							NewN++;
							in[k] = 1;
						}
					}
					
					else if(theta > (3*pi)/2.0)
					{
						
						beta = fabs( atan(y[k]/x[k]) );
						if ( (beta < 2*pi - theta)&&(y[k]<=0.0)&&(x[k]>=0.0))
						{
							NewN++;
							in[k] = 1;													
						}
						
					}
					
					else if(theta == pi/2.0) 
					{
						
						if((x[k]<=0.0)||(y[k]<=0.0) )
						{
							NewN++;
							in[k] = 1;				
						}
						
					}
					
					else if(theta == pi)
					{
						
						if( ( (x[k]<=0.0)&&(y[k]<=0.0) )||( ( x[k]>=0.0)&&(y[k]<=0.0) ) )
						{
							
							NewN++;
							in[k] == 1;
							
						}
						
					}
					
					else if(theta == (3*pi)/2.0) 
					{
						
						if( (x[k]>=0.0)&&(y[k]<=0.0) )
						{
							NewN++;
							in[k]==1;
						}
						
					}
					
				}
								
				
			}
		

fprintf(out_file, "%d\n", NewN);
fprintf(out_file, "PacMan\n");

	for(k = 0; k < N; k++)
	{
	

	if(in[k] == 1)fprintf(out_file, "1 %16.15lf %16.15lf %16.15lf atom_vector  %16.15lf %16.15lf %16.15lf\n", x[k], y[k], z[k], mx[k], my[k], mz[k]);

	}

	

break;
}
//
//============================================================================================================================================
//												TUBULAR
//============================================================================================================================================
//
case 4: 
{

if(Lx == Ly)
{

double r;
double Rs = (Lx)/2.0;


for(k = 0; k < N; k++)
{
	
	r = sqrt(x[k]*x[k] + y[k]*y[k]); 
	
	if(r < Rs && r > r_hole) 
	{
	
	NewN++;
	
	in[k] = 1;
	
	}
	
}

}

if(Lx != Ly) 
			 //http://math.stackexchange.com/questions/41940/is-there-an-equation-to-describe-regular-polygons
{
	
	double a = (Lx)/2.0; 
	double b = (Ly)/2.0;
	double Rs; 
	
	for(k = 0; k < N; k++)
	{
		
		Rs = ( (x[k]*x[k])/(a*a) ) + ( (y[k]*y[k])/(b*b) );
		
		if(Rs < 1.01 && Rs > r_hole)


		{
			
			NewN++;
			
			in[k] = 1;
			
		}
		
	}
	
}


fprintf(out_file, "%d\n", NewN);
fprintf(out_file, "Ringlike\n");

	for(k = 0; k < N; k++)
	{
	

	if(in[k] == 1)fprintf(out_file, "1 %16.15lf %16.15lf %16.15lf atom_vector  %16.15lf %16.15lf %16.15lf\n", x[k], y[k], z[k], mx[k], my[k], mz[k]);

	}
break;
}



//
//============================================================================================================================================
//													TOWER OF HANOI
//============================================================================================================================================
//

case 5:
{

double r;
double Rs = double(Lx+1)/2.0;
printf("Rs is %lf\n", Rs);
        for(k = 0; k < N; k++)
        {
        
          r = sqrt(x[k]*x[k] + y[k]*y[k]);

          //Site is only part of structure where r(h1) < r1 and r(h2) < r2
          if( (abs(z[k]) < double(h1) ) && (r <  Rs) )
          {
            
          NewN++;
          in[k] = 1;
            
          }
          
          else if( (abs(z[k]) < double(h2)) && (abs(z[k]) >= double(h1)) && (r < r2 ) )
          {
          
          NewN++;
          in[k] = 1;
          
          }
          
        
        }


fprintf(out_file, "%d\n", NewN);
fprintf(out_file, "Hanoi\n");

	for(k = 0; k < N; k++)
	{
	

	if(in[k] == 1)fprintf(out_file, "1 %16.15lf %16.15lf %16.15lf atom_vector  %16.15lf %16.15lf %16.15lf\n", x[k], y[k], z[k], mx[k], my[k], mz[k]);

	}


break;
}


}

//
//============================================================================================================================================
//													ENDOFMAIN
//============================================================================================================================================
//

fclose(out_file);

printf("\n<================================ Program exitting now. No one died, apparently!\n");

return 0;



}

double aleatorio(void)//Simple random numbers - not for use on actual simulations 
{

double semente_tmp = 0.0;

n_semente = (8127*n_semente + 28417) % 134453;
semente_tmp = (double)n_semente;
nran = semente_tmp / 134453.0;

return (nran);
}

double unit_vec(void)
{

double mm;
int u;

for(u = 0; u < N; u++)
{

mm = sqrt(mx[u] * mx[u] + my[u] * my[u] + mz[u] * mz[u]);

mx[u] = mx[u] / mm;
my[u] = my[u] / mm;
mz[u] = mz[u] / mm;

}

return 0;

}	
	
	
 
 	
	
	
	

 
