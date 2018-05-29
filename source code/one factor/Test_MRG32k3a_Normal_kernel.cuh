//	GPU device code:  Header Files

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>


__global__ void Curand_European_Option(
	int n_timestep, int n_path, double r0, double kappa, double theta, double sigma, double h, double K, double T, double S, double dt, 
	double * d_normals, double A, double B, double *d_result, bool *d_flag, bool is_call);
