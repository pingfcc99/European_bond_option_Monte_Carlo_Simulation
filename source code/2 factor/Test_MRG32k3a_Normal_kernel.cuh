//	GPU device code:  Header Files

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

__global__ void Curand_European_Option(
	int n_timestep, int n_path, double x0, double y0, double a, double sigma, double b, double eta, double rho, double rho2, double K, double T, double S, double dt,
	double * d_normals, double A, double B1, double B2, double *d_result, bool is_call);
