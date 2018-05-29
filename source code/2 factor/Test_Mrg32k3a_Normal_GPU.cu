// Use curand to simulate the normal random variables

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include "stdafx.h"
#include "Test_MRG32k3a_Normal_kernel.cuh"
double V(double a, double sigma, double b, double eta, double rho, double tau);

int main(int argc, char *argv[])
{
	int  n_timestep, n_path;
	double T, S, x0, y0, a, sigma, b, eta, rho, rho2, K, dt, sq_dt;
	double *d_normals, *h_result_call, *h_result_put, *d_result_call, *d_result_put, put, call;
	//bool * h_flag, *d_flag; //flag to filter if r is negative in each path, in case of sqrt(r) gives -NAN(IND) value
	T = 2.0; // option maturity
	S = 3.0; // bond maturity
	K = 96;
	n_timestep = 365*T; 
	n_path = 100000;
	
	x0 = 0.01;
	y0 = 0.00705;
	a = 5;
	b = 0.35;
	sigma = 0.15;
	rho = -0.9;
	eta = 0.08;
	rho2 = sqrt(1 - rho*rho);

	// for the convience of computing P(T,S) in the kernel function Curand_European_Option()
	double P_2 = 0.946648 * 100;
	double P_3 = 0.918169 * 100;
	double tau = S - T;
	double A = P_3 / P_2 * exp((V(a, sigma, b, eta, rho, tau) - V(a, sigma, b, eta, rho, S) + V(a, sigma, b, eta, rho, T))*.5);
	double B1 = (1. - exp(-a*tau) ) / a;
	double B2 = (1. - exp(-b*tau)) / b;

	size_t N_NORMALS = 2*n_timestep*n_path;
	dt = T/double(n_timestep);
	sq_dt = sqrt(dt);
	
	cudaMalloc((void **)&d_normals, (N_NORMALS)*sizeof(double));
	//h_flag = (bool *)malloc(n_path*sizeof(bool));
	//cudaMalloc((void **)&d_flag, n_path*sizeof(bool));
	h_result_call = (double *)malloc(n_path*sizeof(double));
	cudaMalloc((void **)&d_result_call, n_path*sizeof(double));
	h_result_put = (double *)malloc(n_path*sizeof(double));
	cudaMalloc((void **)&d_result_put, n_path*sizeof(double));

	// generate normal random numbers N(0,sq_dt)
    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL) ;
	curandGenerateNormalDouble(curandGenerator, d_normals, N_NORMALS, 0.0f, sq_dt);

	printf("Calculate European call option prices (T=2) written a zero-coupon bond (S=3) with G2++ Model.\n");
	printf("x0 = %f  y0 = %f  a = %f  sigma = %f  \nb = %f  eta = %f  K = %f  option maturity = %f bond maturity = %f \n", x0,y0,a,sigma,b,eta, K, T, S);
	printf(" \n  Now running time test on GPU, %i simulations per path and %i separate simuation paths \n",
				n_timestep, n_path);
	
	const int BLOCK_SIZE = 512;
	int GRID_SIZE = n_path / BLOCK_SIZE + 1;

	double t1 = double(clock()) / CLOCKS_PER_SEC;

	// call option
	Curand_European_Option <<<GRID_SIZE, BLOCK_SIZE >>>(
		n_timestep, n_path, x0, y0, a, sigma, b, eta, rho, rho2, K, T, S, dt, d_normals, A, B1, B2, d_result_call, true);
		
	cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_call, d_result_call, n_path*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_flag, d_flag, n_path*sizeof(bool), cudaMemcpyDeviceToHost);

	// put option
	Curand_European_Option <<<GRID_SIZE, BLOCK_SIZE >>>(
		n_timestep, n_path, x0, y0, a, sigma, b, eta, rho, rho2, K, T, S, dt, d_normals, A, B1, B2, d_result_put, false);
	cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_put, d_result_put, n_path*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_flag, d_flag, n_path*sizeof(bool), cudaMemcpyDeviceToHost);

	double t2 = double(clock()) / CLOCKS_PER_SEC;

	// calculate avg of payoff
	call = 0.0;
	put = 0.0;
	//double flag_sum = 0.0;
	for (int i = 0; i < n_path; i++) 
	{

		call = (i / (i + 1.))*call + (1./ (i + 1.))*h_result_call[i];
		put = (i / (i + 1.))*put + (1. / (i + 1.))*h_result_put[i];
		/*
		if (h_flag[i] == true)
		{
			call = ((i- flag_sum) / (i + 1.-flag_sum))*call + (1./ (i + 1.- flag_sum))*h_result_call[i];
			put = ((i - flag_sum) / (i + 1. - flag_sum))*put + (1. / (i + 1.-flag_sum))*h_result_put[i];
		}
		else
			flag_sum += 1;
		*/
	}
	
	printf ( "GPU Monte Carlo Computation:   %f ms\n", (t2 - t1)*1e3 );
	printf("\nThe European bond call price is %f    \n", call);
	printf("\nThe European bond put price is %f    \n", put);

	// Release memory allocation
	//free(h_flag);
	free(h_result_call);
	free(h_result_put);
	//	Release device memory
	//cudaFree(d_flag);
	cudaFree(d_result_call);
	cudaFree(d_result_put);
	cudaFree(d_normals);
	// destroy generator
	curandDestroyGenerator(curandGenerator);

	cudaDeviceReset();
	system("pause");
	return 0;

}


double V(double a, double sigma, double b, double eta, double rho, double tau)
{
	double tmp1 = (pow(sigma, 2.) / pow(a, 2.))*(tau + (2. / a)*exp(-a*tau) - (.5 / a)*exp(-2.*a*tau) - 1.5 / a);
	double tmp2 = (pow(eta, 2.) / pow(b, 2.))*(tau + (2. / b)*exp(-b*tau) - (.5 / b)*exp(-2.*b*tau) - 1.5 / b);
	double tmp3 = ((2.*rho*sigma*eta) / (a*b))*(tau + (exp(-a*tau) - 1) / a + (exp(-b*tau) - 1) / b - (exp(-(a + b)*tau) - 1) / (a + b));
	return tmp1 + tmp2 + tmp3;
}
