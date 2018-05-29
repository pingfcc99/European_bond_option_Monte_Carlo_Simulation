// Use curand to simulate the normal random variables

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include "stdafx.h"
#include "Test_MRG32k3a_Normal_kernel.cuh"


int main(int argc, char *argv[])
{
	int  n_timestep, n_path;
	double T, S, r0, kappa, theta, sigma, h, K, dt, sq_dt;
	double *d_normals, *h_result_call, *h_result_put, *d_result_call, *d_result_put, put, call;
	bool * h_flag, *d_flag; //flag to filter if r is negative in each path, in case of sqrt(r) gives -NAN(IND) value
	T = 2.0; // option maturity
	S = 3.0; // bond maturity
	K = 96;
	n_timestep = 365*T; 
	n_path = 100000;
	
	r0 = 0.01705;
	kappa = 0.057;
	theta = 0.06;
	sigma = 0.071;
	h = sqrt(kappa*kappa+2*sigma*sigma);

	if (2*kappa*theta > sigma*sigma)
		printf("Initial parameters are good.");
	else
		printf("Initial parameters are bad. Please stop running and reset parameters.");

	double tau = S - T;
	double denominator = (2.0*h + (kappa + h)*(exp(h*tau) - 1.0));
	double power = 2.0*kappa*theta / pow(sigma, 2.0);
	double numerator = 2.0*h*exp(0.5*(kappa + h)*tau);
	double A = pow((numerator / denominator), power);
	double B = (2.0*(exp(h*tau) - 1.0)) / denominator;

	size_t N_NORMALS = n_timestep*n_path;
	dt = T/n_timestep;
	sq_dt = sqrt(dt);
	
	cudaMalloc((void **)&d_normals, (N_NORMALS)*sizeof(double));
	h_flag = (bool *)malloc(n_path*sizeof(bool));
	cudaMalloc((void **)&d_flag, n_path*sizeof(bool));
	h_result_call = (double *)malloc(n_path*sizeof(double));
	cudaMalloc((void **)&d_result_call, n_path*sizeof(double));
	h_result_put = (double *)malloc(n_path*sizeof(double));
	cudaMalloc((void **)&d_result_put, n_path*sizeof(double));

	// generate normal random numbers N(0,sq_dt)
    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, 1256);//1234ULL) ;
	curandGenerateNormalDouble(curandGenerator, d_normals, N_NORMALS, 0.0f, sq_dt);

	printf("Calculate European call option prices (T=2) written a zero-coupon bond (S=3) with CIR Model.\n");
	printf("r0 = %f  kappa = %f  theta = %f  sigma = %f  K = %f \noption maturity = %f bond maturity = %f \n", r0, kappa, theta, sigma, K, T, S);
	printf(" \n  Now running time test on GPU, %i simulations per path and %i separate simuation paths \n",
				n_timestep, n_path);
	
	const int BLOCK_SIZE = 512;
	int GRID_SIZE = n_path / BLOCK_SIZE + 1;

	double t1 = double(clock()) / CLOCKS_PER_SEC;
	// call option
	Curand_European_Option <<<GRID_SIZE, BLOCK_SIZE >>>(
		n_timestep, n_path, r0, kappa, theta, sigma, h, K, T, S, dt, d_normals, A, B, d_result_call, d_flag, true);
		
	cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_call, d_result_call, n_path*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_flag, d_flag, n_path*sizeof(bool), cudaMemcpyDeviceToHost);

	// put option
	Curand_European_Option <<<GRID_SIZE, BLOCK_SIZE >>>(
		n_timestep, n_path, r0, kappa, theta, sigma, h, K, T, S, dt, d_normals, A, B, d_result_put, d_flag, false);
	cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_put, d_result_put, n_path*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_flag, d_flag, n_path*sizeof(bool), cudaMemcpyDeviceToHost);
	double t2 = double(clock()) / CLOCKS_PER_SEC;
	// calculate avg
	call = 0.0;
	put = 0.0;
	double flag_sum = 0.0;
	for (int i = 0; i < n_path; i++) 
	{
		if (h_flag[i] == true)
		{
			call = ((i- flag_sum) / (i + 1.-flag_sum))*call + (1./ (i + 1.- flag_sum))*h_result_call[i];
			put = ((i - flag_sum) / (i + 1. - flag_sum))*put + (1. / (i + 1.-flag_sum))*h_result_put[i];
		}
		else
			flag_sum += 1;
		
	}
	printf("GPU Monte Carlo Computation:   %f ms\n", (t2 - t1)*1e3);
	printf("\nThe European bond call price is %f    \n", call);
	printf("\nThe European bond put price is %f    \n", put);

	// Release memory allocation
	free(h_result_call);
	free(h_result_put);
	//	Release device memory
	cudaFree(d_result_call);
	cudaFree(d_result_put);
	cudaFree(d_normals);
	// destroy generator
	curandDestroyGenerator(curandGenerator);

	cudaDeviceReset();
	system("pause");
	return 0;

}




