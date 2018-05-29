//	GPU device code

#include "Test_MRG32k3a_Normal_kernel.cuh"
#include <math.h>       /* pow */


//	Process an array of n_path simulations of the simulation model on GPU
//	with n_timestep simulations per path, using call to curand function


__global__ void Curand_European_Option(
	int n_timestep, int n_path, double x0, double y0, double a, double sigma, double b, double eta, double rho, double rho2, double K, double T, double S, double dt, 
	double * d_normals, double A, double B1, double B2, double *d_result, bool is_call)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	iSim = tid;
	if (iSim < n_path) {
		int i, index;
		double x, y, z1, z2, bond_T_S, payoff, discount_t_T;  // P(T,S), D(t,T)
		x = x0;
		y = y0;
		index = 2*tid*n_timestep; //*******************
		discount_t_T = 0;
		
		for (i = 0; i < n_timestep; i++) {
			discount_t_T += (x+y)*dt;  // numerically copmpute integral
			z1 = d_normals[index]; 
			z2 = d_normals[index+1];		
			x += -a*x*dt + sigma*z1;
			y += -b*y*dt + eta*(rho*z1+rho2*z2);
			index+=2;
		}
		discount_t_T = exp(-discount_t_T);
		// compute P(T,S)
		bond_T_S = 100.*A*exp(-B1*x-B2*y);
		//printf(" %f  ", bond_T_S);

		if (is_call == true) {
			if (bond_T_S > K)
				payoff = bond_T_S - K;
			else
				payoff = 0;
		}
		else {
			if (bond_T_S > K)
				payoff = 0;
			else
				payoff = K - bond_T_S;
		}
		d_result[iSim] = payoff*discount_t_T;	
	}
}

