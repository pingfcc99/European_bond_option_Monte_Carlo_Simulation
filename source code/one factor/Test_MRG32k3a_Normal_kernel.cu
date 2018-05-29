//	GPU device code

#include "Test_MRG32k3a_Normal_kernel.cuh"
#include <math.h>       /* pow */


//	Process an array of n_path simulations of the simulation model on GPU
//	with n_timestep simulations per path, using call to curand function


__global__ void Curand_European_Option(
	int n_timestep, int n_path, double r0, double kappa, double theta, double sigma, double h, double K, double T, double S, double dt, 
	double * d_normals, double A, double B, double *d_result, bool *d_flag, bool is_call)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	iSim = tid;
	if (iSim < n_path) {
		int i, index;
		double r, z, sq_r, bond_T_S, payoff, discount_t_T;  // P(T,S), D(t,T)
		r = r0;
		index = tid*n_timestep; //*******************
		discount_t_T = 0;
		
		for (i = 0; i < n_timestep; i++) {
			discount_t_T += r*dt;  // numerically copmpute integral
			z = d_normals[index]; 
			if (r < 0)
			{
				d_flag[iSim] = false;
				break;
			}
			else
				d_flag[iSim] = true;
			sq_r = sqrt(r);
			r += kappa*(theta - r)*dt + sigma*sq_r*z;
			index++;
		}
		discount_t_T = exp(-discount_t_T);
		// compute P(T,S)
		bond_T_S = 100.*A*exp(-B*r);

		/*
		if (is_call == true) {
		payoff = (bond_T_S > K ? bond_T_S - K : 0.0);
		}
		else {
		payoff = (bond_T_S < K ? K - bond_T_S : 0.0);
		}
		*/
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

