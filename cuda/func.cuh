
#ifndef FUNC_CUH
#define FUNC_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

__global__ void transformKernel(unsigned int *input, float c, float * output, int count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
		output[i] = input[i] / c;
}

__global__ void expTransformKernel(float *data, float beta, unsigned int count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
	{
		data[i] = ((-1 / beta)*log(1 - data[i])); //because cdf of exp() is y = 1 - exp(-ax), inverse is: x = 1/-a * log(1-y), and beta = 1/a so for mean 2 we want 1/mean = a.
	}
}

__global__ void mapKernel(float *input, int * output, int count, float* thresholds, unsigned int t_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
	{
		output[i] = 0;
		for (int j = 0; j < t_count; j++)
		{
			if (input[i] < thresholds[j])
			{
				break;
			}
			else
			{
				output[i]++;
			}
		}
	}
}

#endif