
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

void prng(int multipl, int mod_base, int c, int* values, int seed, int amount)
{
	values[0] = (seed * multipl + c) % mod_base;
	for (int i = 1; i < amount; i++)
	{
		values[i] = ( values[i - 1] * multipl + c) % mod_base;
	}
}

__global__ void transformKernel(int *input, float c, float * output, int count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;i += blockDim.x * gridDim.x)
		output[i] = input[i]/c;
}

template<typename T>
void print(T* input, int size)
{
	for (int i = 0; i < size; i++)
		std::cout << input[i] << ' ';
	std::cout<<std::endl;
}
__global__ void mapKernel(float *input, int * output, int count, float* thresholds, int t_count)
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
		



int main()
{
	
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}

    const int arraySize = 1000000;
    int * a = new int[arraySize];

	prng(4, 9, 1, a, 1, arraySize);
	//print(a, arraySize);


	int * dev_a;
	float * dev_transformed;
	
	cudaMalloc(&dev_a, arraySize * sizeof(float));
	cudaMalloc(&dev_transformed, arraySize * sizeof(float));
	cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	transformKernel << <4096,256>> > (dev_a, 9, dev_transformed, arraySize);



	float th[10] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
	//cudaFree(dev_a);
	float * transformed_res = new float[arraySize];
	cudaMemcpy(transformed_res, dev_transformed, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
	float * dev_th;
	//print(transformed_res, arraySize);

	cudaMalloc(&dev_th, 10 * sizeof(float));
	cudaMemcpy(dev_th, th, 10 * sizeof(float), cudaMemcpyHostToDevice);
	int* dev_res;
	cudaMalloc(&dev_res, arraySize * sizeof(int));
	mapKernel << <4096, 256 >> > (dev_transformed, dev_res, arraySize, dev_th, 10);

	int * res = new int[arraySize];
	cudaMemcpy(res, dev_res, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	//print(res, arraySize);
	system("pause");
    return 0;
}




