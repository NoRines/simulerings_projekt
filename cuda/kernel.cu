
#include "func.cuh"
#include "prng.h"
#include <stdio.h>
#include <iostream>
#define NUMBER_OF_VALUES (0x2FFFFFFF)
template<typename T>
void print(T* input, unsigned int size)
{
	for (int i = 0; i < size; i++)
		std::cout << input[i] << ' ';
	std::cout << std::endl;
}


int main(int argc, int argv[])
{
	unsigned int arraySize = NUMBER_OF_VALUES;
	if (argc == 2)
		arraySize = argv[2];
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
    
	unsigned int * a = new unsigned int[arraySize];
	prng(a, 4, 1, 9, 1, arraySize);
	unsigned int * dev_a;
	float * dev_transformed;
	cudaMalloc(&dev_a, arraySize * sizeof(unsigned int));
	cudaMalloc(&dev_transformed, arraySize * sizeof(float));
	cudaMemcpy(dev_a, a, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);
	transformKernel << <2048,128>> > (dev_a, 9, dev_transformed, arraySize);
	cudaFree(dev_a);

	float * transformed_res = new float[arraySize];
	expTransformKernel << <2048, 128 >> > (dev_transformed, 2, arraySize);
	cudaMemcpy(transformed_res, dev_transformed, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
	print(transformed_res, arraySize);
	system("pause");
    return 0;
}




