
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>

__global__ void medianFilter(float *input_img, float *output_img, int dim_N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float rez = 0;

	for (int k1 = -1; k1 < 2; k1++)
	{
		for (int k2 = -1; k2 < 2; k2++)
		{
			if ((k1 + i) >= 0 && (k1 + i) < dim_N && (k2 + j) >= 0 && (k2 + j) < dim_N)
			{
				int memory_loc = (k1 + i) + (k2 + j)*dim_N;
				rez += input_img[memory_loc];
			}
		}
	}

	output_img[i + j * dim_N] = rez / 9;

}

int main(int argc, char **argv) {

	int N = 1024;

	dim3 bs(32, 32);
	dim3 gs(N / bs.x, N / bs.y);

	//alocare memorie host
	float *img = (float*)malloc(N*N * sizeof(float));
	float *img_2 = (float*)malloc(N*N * sizeof(float));

	//initializare
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			img[i + j * N] = i % 2;


	//alocare memorie device
	float *img_in;
	cudaMalloc((void**)&img_in, N*N * sizeof(float));
	float *img_out;
	cudaMalloc((void**)&img_out, N*N * sizeof(float));


	cudaMemcpy(img_in, img, N*N * sizeof(float), cudaMemcpyHostToDevice);

	medianFilter << <gs, bs >> > (img_in, img_out, N);


	cudaMemcpy(img_2, img_out, N*N * sizeof(float), cudaMemcpyDeviceToHost);


	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			printf("%.2f ", img_2[i + j * N]);
		}

		printf("\n");
	}

	cudaFree(img);
	cudaFree(img_2);

	return 0;
}

