
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void test(int *a, int *b, int *c)
{
	*c = *a + *b;
}

int main()
{
	int a = 1, b = 2, c;
	int *a_dev, *b_dev, *c_dev;

	//Alocare pe GPU
	cudaMalloc((void**)&a_dev, sizeof(int));
	cudaMalloc((void**)&b_dev, sizeof(int));
	cudaMalloc((void**)&c_dev, sizeof(int));

	//copiere CPU - GPU
	cudaMemcpy(a_dev, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, &b, sizeof(int), cudaMemcpyHostToDevice);

	//lansare in executie a kernelului
	test << <1, 1 >> > (a_dev, b_dev, c_dev);

	//copiere GPU-CPU
	cudaMemcpy(&c, c_dev, sizeof(int), cudaMemcpyDeviceToHost);

	printf("C= %d", c);

    return 0;
}

