

#include "pch.h"
#include <iostream>
#include <omp.h>
#include <time.h>
#include <string>


int main()
{
	clock_t start = clock();

	//linii, coloane
	int N = 1000;
	int M = 1000;

	//matrice, vector, resultat
	double *A, *B, *C;
	A = new double[N*M];
	B = new double[N];
	C = new double[N];

	//initializare matrice
	for (int i = 0; i < N; i++)
	{
		//initializare vector
		B[i] = 2;
		for(int j = 0; j < M; j++)
		{ 
			A[i*N + j] = 2;
		}
	}

	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < M; j++)
	//	{
	//		C[i] = A[i*N + j] * B[j];
	//	}
	//}

omp_set_num_threads(8);

#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nr = omp_get_num_threads();
		int start = id * N / nr;
		int end = (id + 1)*N / nr;
		for (int i = start; i < end; i++)
		{
			for (int j = 0; j < N; j++) {
				C[i] = A[i*N + j] * B[j];
			}
		}

	}

//	omp_set_num_threads(8);
//#pragma omp parallel for
//		for (int i = 0; i < N; i++)
//		{
//			for (int j = 0; j < M; j++)
//			{
//				C[i] = A[i*N + j] * B[j];
//			}
//			
//		}

	for (int i = 0; i < N; i++)
	{
		std::cout << C[i] << std::endl;
	}

	clock_t end = clock();
	printf("Time: %fl \n", (end - start) / CLOCKS_PER_SEC);
}



