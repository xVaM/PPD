
#include "pch.h"
#include <iostream>
#include <time.h>

float gen(float a, float b)
{
	float random = ((float)rand()) / (float)RAND_MAX;
	float dif = b - a;
	float r = dif * random;
	return a + r;
}

int main()
{
	int counter = 0;
	int numbers = 10'000'000;

	const float a = -1.f;
	const float b = 1.f;
	const float cx = 0.f;
	const float cy = 0.f;
	const float radius = 1.f;
	float pi = 0.f;

	clock_t start = clock();

#pragma omp parallel for reduction(+:counter)
	{
		for (int i = 0; i < numbers; i++)
		{
			float rand_x = gen(a, b);
			float rand_y = gen(a, b);

			float distance = sqrt((double)(rand_x - 0.0)*(rand_x - 0.0) + (rand_y - 0.0)*(rand_y - 0.0));
			if (distance <= radius)
			{
				counter += 1;
			}

			pi = 4 * (counter / (float)(numbers));
		}
	}

	std::cout << pi << std::endl;

	clock_t end = clock();
	printf("Time: %fl \n", (end - start) / CLOCKS_PER_SEC);


}
