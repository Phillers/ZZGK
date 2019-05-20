
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include<thrust/reduce.h>
#include <stdio.h>

const int size = 16;
const int bits = 4;
__global__ void joinGroups(int tab1[], int out[], int N) {
	extern __shared__ int tab[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		tab[threadIdx.x] = tab1[idx];
	}

	_syncthreads();
	if (idx < N) {
		int a, b;
		if (threadIdx.x > 0) {
			a = tab[threadIdx.x - 1];
			b = tab[threadIdx.x];
		}
		else if (idx > 0) {
			a = tab1[idx - 1];
			b = tab[threadIdx.x];
		}
		int dif = a ^ b;
		if (__popc(dif) == 2&& dif > a&&dif > b) {
				out[idx] = 0;
		}
		else {
			out[idx] = 1;
		}
	}
}

struct isPositive {
	__host__ __device__ bool operator()(int x) {
		return x > 0;
	}
};

int main()
{
	thrust::host_vector<int> h_tab1;
	thrust::device_vector<int> d_tab1;
	thrust::device_vector<int> d_tab2;

	std::vector<int> sizes;
	int sum = 0;
	while(sum < size){
		int x = rand() % (size/2) + 1;
		if (sum + x > size)
			x = size - sum;
		sizes.push_back(x);
		sum += x;
	}
	int prefix = 0;
	int plength=0;
	for (int i = 1; i < bits;i++) {
		prefix <<= 1;
		prefix |= 1;
		plength += 1;
	}
	int dif = 1 << (plength-1);
	for (int x : sizes) {
		unsigned a = 1 << 31;
		for (int i = 0;i < x;i++) {
			h_tab1.push_back(a | prefix);
			a >>= 1;
		}
		prefix += dif;
		if (dif >> (plength - 1) == 1) {
			plength += 1;
		}
		dif >>= 1;
		if (dif == 0) {
			dif = 1 << (plength - 1);
		}
	}
	d_tab1 = h_tab1;
	d_tab2.resize(size);

	for (int i = 0;i < size;i++) {
		printf("%x\n", h_tab1[i]);
	}
		printf("\n\n");
	compare10 << <size / 64, 64, 64*sizeof(int) >> > (d_tab1.data().get(), d_tab2.data().get(), size);
	thrust::copy_if(thrust::c (0),
		thrust::make_counting_iterator(size),
		d_tab2.begin(),
		d_tab1.begin(),
		thrust::identity<int>());
	h_tab1 = d_tab1;
	for (int i = 0;i < size;i++) {
		printf("%d;", h_tab1[i]);
	}
	printf("\n\n");

	printf("%f", (h_tab1[size / 2] + h_tab1[size / 2 + 1]) / 2.0);
	*/
	return 0;
}

