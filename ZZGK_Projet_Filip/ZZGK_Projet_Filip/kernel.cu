
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <stdio.h>

const int size = 8;
const int bits = 4;
const int blocksize = 64;

__global__ void joinGroups(unsigned tab1[], unsigned out[], unsigned N) {
	extern __shared__ int tab[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		tab[threadIdx.x] = tab1[idx];
	}

	__syncthreads();
	if (idx < N) {
		unsigned a, b;
		if (threadIdx.x > 0) {
			a = tab[threadIdx.x - 1];
			b = tab[threadIdx.x];
		}
		else if (idx > 0) {
			a = tab1[idx - 1];
			b = tab[threadIdx.x];
		}
		if (idx > 0) {
			unsigned dif = a ^ b;
			//printf("%d %x %x %x\n", idx, a, b, dif);
			if (__popc(dif) == 2 && dif > a && dif > b) {
				out[idx] = 0;
			}
			else {
				out[idx] = 1;
			}
		}
	}
}

__global__ void countCombination(unsigned sizes[], unsigned sizes_size, unsigned combinations[]) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < sizes_size) {
		unsigned tmp = sizes[idx];
		tmp = tmp * (tmp - 1) / 2;
		combinations[idx] = tmp;
	}
}
struct isPositive {
	__host__ __device__ bool operator()(int x) {
		return x > 0;
	}
};

__global__ void initGI(unsigned A[], unsigned Jt[], unsigned G[], unsigned I[], unsigned size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		printf("%d %d\n", idx, A[idx], size);
		if (A[idx]<31) atomicAdd(&G[A[idx]], 1);
		if (Jt[idx] != 0)
			I[A[idx]] = A[idx];
	}
}

__global__ void finalGI(unsigned G[], unsigned I[], unsigned size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		G[idx] = G[idx] - 1;
		I[idx] = idx - I[idx];
	}
}

int forGroups(thrust::device_vector<unsigned>& Jt, thrust::device_vector<unsigned>& G, thrust::device_vector<unsigned>& I,unsigned size) {
	thrust::device_vector<unsigned> A(size);
	thrust::exclusive_scan(Jt.begin(), Jt.end(), A.begin());
	int nt = A[size - 1] + Jt[size - 1];
	thrust::device_vector<unsigned> G2, I2;
	G.clear();G.resize(nt);
	I.clear();I.resize(nt);
	G2.clear();G2.resize(nt);
	I2.clear();I2.resize(nt);
	initGI << <(size + blocksize - 1) / blocksize, blocksize >> >( A.data().get(), Jt.data().get(), G2.data().get(), I2.data().get(), size);
	thrust::inclusive_scan(G2.begin(), G2.end(), G.begin());
	thrust::inclusive_scan(I2.begin(), I2.end(), I.begin(), thrust::maximum<unsigned>());
	finalGI << <(nt + blocksize - 1) / blocksize, blocksize >> > (G.data().get(), I.data().get(), nt);
	return nt;
}

__global__ void joinCombination(unsigned G[], unsigned I[], unsigned A[], unsigned outA[], unsigned outB[], unsigned outC[], int size, unsigned in[], int inSize, unsigned sizes[], int sizes_size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {

		int g = G[idx];

		int p = I[idx];

		int a = A[g];

		int n = sizes[g];

		int c = n - 1;

		int i = 0;
		while (p >= c) {
			p -= c;
			i++;
			c = n - i - 1;
		}

		unsigned inA = in[a + i];

		outA[idx] = inA;
		i++;
		c = 1;
		while (p >= c) {
			p -= c;
			i++;
		}
		unsigned inB = in[a + i];

		outB[idx] = inB;
		outC[idx] = inA | inB;
	}
}

__device__ int hashFunction(unsigned key, int hashSize, int i) {
	return (key + i) % hashSize;
}

__device__ bool check(unsigned *hashTab, int hashSize, unsigned key) {
	int hashIndex = hashFunction(key, hashSize, 0);
	bool cont = false;
	int i = 0;
	do {
		unsigned whatsInside;
		whatsInside = hashTab[hashIndex];
		if (whatsInside == 0) {
			return false;
		}
		else {
			if (whatsInside != key) {//Kolizja
				hashIndex = hashFunction(key, hashSize, ++i);
				cont = true;
			}
			else { //Klucz ju¿ w tablicy haszowej
				cont = false;
				return true;
			}
		}
	} while (cont);
	return false;
}
__global__ void checkSubComb(unsigned C[], int size, int count, unsigned out[], unsigned in[], int hashSize, int no[]) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		int index = idx / count;
		int bit = idx % count;
		unsigned set = C[index];
		unsigned tmp = set;
		int shift = 0;
		if (tmp & 1) {
			bit--;
		}
		while (bit >= 0) {
			tmp >>= 1;
			shift++;
			if (tmp & 1) {
				bit--;
			}
		}
		set ^= 1 << shift;
		if(check(in, hashSize, set))
			out[idx] = set;
		else no[index] = 1;
	}

}

__device__ bool insert(unsigned *hashTab, int hashSize, unsigned key) {
	int hashIndex = hashFunction(key, hashSize, 0);
	bool cont = false;
	int i = 0;
	do {
		unsigned whatsInside;
		whatsInside = atomicCAS(hashTab + hashIndex, 0, key);
		if (whatsInside == 0) {
			cont = false;
		}
		else {
			if (whatsInside != key) {//Kolizja
			hashIndex = hashFunction(key, hashSize, ++i);
			cont = true;
			} else { //Klucz ju¿ w tablicy haszowej
				cont = false;
				return true;
			}
		}
	} while (cont);
	return false;
}
__global__ void insertInput(unsigned input[], unsigned hashTable[], int size, int hashSize) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		insert(hashTable, hashSize, input[idx]);
	}
}

int main()
{
	thrust::host_vector<unsigned> h_tab1;
	thrust::device_vector<unsigned> d_tab1;
	thrust::device_vector<unsigned> d_tab2;
	thrust::device_vector<unsigned> d_tab3;
	thrust::device_vector<unsigned> d_tab4;
	std::vector<int> sizes;
/*	int sum = 0;
	while(sum < size){
		int x = rand() % (size/2) + 1;
		if (sum + x > size)
			x = size - sum;
		sizes.push_back(x);
		sum += x;
	}
	int prefix = 0;
	int plength=0;
	for (int i = 1u; i < bits;i++) {
		prefix <<= 1;
		prefix |= 1;
		plength += 1;
	}
	int dif = 1 << (plength-1);
	for (int x : sizes) {
		unsigned a = 1u << 31;
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
	}*/
	h_tab1.push_back(1 << 1 | 1 << 2 | 1 << 3 | 1 << 4);
	h_tab1.push_back(1 << 1 | 1 << 2 | 1 << 3 | 1 << 5);
	h_tab1.push_back(1 << 1 | 1 << 2 | 1 << 3 | 1 << 6);
	h_tab1.push_back(1 << 1 | 1 << 2 | 1 << 4 | 1 << 5);
	h_tab1.push_back(1 << 1 | 1 << 3 | 1 << 4 | 1 << 5);
	h_tab1.push_back(1 << 1 | 1 << 3 | 1 << 4 | 1 << 6);
	h_tab1.push_back(1 << 2 | 1 << 3 | 1 << 4 | 1 << 5);
	h_tab1.push_back(1 << 2 | 1 << 3 | 1 << 4 | 1 << 6);
	d_tab1 = h_tab1;
	d_tab2.resize(size);

	for (int i = 0;i < size;i++) {
		printf("%x\n", h_tab1[i]);
	}
		printf("\n\n");
	joinGroups << <(size + blocksize - 1)/ blocksize, blocksize, size*sizeof(int) >> > (d_tab1.data().get(), d_tab2.data().get(), size);
	h_tab1 = d_tab2;
	for (int i = 0;i < size;i++) {
		printf("%d;", h_tab1[i]);
	}
	printf("\n\n");
	d_tab3.resize(size);
	thrust::inclusive_scan(d_tab2.begin(), d_tab2.end(), d_tab3.begin());
	h_tab1 = d_tab3;
	for (int i = 0;i < size;i++) {
		printf("%d;", h_tab1[i]);
	}
	printf("\n\n");
	d_tab4.resize(size);
	auto x = thrust::reduce_by_key(d_tab3.begin(), d_tab3.end(), thrust::make_constant_iterator(1), d_tab2.begin(), d_tab4.begin());
	int sizes_size = x.second - d_tab4.begin();
	h_tab1 = d_tab4;
	for (int i = 0;i < sizes_size;i++) {
		printf("%d:%d\n", h_tab1[i], 0);//sizes[i]);
	}
	printf("\n\n");
	
	countCombination << < (sizes_size + blocksize - 1) / blocksize, blocksize >> > (d_tab4.data().get(), sizes_size, d_tab2.data().get());
	h_tab1 = d_tab2;
	for (int i = 0;i < sizes_size;i++) {
		printf("%d;", h_tab1[i]);
	}
	printf("\n\n");
	thrust::device_vector<unsigned> G, I;
	int nt = forGroups(d_tab2, G, I,sizes_size);
	h_tab1 = G;
	thrust::host_vector<unsigned> h_tab2 = I;
	for (int i = 0;i < nt;i++) {
		printf("%d\t%d\n", h_tab1[i], h_tab2[i]);
	}
	printf("\n\n");
	thrust::device_vector<unsigned> outA,outB,outC;
	outA.resize(nt);
	outB.resize(nt);
	outC.resize(nt);
	d_tab3.resize(sizes_size);
	thrust::exclusive_scan(d_tab4.begin(), x.second, d_tab3.begin());
	joinCombination<<<(nt+blocksize-1)/blocksize, blocksize >>>(G.data().get(), I.data().get(), d_tab3.data().get(), outA.data().get(), outB.data().get(), outC.data().get(), nt, d_tab1.data().get(), size, d_tab4.data().get(), sizes_size);
	h_tab1 = outA;
	h_tab2 = outB;
	thrust::host_vector<unsigned> h_tab3 = outC;
	for (int i = 0;i < nt;i++) {
		printf("%08x\t%08x\t%08x\n", h_tab1[i], h_tab2[i], h_tab3[i]);
	}
	printf("\n\n");
	thrust::device_vector<unsigned> hashTable(size * 3);
	insertInput << <(size + blocksize - 1) / blocksize, blocksize >> > (d_tab1.data().get(),hashTable.data().get() ,size,size*3);
	thrust::device_vector<unsigned> out(nt * (bits + 1));
	thrust::device_vector<int> outNo(nt);
	checkSubComb << <(nt * (bits + 1) + blocksize - 1) / blocksize, blocksize >> > (outC.data().get(), nt * (bits + 1), bits + 1, out.data().get(), hashTable.data().get(), size*3,outNo.data().get());
	h_tab1 = out;
	for (int i = 0;i < (nt );i++) {
		for(int j=0;j<bits;j++)
		printf("%08x;", h_tab1[i*(bits+1)+j]);
	}
	printf("\n\n");
	h_tab1 = outNo;
	for (int i = 0;i < (nt);i++) {
		printf("%d;", h_tab1[i]);
	}
	printf("\n\n");
		return 0;
	
}

