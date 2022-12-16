#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <iostream>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

using namespace std;

void verify_result(int* h_v, int N, int GpuRes) {
	chrono::system_clock::time_point startCpu = chrono::system_clock::now();
	for (int i = 1; i < N; i++) {
		h_v[0] += h_v[i];
	}
	chrono::system_clock::time_point endCpu = chrono::system_clock::now();
	auto timeCpu = chrono::duration_cast<chrono::nanoseconds>(endCpu - startCpu).count();
	cout << "CPU TIME : " << timeCpu << "ns\n";
	//assert(h_v[0] == GpuRes);
	printf("CPU res is %d \n", h_v[0]);
}

__global__ void sum_reduction(int* v, int* v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = rand() % 10;
	}
}

int main() {
	// Vector size
	int n = 1 << 8;
	size_t bytes = n * sizeof(int);

	// Original vector and result vector
	int* h_v, * h_v_r;
	int* d_v, * d_v_r;

	// Allocate memory
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// Initialize vector
	initialize_vector(h_v, n);

	// Copy to device
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// TB Size
	int TB_SIZE = SIZE;

	// Grid Size (No padding)
	int GRID_SIZE = n / TB_SIZE;

	chrono::system_clock::time_point start = chrono::system_clock::now();

	// Call kernel
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	chrono::system_clock::time_point end = chrono::system_clock::now();
	auto time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

	// Copy to host;
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	cout << "GPU TIME : " << time << "ns\n";
	printf("GPU result is %d \n", h_v_r[0]);
	verify_result(h_v, n, h_v_r[0]);
	//assert(h_v_r[0] == 65536);
	/*for (int i = 0; i < n; i++) {
		printf("The %d element is %d \n", i + 1, h_v[i]);
	}*/

	return 0;
}