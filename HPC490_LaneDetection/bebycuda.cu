// Lab 4
// James Pfleger, Hemantha Akkaraju, Anya Biryukova
// Based on: "Matrix Multiplication with CUDA — A basic introduction to the CUDA programming model"
// NOTE: We did not get a chance to compare this against the Program 2 kernels

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h> 
#include <math.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <time.h> 

#include "bebycuda.h"

// Threadblock size
#define BLOCK_SIZE 16

__global__ void optimizedMatrixMultKernel(const float* A, const float* B, float* C, const unsigned int dimension);

// Optimized matrix multiplication helper function
void optimizedMMcuda(const float* A, const float* B, float* C, const unsigned int dimension) {
	// Load A and B to device memory
	float* d_A;
	size_t mem_size = dimension * dimension * sizeof(float);
	cudaMalloc((void**)&d_A, mem_size);
	cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);

	float* d_B;
	cudaMalloc((void**)&d_B, mem_size);
	cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice);

	// Allocate C in device memory
	float* d_C;
	cudaMalloc((void**)&d_C, mem_size);
	cudaMemcpy(d_C, C, mem_size, cudaMemcpyHostToDevice);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(dimension / dimBlock.x, dimension / dimBlock.y);
	optimizedMatrixMultKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C, dimension);
	cudaThreadSynchronize();

	// Read C from device memory
	cudaMemcpy(C, d_C, mem_size, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

// Optimized matrix multiplication kernel code using shared memory
__global__ void optimizedMatrixMultKernel(const float* A, const float* B, float* C, const unsigned int dimension) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each threadblock computes one sub-matrix of C
	float* Csub = &C[dimension * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];
	float temp_Value = 0.0f;

	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Loop over the sub-matrices of A and B that are required to compute Csub
	// Multiply each pair of sub-matrices together and accumulate the results
	for (int i = 0; i < (dimension / BLOCK_SIZE); i++) {
		// Get sub-matrix of A
		const float* Asub = &A[dimension * BLOCK_SIZE * blockRow + BLOCK_SIZE * i];

		// Get sub-matrix of B
		const float* Bsub = &B[dimension * BLOCK_SIZE * i + BLOCK_SIZE * blockCol];

		// Shared memory used to store Asub and Bsub respectively
		__shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

		// Each thread loads one element of each sub-matrix
		sharedA[row][col] = Asub[row * dimension + col];
		sharedB[row][col] = Bsub[row * dimension + col];

		// Wait for all the submatrices to be loaded in
		__syncthreads();

		// Multiply Asub and Bsub together
		for (int subi = 0; subi < BLOCK_SIZE; subi++) {
			temp_Value += sharedA[row][subi] * sharedB[subi][col];
		}

		// Wait for all threads to finish calculating the current submatrices
		__syncthreads();
	}

	// Write Csub to device memory
	Csub[row * dimension + col] = temp_Value;
}

/*int main(int argc, char* argv[]) {
	// Use current time as seed for random generator 
	srand(time(0));

	float* A;
	float* B;
	float* C;
	int n = 16 * 10;  // n x n matrix
	size_t mem_size = n * n * sizeof(float);

	A = (float*)malloc(mem_size);
	B = (float*)malloc(mem_size);
	C = (float*)malloc(mem_size);

	for (int i = 0; i < n * n; i++) {
		A[i] = rand() % (10 + 1);
		B[i] = rand() % (10 + 1);
		C[i] = 0.0f;
	}

	optimizedMMcuda(A, B, C, n);
}*/