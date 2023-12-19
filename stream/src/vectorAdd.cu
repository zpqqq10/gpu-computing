/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /**
  * Vector addition: C = A + B.
  *
  * This sample is a very basic sample that implements element by element
  * vector addition. It is the same as the sample illustrating Chapter 2
  * of the programming guide with some additions like error checking.
  */

#include <stdio.h>

  // For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include "book.h"
#include <omp.h>

#define START_GPU \
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));


#define START_CPU {\
double start = omp_get_wtime();

#define END_CPU \
double end = omp_get_wtime();\
double duration = end - start;\
printf("CPU Time used: %3.1f ms\n", duration * 1000);}

#define STREAMNUM 2
#define N 1024*1024

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const double* A, const double* B, double* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = cos(A[i]) / sin(B[i]);
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    cudaDeviceProp prop; 
    int whichDevice; 
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if(!prop.deviceOverlap){
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaStream_t streams[STREAMNUM];
    double *d_A[STREAMNUM];
    double *d_B[STREAMNUM];
    double *d_C[STREAMNUM];

    // Print the vector length to be used, and compute its size
    int numElements = 100*N;
    size_t size = numElements * sizeof(double);
    printf("[Using %d streams]\n", STREAMNUM);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    double* h_A = NULL;
    HANDLE_ERROR(cudaHostAlloc((void**)&h_A, 
                        size, 
                        cudaHostAllocDefault));

    // Allocate the host input vector B
    double* h_B = NULL;
    HANDLE_ERROR(cudaHostAlloc((void**)&h_B, 
                        size, 
                        cudaHostAllocDefault));

    // Allocate the host output vector C
    double* h_C = NULL;
    HANDLE_ERROR(cudaHostAlloc((void**)&h_C, 
                        size, 
                        cudaHostAllocDefault));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (double)RAND_MAX;
        h_B[i] = rand() / (double)RAND_MAX;
    }

    // Allocate the device input vectors
    for(int i=0; i<STREAMNUM; i++){
        HANDLE_ERROR(cudaStreamCreate(&streams[i]));
        HANDLE_ERROR(cudaMalloc((void**)&d_A[i], N*sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&d_B[i], N*sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&d_C[i], N*sizeof(double)));
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    int threadsPerBlock = 256;
    START_GPU
    for(int i=0; i<numElements; i+=N*STREAMNUM){
        for(int j=0; j<STREAMNUM; j++){
            if(i+j*N < numElements){
                HANDLE_ERROR(cudaMemcpyAsync(d_A[j], h_A+i+j*N, N*sizeof(double), cudaMemcpyHostToDevice, streams[j]));
                // depth-first
                // HANDLE_ERROR(cudaMemcpyAsync(d_B[j], h_B+i+j*N, N*sizeof(double), cudaMemcpyHostToDevice, streams[j]));
                // vectorAdd <<<N/threadsPerBlock, threadsPerBlock, 0, streams[j] >>> (d_A[j], d_B[j], d_C[j], numElements);
                // HANDLE_ERROR(cudaMemcpyAsync(h_C+i+j*N, d_C[j], N*sizeof(double), cudaMemcpyDeviceToHost, streams[j]));
            }
            else {
                break;
            }
        }
        for(int j=0; j<STREAMNUM; j++){
            if(i+j*N < numElements){
                HANDLE_ERROR(cudaMemcpyAsync(d_B[j], h_B+i+j*N, N*sizeof(double), cudaMemcpyHostToDevice, streams[j]));
            }
            else {
                break;
            }
        }
        // Launch the Vector Add CUDA Kernel
        for(int j=0; j<STREAMNUM; j++){
            if(i+j*N < numElements){
                vectorAdd <<<N/threadsPerBlock, threadsPerBlock, 0, streams[j] >>> (d_A[j], d_B[j], d_C[j], numElements);
            }
            else {
                break;
            }
        }
        for(int j=0; j<STREAMNUM; j++){
            if(i+j*N < numElements){
                HANDLE_ERROR(cudaMemcpyAsync(h_C+i+j*N, d_C[j], N*sizeof(double), cudaMemcpyDeviceToHost, streams[j]));
            }
            else {
                break;
            }
        }
    }

    for(int i=0; i<STREAMNUM; i++){
        HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
    }
    END_GPU

    START_CPU
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(cos(h_A[i])/sin( h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    END_CPU

    printf("Test PASSED\n");

    // Free device global memory
    for(int i=0; i<STREAMNUM; i++){
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));
        HANDLE_ERROR(cudaFree(d_A[i]));
        HANDLE_ERROR(cudaFree(d_B[i]));
        HANDLE_ERROR(cudaFree(d_C[i]));
    }

    // Free host memory
    HANDLE_ERROR(cudaFreeHost(h_A));
    HANDLE_ERROR(cudaFreeHost(h_B));
    HANDLE_ERROR(cudaFreeHost(h_C));

    printf("Done\n");
    return 0;
}

