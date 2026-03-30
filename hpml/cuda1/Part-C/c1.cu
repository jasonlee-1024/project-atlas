// Includes
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "timer.h"

// Variables for host and device vectors.
double* h_I0; 
double* h_F; 
double* h_O; 
double* d_I0; 
double* d_F; 
double* d_O; 

#define C 3
#define H 1024
#define W 1024
#define K 64
#define FH 3
#define FW 3

/*
    I0: 3 x 1026 x 1026
    F: 64 x 3 x 3 x 3
    O: 64 x 1024 x 1024
*/

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

__global__ void CalculateResults(const double* I0, const double* F, double* O);

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    size_t size_i0 = C * (H + 2) * (W + 2) * sizeof(double);
    size_t size_f = K * C * FH * FW * sizeof(double);
    size_t size_o = K * H * W * sizeof(double);
                
    
    // Allocate input vectors h_I0 and h_F in host memory
    h_I0 = (double*)malloc(size_i0);
    if (h_I0 == 0) Cleanup(false);
    h_F = (double*)malloc(size_f);
    if (h_F == 0) Cleanup(false);
    h_O = (double*)malloc(size_o);
    if (h_O == 0) Cleanup(false);
    
    // Allocate vectors in device memory.
    cudaError_t error;
    error = cudaMalloc((void**)&d_I0, size_i0);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_F, size_f);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_O, size_o);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize host vectors h_I0 and h_F
    for(int i=0; i<C; ++i){
        for (int j=0; j<H+2; ++j){
            for (int k=0; k<W+2; ++k){
                if (j==0 || j==H+1 || k==0 || k==W+1){
                    h_I0[i*(H+2)*(W+2)+j*(W+2)+k] = 0.0;
                } else {
                    h_I0[i*(H+2)*(W+2)+j*(W+2)+k] = (double)(i*(k-1+j-1));
                }
            }
        }
    }

    for (int i=0; i<K; ++i){
        for (int j=0; j<C; ++j){
            for (int k=0; k<FH; ++k){
                for (int l=0; l<FW; ++l){
                    h_F[i*C*FH*FW+j*FH*FW+k*FW+l] = (double)(i+j)*(k+l);
                }
            }
        }
    }

    // Copy host vectors h_I0 and h_F to device vectores d_I0 and d_F
    error = cudaMemcpy(d_I0, h_I0, size_i0, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_F, h_F, size_f, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    // Tell CUDA how big to make the grid and thread blocks.
    // Since this is a vector addition problem,
    // grid and thread block are both one-dimensional.
    dim3 dimGrid(W, K);                    
    dim3 dimBlock(H);             

    // Warm up
    CalculateResults<<<dimGrid, dimBlock>>>(d_I0, d_F, d_O);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
        Cleanup(false);
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(error));
        Cleanup(false);
    }

    // Initialize timer  
    initialize_timer();
    start_timer();

    CalculateResults<<<dimGrid, dimBlock>>>(d_I0, d_F, d_O);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
        Cleanup(false);
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(error));
        Cleanup(false);
    }
    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

	// Report timing data.
    printf("Time: %.3f ms\n", time*1000.0);
     
    // Copy result from device memory to host memory
    error = cudaMemcpy(h_O, d_O, size_o, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);

    // Verify & report result
    double checksum = 0.0;
    for (int i = 0; i < K * H * W; ++i) {
        checksum += h_O[i];
    }

    printf("Checksum: %lf\n", checksum);

    // Clean up and exit.
    Cleanup(true);
}



__global__ void CalculateResults(const double* I0, const double* F, double* O)
{
    int k = blockIdx.y;
    int x = blockIdx.x;
    int y = threadIdx.x;

    double Ovalue = 0.0;

    for (int c=0; c<C; ++c){
        for(int i=0; i< FH; ++i) {
            for (int j=0; j< FW; ++j) {
                int f_index = k * C * FH * FW + c * FH * FW + (2-i)*FW + 2-j;
                int i0_index = c * (H + 2) * (W + 2) + (x + i) * (H + 2) + (y + j);
                Ovalue += F[f_index] * I0[i0_index];
            }
        }
    }
    O[k*H*W + x*H + y] = Ovalue;
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_I0)
        cudaFree(d_I0);
    if (d_F)
        cudaFree(d_F);
    if (d_O)
        cudaFree(d_O);

    // Free host memory
    if (h_I0)
        free(h_I0);
    if (h_F)
        free(h_F);
    if (h_O)
        free(h_O);
        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}


