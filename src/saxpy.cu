#include <stdio.h>
#include <stdlib.h>

// CUDA kernel to perform SAXPY operation: Y := a*X + Y
__global__
void saxpy(int n, float a, float *x, float *y)
{
  // Calculate global index for each thread
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  // Perform SAXPY operation if within bounds
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  // Set array dimension to 2^20
  int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  // Allocate host memory
  float *x, *y;
  x = (float*)malloc(bytes); 
  y = (float*)malloc(bytes);

  // Allocate device memory
  float *d_x, *d_y;
  cudaMalloc(&d_x, bytes); 
  cudaMalloc(&d_y, bytes);

  // Initialize arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  } 

  // Copy arrays from host to device
  cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

  // Set number of threads per block and number of blocks
  int threadsPerBlock = 256;
  int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Invoke the SAXPY kernel
  saxpy<<<numBlocks, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

  // Copy the result back to the host
  cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

  // Compare results to expected result
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  // Free allocated memory on device
  cudaFree(d_x);
  cudaFree(d_y);

  // Free allocated memory on host
  free(x);
  free(y);
}
