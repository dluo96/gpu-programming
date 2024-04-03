#include <stdio.h>
#include <stdlib.h>

// CUDA kernel to perform SAXPY operation: Y := a*X + Y
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; // Calculate global index for each thread
  if (i < n) y[i] = a*x[i] + y[i]; // Perform SAXPY operation if within bounds
}

int main(void)
{
  int N = 1<<20; // Number of elements in arrays, 1<<20 represents 2^20
  float *x, *y, *d_x, *d_y; // Pointers for host (x, y) and device (d_x, d_y) arrays

  // Allocate memory on the host
  x = (float*)malloc(N*sizeof(float)); 
  y = (float*)malloc(N*sizeof(float));

  // Allocate memory on the device
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  // Initialize arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f; // Set all elements of x to 1.0f
    y[i] = 2.0f; // Set all elements of y to 2.0f
  }

  // Copy arrays from host to device
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Launch the SAXPY kernel
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y); // (N+255)/256 calculates the number of blocks needed

  // Copy the result back to the host
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

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
