
// This program queries the attached CUDA device(s), retrieving information about them 
// using the CUDA Runtime API and outputting everything to stdout

#include <stdio.h>

int getNumCudaCores(cudaDeviceProp devProp) {
    int numSMs = devProp.multiProcessorCount;
    int numCudaCores = 0;
    switch (devProp.major) {
     case 2: // Fermi
      if (devProp.minor == 1) numCudaCores = numSMs * 48;
      else numCudaCores = numSMs * 32;
      break;
     case 3: // Kepler
      numCudaCores = numSMs * 192;
      break;
     case 5: // Maxwell
      numCudaCores = numSMs * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) numCudaCores = numSMs * 128;
      else if (devProp.minor == 0) numCudaCores = numSMs * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) numCudaCores = numSMs * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) numCudaCores = numSMs * 64;
      else if (devProp.minor == 6) numCudaCores = numSMs * 128;
      else if (devProp.minor == 9) numCudaCores = numSMs * 128; // Ada Lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) numCudaCores = numSMs * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n");
      break;
    }
    return numCudaCores;
}

void printDevProp(cudaDeviceProp devProp)
{
    printf("  Device name:                   %s\n", devProp.name);
    printf("  Major revision number:         %d\n", devProp.major);
    printf("  Minor revision number:         %d\n", devProp.minor);
    printf("  Total global memory:           %.2f GB\n", devProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Total shared memory per block: %.0f KB\n", devProp.sharedMemPerBlock / 1024.0);
    printf("  Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("  Warp size (threads):           %d\n", devProp.warpSize);
    printf("  Maximum memory pitch:          %.0f MB\n", devProp.memPitch / 1024.0 / 1024.0);
    printf("  Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("  Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("  Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("  Core clock rate:               %.2f MHz\n", devProp.clockRate / 1000.0);
    printf("  Total constant memory:         %.0f KB\n", devProp.totalConstMem / 1024.0);
    printf("  Texture alignment:             %lu bytes\n", devProp.textureAlignment);
    printf("  Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("  Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("  Memory clock rate:             %.4f GHz\n", devProp.memoryClockRate / 1.0e6);
    printf("  Memory bus width:              %lu bits\n", devProp.memoryBusWidth); 
    printf("  Peak Memory Bandwidth:         %.4f GB/s\n", 2.0 * devProp.memoryClockRate * (devProp.memoryBusWidth / 8) / 1.0e6); 
    printf("  Number of SMs:                 %d\n", devProp.multiProcessorCount);
    printf("  Number of CUDA cores:          %i\n\n", getNumCudaCores(devProp));
}

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Querying CUDA device(s) ...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
    return 0;
}