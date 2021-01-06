#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(int* output, size_t pitch, int width, int height, float lowerX, float lowerY, float stepX, float stepY, 
    int maxIterations, int group_size) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

  int i = (blockIdx.x * blockDim.x + threadIdx.x) * group_size;
  int j = (blockIdx.y * blockDim.y + threadIdx.y) * group_size;

  for(int gi=0; gi < group_size; gi++) {
    for(int gj=0; gj < group_size; gj++) {
        float x = lowerX + (gi + i) * stepX;
        float y = lowerY + (gj + j) * stepY;

        int *ele = (int*)((char*)output + (gj + j) * pitch) + ( gi + i  );
        *ele = mandel(x, y, maxIterations);	
    }
  }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;
	
  int* cuda_output;
  int* output;
   cudaHostAlloc((void**)&output, resX * resY * sizeof(int) , cudaHostAllocDefault) ;
  size_t pitch;
  cudaMallocPitch((void**)&cuda_output, &pitch, resX * sizeof(int), resY);

  int block_size = 8;
  int group_size = 2; // 1 thread process 4 pixel
  dim3 blocksize(block_size, block_size);
  dim3 gridsize(resX / block_size / group_size, resY / block_size / group_size);

  mandelKernel<<<gridsize, blocksize>>>(cuda_output, pitch, resX, resY, lowerX, lowerY, stepX, stepY, maxIterations, group_size );
  cudaMemcpy2D(output, resX * sizeof(int), cuda_output, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

  memcpy(img, output, resX*resY*sizeof(int));

  cudaFree(cuda_output);
  cudaFreeHost(output);
}
