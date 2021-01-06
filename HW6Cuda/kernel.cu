
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void convKernel(float* output, float* inputImage, float* filter, int imageWidth, int imageHeight, int filterWidth) {
    

    // __shared__ float sharedFilter[filterWidth][filterWidth];
    
    // this array shift right and shift down "half filter size" pixel
    // __shared__ float sharedInput[ 16*16 ];
    
    // int col = threadIdx.x, row = threadIdx.y;
    // int maxWidth = 10+6;
    // int idx = (row * maxWidth + col )*3;

    int rowidx = blockIdx.y * blockDim.y + threadIdx.y;
    int colidx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    int half_filter_size = filterWidth / 2;
    // int imageSize = imageHeight * imageWidth;
    // int sidx = (rowidx - half_filter_size) * imageWidth + colidx - half_filter_size + idx;
    // if (sidx >= 0 && sidx < imageSize)
    //     sharedInput[ idx ] = inputImage[ sidx ];
    // if (sidx + 1 >= 0 && sidx+1 < imageSize)
    //     sharedInput[ idx+1 ] = inputImage[ sidx+1 ];
    // if (sidx + 2 >= 0 && sidx+2 < imageSize)
    //     sharedInput[ idx+2 ] = inputImage[ sidx+2 ];
    
    // if ( row <  filterWidth ) {
    //     sharedInput[row + 8][col] = inputImage[ (rowidx + 8) * imageWidth + colidx ];
    // }
    // if ( col < filterWidth) {
    //     sharedInput[row][col + 8] = inputImage[ (rowidx) * imageWidth + colidx + 8];
    // }

    // __syncthreads();

    for (int k = -half_filter_size,  fi = 0; k<=half_filter_size; k++) {

        for(int l = -half_filter_size; l<=half_filter_size; fi++, l++) {

            if ( rowidx + k >= 0 && rowidx + k < imageHeight &&
                 colidx + l >= 0 && colidx + l < imageWidth ) 
            {
                sum += inputImage[ (rowidx + k) * imageWidth + colidx + l ] * filter[fi];
                // sum += sharedInput[ k + half_filter_size + row][ l + half_filter_size + col] * filter[fi];
                // int xidx = (k + half_filter_size + row) * 16 + l + half_filter_size + col;
                // sum += sharedInput[ xidx ] * filter[fi];

            }
        }
    }
    // __syncthreads();

    output[ rowidx * imageWidth + colidx] = sum;

}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
    float *inputImage, float *outputImage) {

    int resX = imageWidth;
    int resY = imageHeight;

    float* cuda_output;
    float* cuda_filter;
    float* cuda_input;
	float* output = (float*)malloc( resX * resY * sizeof(float) );
    cudaMalloc(&cuda_output, resX * resY * sizeof(float) );
	cudaMalloc(&cuda_filter, filterWidth * filterWidth * sizeof(float) );
	cudaMalloc(&cuda_input, resX * resY * sizeof(float) );
    
    cudaMemcpy(cuda_filter, filter, filterWidth * filterWidth * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_input, inputImage, resX * resY * sizeof(float) , cudaMemcpyHostToDevice);

    int block_size_x = 10;

    dim3 blocksize(block_size_x, block_size_x);
	dim3 gridsize(resX / block_size_x, resY / block_size_x) ;

    convKernel<<<gridsize, blocksize>>>(cuda_output, cuda_input, cuda_filter, resX, resY, filterWidth );
    cudaMemcpy(output, cuda_output, resX * resY * sizeof(float) , cudaMemcpyDeviceToHost);
    memcpy(outputImage, output, resX*resY*sizeof(float));

	cudaFree(cuda_output);
	free(output);
        
}