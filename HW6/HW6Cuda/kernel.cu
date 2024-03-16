
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void convKernel(float* output, float* inputImage, float* filter, int imageWidth, int imageHeight, int filterWidth) {
    

    // this array shift right and shift down "half filter size" pixel
    int half_filter_size = filterWidth / 2;
    int block_size = 8;
    const int bound = block_size + half_filter_size*2;

    __shared__ float sharedInput[ 20 * 20 ];
    __shared__ int x1;
    __shared__ int y1;

    

    int col = threadIdx.x, row = threadIdx.y;
    // idx: 0~300
    int idx = (row * block_size + col ) * 4;
    int rowidx = blockIdx.y * blockDim.y + threadIdx.y;
    int colidx = blockIdx.x * blockDim.x + threadIdx.x;

    // first element store head position
    if (row == 0 && col == 0) {
        x1 = rowidx - half_filter_size;
        y1 = colidx - half_filter_size;
    }
    __syncthreads();

    float sum = 0;
    // int imageSize = imageHeight * imageWidth;

    // store image to local 
    int r = x1 + ( idx / bound);
    int c = y1 + ( idx % bound);
    if ( r >=0 && r < imageHeight && c >= 0 && c < imageWidth )
        sharedInput[ idx   ] = inputImage[ (r) * imageWidth + (c)     ];

    r = x1 + ( (idx+1) / bound);
    c = y1 + ( (idx+1) % bound);
    if ( r >=0 && r < imageHeight && c >= 0 && c < imageWidth )
        sharedInput[ idx+1 ] = inputImage[ (r) * imageWidth + (c)  ];

    r = x1 + ( (idx+2) / bound);
    c = y1 + ( (idx+2) % bound);
    if ( r >=0 && r < imageHeight && c >= 0 && c < imageWidth )
        sharedInput[ idx+2 ] = inputImage[ (r) * imageWidth + (c)  ];
    r = x1 + ( (idx+3) / bound);
    c = y1 + ( (idx+3) % bound);
    if ( r >=0 && r < imageHeight && c >= 0 && c < imageWidth )
        sharedInput[ idx+3 ] = inputImage[ (r) * imageWidth + (c)  ];

    // if (row == 0 && col == 0) {
    //     // load image
    //     for(int i=0; i<bound; i++) {
    //         for(int j=0; j<bound; j++) {
    //             int r = rowidx - half_filter_size + i;
    //             int c = colidx - half_filter_size + j;
    //             if ( r >=0 && r < imageHeight && c >= 0 && c < imageWidth ) {
    //                 sharedInput[i * bound + j] = inputImage[ (r) * imageWidth + (c) ];
    //             }
    //         }
    //     }
    // }
    __syncthreads();

    

    for (int k = -half_filter_size,  fi = 0; k<=half_filter_size; k++) {

        for(int l = -half_filter_size; l<=half_filter_size; fi++, l++) {

            if ( rowidx + k >= 0 && rowidx + k < imageHeight &&
                 colidx + l >= 0 && colidx + l < imageWidth ) 
            {


                int xx = k + half_filter_size + row;
                int yy = l + half_filter_size + col;
                sum += sharedInput[ xx * bound +  yy ] * filter[fi];

            }
        }
    }
    __syncthreads();

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

    int block_size_x = 8;

    dim3 blocksize(block_size_x, block_size_x);
	dim3 gridsize(resX / block_size_x, resY / block_size_x) ;

    convKernel<<<gridsize, blocksize>>>(cuda_output, cuda_input, cuda_filter, resX, resY, filterWidth );
    cudaMemcpy(output, cuda_output, resX * resY * sizeof(float) , cudaMemcpyDeviceToHost);
    memcpy(outputImage, output, resX*resY*sizeof(float));

	cudaFree(cuda_output);
	free(output);
        
}