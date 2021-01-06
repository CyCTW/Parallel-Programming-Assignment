__kernel void convolution(
    const __global float* inputImage, 
    __global float* outputImage,
    __constant float* filter,
    int imageWidth,
    int imageHeight,
    int filterWidth,
    int block_size
) 
{
    int half_filter_size = filterWidth / 2;
    // int block_size = 8;
    const int bound = block_size + half_filter_size*2;

    __local float sharedInput[ 20 * 20 ];
    __local int x1;
    __local int y1;

    int col = get_local_id(0), row = get_local_id(1);
    // idx: 0~300
    int idx = (row * block_size + col ) * 3;

    int rowidx = get_global_id(1);
    int colidx = get_global_id(0);
    
    if (row == 0 && col == 0) {
        x1 = rowidx - half_filter_size;
        y1 = colidx - half_filter_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

    // r = x1 + ( (idx+3) / bound);
    // c = y1 + ( (idx+3) % bound);
    // if ( r >=0 && r < imageHeight && c >= 0 && c < imageWidth )
    //     sharedInput[ idx+3 ] = inputImage[ (r) * imageWidth + (c)  ];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

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
    barrier(CLK_LOCAL_MEM_FENCE);

    outputImage[ rowidx * imageWidth + colidx] = sum;
}
