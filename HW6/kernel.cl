__kernel void convolution(
    const __global float* inputImage, 
    __global float* outputImage,
    __constant float* filter,
    int imageWidth,
    int imageHeight,
    int filterWidth,
    int block_width
) 
{
    int rowidx = get_global_id(1);
    int colidx = get_global_id(0);

    if (rowidx >= imageHeight || colidx >= imageWidth ) {
        return;
    }
    int hfs = filterWidth / 2;

    __local float sharedInput[ 30][ 30 ];
    // __local int x1;
    // __local int y1;

    int col = get_local_id(0), row = get_local_id(1);
    sharedInput[row + hfs][col + hfs] = inputImage[ rowidx * imageWidth + colidx];

    // // four edge & corner
    if (row < hfs && (rowidx - hfs) >= 0) {
        sharedInput[row][col + hfs] = inputImage[ (rowidx - hfs)*imageWidth + (colidx) ];

        if (col < hfs && (colidx-hfs) >= 0) {
            sharedInput[row][col] = inputImage[ (rowidx - hfs)*imageWidth + (colidx - hfs) ];
        }
        else if (col >= (block_width - hfs) && (colidx + hfs) < imageWidth ) {
            sharedInput[row][col + (2 * hfs)] = inputImage[ (rowidx - hfs)*imageWidth + (colidx + hfs) ];
        }
    }
    if (col < hfs && (colidx-hfs) >= 0) {
        sharedInput[row + hfs][col] = inputImage[ (rowidx)*imageWidth + (colidx - hfs) ];
    }
    if (row >= (block_width - hfs) && (rowidx + hfs) < imageHeight) {
        sharedInput[row + (2*hfs)][col + hfs] = inputImage[ (rowidx + hfs)*imageWidth + (colidx) ];
        
        if (col < hfs && (colidx-hfs) >= 0) {
            sharedInput[row + (2*hfs)][ col ] = inputImage[ (rowidx + hfs)*imageWidth + (colidx - hfs) ];
        }else if (col >= block_width - hfs && (colidx + hfs) < imageWidth) {
            sharedInput[row + (2*hfs)][ col + (2*hfs) ] = inputImage[ (rowidx + hfs)*imageWidth + (colidx + hfs) ];
        }
    }
    if (col >= (block_width - hfs) && (colidx + hfs) < imageWidth) {
        sharedInput[row + hfs][col + (2*hfs)] = inputImage[ (rowidx)*imageWidth + (colidx + hfs) ];
    }

    const int bound = block_width + hfs*2;


    float sum = 0;

    for (int k = -hfs,  fi = 0; k<=hfs; k++) {
        for(int l = -hfs; l<=hfs; fi++, l++) {

            if ( rowidx + k >= 0 && rowidx + k < imageHeight &&
                 colidx + l >= 0 && colidx + l < imageWidth ) 
            {
                int xx = k + hfs + row;
                int yy = l + hfs + col;
                // sum += sharedInput[ xx * bound +  yy ] * filter[fi];
                sum += sharedInput[xx][yy] * filter[fi];
            }
        }
    }

    outputImage[ rowidx * imageWidth + colidx] = sum;
}