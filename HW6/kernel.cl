__kernel void convolution(
    const __global float* inputImage, 
    __global float* outputImage,
    __constant float* filter,
    int imageWidth,
    int imageHeight,
    int filterWidth
) 
{
    int rowidx = get_global_id(1);
    int colidx = get_global_id(0);

    float sum = 0;
    int half_filter_size = filterWidth / 2;

    for (int k = -half_filter_size,  fi = 0; k<=half_filter_size; k++) {
        for(int l = -half_filter_size; l<=half_filter_size; fi++, l++) {

            if ( rowidx + k >= 0 && rowidx + k < imageHeight &&
                 colidx + l >= 0 && colidx + l < imageWidth ) 
            {
                sum += inputImage[ (rowidx + k) * imageWidth + colidx + l ] * filter[fi];
                // filter[ (k + half_filter_size) * filterWidth + l + half_filter_size];
            }
        }
    }

    outputImage[ rowidx * imageWidth + colidx] = sum;
}
