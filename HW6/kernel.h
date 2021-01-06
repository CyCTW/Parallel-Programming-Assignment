#ifndef KERNEL_H_
#define KERNEL_H_

//extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
    float *inputImage, float *outputImage);

#endif /* KERNEL_H_ */
