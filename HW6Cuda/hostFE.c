#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
    
    cl_mem input_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize*sizeof(float), NULL, NULL);
    cl_mem output_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize*sizeof(float), NULL, NULL);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize*sizeof(float), NULL, NULL);


    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);
    clEnqueueWriteBuffer(command_queue, input_mem, CL_TRUE, 0, imageSize*sizeof(float), inputImage, 0, NULL, NULL);
    // clEnqueueWriteBuffer(command_queue, output_mem, CL_TRUE, 0, imageSize*sizeof(int), outputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0, filterSize*sizeof(float), filter, 0, NULL, NULL);


    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&filter_mem);

    clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);

    size_t local_work_size[2] = {4, 4};
    size_t global_work_size[2] = {imageWidth, imageHeight};

    clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, global_work_size, local_work_size, 0, NULL, NULL);

    clEnqueueReadBuffer(command_queue, output_mem, CL_TRUE, 0, imageSize*sizeof(float), outputImage, 0, NULL, NULL);

    // Release resource
    // clFlush(command_queue);
    // clFinish(command_queue);
    // clReleaseKernel(kernel);
    // clReleaseProgram(*program);
    // clReleaseMemObject(input_mem);
    // clReleaseMemObject(output_mem);
    // clReleaseMemObject(filter_mem);
    // clReleaseCommandQueue(command_queue);
    // clReleaseContext(*context);


}