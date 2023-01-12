#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int* device_img, int width, float stepX, float stepY, float lowerX, float lowerY, int count, int pitch, int numPixel) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
	
	for (int j = 0; j < numPixel; ++j)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = idx * numPixel + j;
		int thisX = idx % width;
		int thisY = idx / width;
			
		float c_re = lowerX + thisX * stepX;
		float c_im = lowerY + thisY * stepY;
		
		float z_re = c_re, z_im = c_im;
		int i;
		for (i = 0; i < count; ++i)
		{
			float z_re2 = z_re * z_re;
			float z_im2 = z_im * z_im;
			if (z_re2 + z_im2 > 4.f)
				break;

			float new_re = z_re2 - z_im2;
			float new_im = 2.f * z_re * z_im;
			z_re = c_re + new_re;
			z_im = c_im + new_im;
		}

		int* row = (int*)((char*)device_img + thisY * pitch);
		row[thisX] = i;
	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);
    size_t pitch = 0;
	
    int *host_img;
    cudaHostAlloc(&host_img, size, cudaHostAllocMapped);
	
	int *device_img;
    cudaMallocPitch(&device_img, &pitch, resX * sizeof(int), resY);

	int numPixel = 4;
    int numThreads = 400;
    int numBlocks = resX * resY / numThreads / numPixel;
    mandelKernel<<<numBlocks, numThreads>>>(device_img, resX, stepX, stepY, lowerX, lowerY, maxIterations, pitch, numPixel);
	
    cudaMemcpy2D(host_img, resX * sizeof(int), device_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
	memcpy(img, host_img, size);
	
    cudaFree(device_img);
    cudaFreeHost(host_img);
}
