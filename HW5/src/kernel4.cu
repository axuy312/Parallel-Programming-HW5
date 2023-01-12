#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int* device_img, int width, float stepX, float stepY, float lowerX, float lowerY, int count) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
	
	int thisX = threadIdx.x, thisY = blockIdx.x;
	int idx = thisX + thisY * 1600;
		
	float c_im = lowerY + thisY * stepY;
	for(; thisX < 1600; thisX += 128, idx += 128)
	{
			
		float c_re = lowerX + thisX * stepX;
		
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

		device_img[idx] = i;
	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);
	int *device_img;
    cudaMalloc(&device_img, size);

    int numThreads = 128;
    int numBlocks = resY;
    mandelKernel<<<numBlocks, numThreads>>>(device_img, resX, stepX, stepY, lowerX, lowerY, maxIterations);
	
    cudaMemcpy(img, device_img, size, cudaMemcpyDeviceToHost);
	
    cudaFree(device_img);
}
