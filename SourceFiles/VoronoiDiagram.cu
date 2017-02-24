#include "VoronoiDiagram.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

__device__
unsigned int distance2(unsigned int dx, unsigned int dy)
{
	return dx*dx + dy*dy;
}

__global__
void naiveGeneration
	(unsigned int width, unsigned int height,
	unsigned int * const d_index,
	unsigned int nbPoints,
	const unsigned int* const d_x,
	const unsigned int* const d_y)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int pixel = y*width + x;
	if (x >= width || y >= height)
		return ;
	
	unsigned int bestPoint = 0;
	unsigned int bestDistance = distance2(d_x[0]-x, d_y[0]-y); 
	for (unsigned int point = 1; point < nbPoints; ++point)
	{
		unsigned int distance = distance2(d_x[point]-x, d_y[point]-y);
		if (distance < bestDistance)
		{
			bestDistance = distance;
			bestPoint = point;
		}
	}

	d_index[pixel] = bestPoint;
}

void CUDA_VoronoiDiagram::naiveParallelGeneration
	(unsigned int width, unsigned int height, 
	unsigned int * const h_index, 
	unsigned int nbPoints, 
	const unsigned int* const h_x,
	const unsigned int* const h_y)
{
	const unsigned int nbBytesVertex = width*height*sizeof(unsigned int);
	const unsigned int nbBytesPoint = nbPoints*sizeof(unsigned int);

	unsigned int* d_index = nullptr;
	cudaMalloc(&d_index, nbBytesVertex);

	unsigned int* d_x = nullptr;
	cudaMalloc(&d_x, nbBytesPoint);
	cudaMemcpy(d_x, h_x, nbBytesPoint, cudaMemcpyHostToDevice);

	unsigned int* d_y = nullptr;
	cudaMalloc(&d_y, nbBytesPoint);
	cudaMemcpy(d_y, h_y, nbBytesPoint, cudaMemcpyHostToDevice);

	const dim3 blockSize(32, 32); 
	const dim3 gridSize((width+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
	naiveGeneration<<<gridSize, blockSize>>>(width, height, d_index, nbPoints, d_x, d_y);

	cudaMemcpy(h_index, d_index, nbBytesVertex, cudaMemcpyDeviceToHost);

	cudaFree(d_index);
	cudaFree(d_x);
	cudaFree(d_y);
}