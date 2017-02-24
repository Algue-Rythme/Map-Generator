#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include "PerlinNoise.cuh"

__device__
unsigned int hash1(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__device__
unsigned int hash2(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__
unsigned int hashPaire(unsigned int a, unsigned int b) {
	return (a >= b)? a*a + a + b : a+ b*b;
}

__device__
float dot(float2 v1, float2 v2) {
	return v1.x*v2.x + v1.y*v2.y;
}

__device__
float norm2(float2 v) {
	return sqrt(dot(v, v));
}

__device__
float distance2(float2 v1, float2 v2) {
	return norm2(make_float2(v1.x - v2.x, v1.y - v2.y));
}

__device__
float2 normalize(float2 v) {
	float n = norm2(v);
	if (n <= 1e-7f)
		return v;
	v.x /= n;
	v.y /= n;
	return v;
}

__device__ 
float2 getGradient(unsigned int xCell, unsigned int yCell) {
	unsigned int hashX = hash1(hashPaire(xCell, yCell));
	unsigned int hashY = hash1(hashPaire(yCell, hashX%(1 << 16)));
	float x = float(hashX % (1 << 10)) - (1 << 9);
	float y = float(hashY % (1 << 10)) - (1 << 9);
	return normalize(make_float2(x, y));
}

__device__
float interpolate(float t, float a, float b) {
	t = t*t*t*(6*t*t-15*t+10);
	return a + t*(b-a);
}

__global__ 
void CUDA_noise(
	float width,
	float height,
	unsigned int nbNodes,
	float* d_x,
	float* d_y, 
	float* d_z, 
	float frequency,
	float amplitude) {

	const unsigned int node = blockIdx.x*blockDim.x + threadIdx.x;
	if (node >= nbNodes)
		return ;

	const float squareSize = height/frequency;

	float x = d_x[node];
	if (x < 0.f) x = squareSize/2;
	if (x >= width) x = width - squareSize/2;

	float y = d_y[node];
	if (y < 0.f) y = squareSize / 2;
	if (y >= height) y = height - squareSize/2;

	unsigned int xCell = x/squareSize;
	unsigned int yCell = y/squareSize;

	const unsigned int directions[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	float values[4];
	for (unsigned int dir = 0; dir < 4; ++dir) {
		unsigned int xCorner = xCell+directions[dir][0];
		unsigned int yCorner = yCell+directions[dir][1];
		float2 gradient = getGradient(xCorner, yCorner);
		float2 pos = make_float2(xCorner*squareSize, yCorner*squareSize);
		float2 arrow = make_float2(pos.x - x, pos.y - y);
		values[dir] = dot(normalize(arrow), gradient);
	}

	float xReal = (x - xCell*squareSize) / squareSize;
	float y1 = interpolate(xReal, values[0], values[2]);
	float y2 = interpolate(xReal, values[1], values[3]);

	float yReal = (y - yCell*squareSize) / squareSize;
	float z = interpolate(yReal, y1, y2);
	d_z[node] += amplitude*z;
}

__host__
void CUDA_PerlinNoise(
	float width, 
	float height, 
	unsigned int nbNodes, 
	float* h_x, 
	float* h_y, 
	float* h_z,
	float fundamental, 
	unsigned int nbOctaves, 
	float persistence) {
	const unsigned int nbBytes = sizeof(float)*nbNodes;

	float* d_x;
	cudaMalloc(&d_x, nbBytes);
	cudaMemcpy(d_x, h_x, nbBytes, cudaMemcpyHostToDevice);

	float* d_y;
	cudaMalloc(&d_y, nbBytes);
	cudaMemcpy(d_y, h_y, nbBytes, cudaMemcpyHostToDevice);

	float* d_z;
	cudaMalloc(&d_z, nbBytes);
	cudaMemset(d_z, 0, nbBytes);

	float frequency = fundamental;
	float amplitude = 1.;
	float amplitudeMax = 0.;
	for (unsigned int octave = 0; octave < nbOctaves; ++octave) {
		amplitudeMax += amplitude;
		const unsigned int blockSize = 128;
		const dim3 gridSize((nbNodes+blockSize-1)/blockSize);
		CUDA_noise<<<gridSize, blockSize>>>(width, height, nbNodes, d_x, d_y, d_z, frequency, amplitude);
		frequency *= 2.5;
		amplitude *= persistence;
	}

	cudaMemcpy(h_z, d_z, nbBytes, cudaMemcpyDeviceToHost);
	for (unsigned int node = 0; node < nbNodes; ++node) {
		h_z[node] /= amplitudeMax;
		h_z[node] = (h_z[node]+1)/2;
	}

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
}
