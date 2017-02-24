#ifndef VORONOI_DIAGRAM_CUDA
#define VORONOI_DIAGRAM_CUDA

namespace CUDA_VoronoiDiagram {

void naiveParallelGeneration
	(unsigned int, unsigned int, unsigned int* const, unsigned int, 
	const unsigned int* const, const unsigned int* const);

}

#endif