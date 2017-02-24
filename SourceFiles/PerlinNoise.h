#ifndef PERLIN_NOISE
#define PERLIN_NOISE

#include <functional>

#include "VoronoiDiagram.h"

std::vector<std::vector<Vector2d>> getGradientField(unsigned int, unsigned int);
std::vector<std::vector<Vector2d>> getCoord(unsigned int, unsigned int, double);

void noise(VoronoiDiagram::VoronoiGraph&, std::function<double (double, double, double)>, double, double);
void perlinNoise(VoronoiDiagram::VoronoiGraph&, std::function<double (double, double, double)>, double, unsigned int, double);
void computeAverageSiteElevation(VoronoiDiagram::VoronoiGraph&);
void computeMap(VoronoiDiagram::VoronoiGraph&, std::function<double (double, double, double)>, double, unsigned int, double);

double linearInterpolate(double, double, double);
double quinticInterpolate(double, double, double);

struct GradientParameters {
	GradientParameters(sf::Color, sf::Color, double);
	sf::Color low;
	sf::Color high;
	double ratio;
};

std::function<sf::Color (double)> linearGradient(const std::vector<GradientParameters>&);
std::function<sf::Color (double)> groundGradient();
std::function<sf::Color (double)> islandGradient();

void makeIsland(VoronoiDiagram::VoronoiGraph&);

namespace CudaPerlinNoise {
	void PerlinNoise(VoronoiDiagram::VoronoiGraph&, double, unsigned int, double);
	void computeParallelMap(VoronoiDiagram::VoronoiGraph&, double, unsigned int, double);
}

#endif
