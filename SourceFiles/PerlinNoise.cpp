#include <chrono>
#include <iostream>
#include <random>

#include "PerlinNoise.h"
#include "PerlinNoise.cuh"
#include "VoronoiDiagram.h"
#include "GeometricUtility.h"

using namespace std;
using namespace VoronoiDiagram;

namespace {
	const unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
	std::default_random_engine generator (seed);
	std::uniform_real_distribution<double> uniform(-1., 1.);
}

vector<vector<Vector2d>> getGradientField(unsigned int width, unsigned int height) {
	vector<vector<Vector2d>> gradientField(height, vector<Vector2d>(width));
	for (auto& line : gradientField) {
		for (auto& vect : line) {
			vect = normalize(Vector2d(uniform(generator), uniform(generator)));
		}
	}
	return gradientField;
}

vector<vector<Vector2d>> getCoord(unsigned int width, unsigned int height, double squareSize) {
	vector<vector<Vector2d>> coord(height, vector<Vector2d>(width));
	for (unsigned int y = 0; y < height; ++y) {
		for (unsigned int x = 0; x < width; ++x) {
			coord[y][x] = Vector2d(x*squareSize, y*squareSize);
		}
	}
	return coord;
}

void noise(
	VoronoiDiagram::VoronoiGraph& graph,
	function<double (double, double, double)> interpolate,
	double frequency, double amplitude) 
{
	using sf::Vector2i;
	enum Cardinal {
		SouthEast = 0,
		SouthWest = 1,
		NorthWest = 2,
		NorthEast = 3
	};
	vector<Vector2i> corners(4); 
	corners[SouthEast] = Vector2i(1, 0);
	corners[SouthWest] = Vector2i(0, 0);
	corners[NorthWest] = Vector2i(0, 1);
	corners[NorthEast] = Vector2i(1, 1);
	
	double squareSize = graph.height / frequency;
	unsigned int height = static_cast<unsigned int>(ceil(frequency) + 1);
	unsigned int width = static_cast<unsigned int>(ceil(graph.width/squareSize) + 1);
	vector<vector<Vector2d>> gradientField = getGradientField(width, height);
	vector<vector<Vector2d>> coord = getCoord(width, height, squareSize);
	for (auto& node : graph.nodes) {
		double x = node->pos.x;
		if (x < 0.0) x = squareSize/2;
		if (x >= graph.width) x = graph.width - squareSize/2;

		double y = node->pos.y;
		if (y < 0.0) y = squareSize/2;
		if (y >= graph.height) y = graph.height - squareSize/2;

		unsigned int xCell = static_cast<unsigned int>(x / squareSize);
		unsigned int yCell = static_cast<unsigned int>(y / squareSize);
		vector<double> values;
		for (const auto& corner : corners) {
			Vector2d cornerPos = coord[yCell+corner.y][xCell+corner.x];
			Vector2d grad = gradientField[yCell+corner.y][xCell+corner.x];
			double value = dot(normalize(cornerPos - node->pos), grad);
			values.push_back(value);
		}

		Vector2d cell = coord[yCell][xCell];
		
		double xReal = (x - cell.x)/squareSize;
		double y1 = interpolate(xReal, values[SouthWest], values[SouthEast]);
		double y2 = interpolate(xReal, values[NorthWest], values[NorthEast]);

		double yReal = (y - cell.y)/squareSize;
		double z = interpolate(yReal, y1, y2);

		node->elevation += z*amplitude;
	}
}

void perlinNoise(
	VoronoiDiagram::VoronoiGraph& graph, 
	std::function<double (double, double, double)> interpolate, 
	double fundamental, 
	unsigned int nbOctaves,
	double persistence) 
{
	double frequency = fundamental;
	double amplitude = 1.;
	double amplitudeMax = 0.;
	for (unsigned int octave = 0; octave < nbOctaves; ++octave) {
		amplitudeMax += amplitude;
		noise(graph, interpolate, frequency, amplitude);
		frequency *= 2.5;
		amplitude *= persistence;
	}

	for (auto& node : graph.nodes) {
		node->elevation /= amplitudeMax;
		node->elevation = (node->elevation+1)/2;
	}
}

void computeAverageSiteElevation(VoronoiDiagram::VoronoiGraph& graph) {
	for (auto& site : graph.sites) {
		double sum = 0.;
		for (const auto& node : site->nodes) {
			sum += node->elevation;
		}
		if (!site->nodes.empty())
			site->elevation = sum/site->nodes.size();
	}
}

void computeMap(
	VoronoiDiagram::VoronoiGraph& graph, 
	std::function<double (double, double, double)> interpolate, 
	double fundamental, 
	unsigned int nbOctaves,
	double persistence) {
		perlinNoise(graph, interpolate, fundamental, nbOctaves, persistence);
		makeIsland(graph);
		computeAverageSiteElevation(graph);
}

double linearInterpolate(double t, double a, double b) {
	return a + t*(b-a);
}

double quinticInterpolate(double t, double a, double b) {
	return linearInterpolate(t*t*t*(6*t*t-15*t+10), a, b);
}

function<sf::Color (double)> linearGradient(const vector<GradientParameters>& parameters) {
	return [=](double elevation) -> sf::Color {
		double sum = 0.;
		for (const auto& param : parameters) {
			double coeff = (elevation-sum)/param.ratio;
			sum += param.ratio;
			if (elevation > sum)
				continue ;
			int deltaRed = int(param.high.r) - int(param.low.r);
			int deltaGreen = int(param.high.g) - int(param.low.g);
			int deltaBlue = int(param.high.b) - int(param.low.b);
			sf::Uint8 red = static_cast<sf::Uint8>(param.low.r + coeff*deltaRed);
			sf::Uint8 green = static_cast<sf::Uint8>(param.low.g + coeff*deltaGreen);
			sf::Uint8 blue = static_cast<sf::Uint8>(param.low.b + coeff*deltaBlue);
			return sf::Color(red, green, blue);
		}
		return parameters.back().high;
	};
}

GradientParameters::GradientParameters(sf::Color _low, sf::Color _high, double _ratio):
low(_low), high(_high), ratio(_ratio){}

function<sf::Color (double)> groundGradient() {
	vector<GradientParameters> param;
	param.emplace_back(sf::Color(0, 98, 145), sf::Color(0, 98, 145), 0.2);
	param.emplace_back(sf::Color(0, 98, 145), sf::Color(0, 162, 255), 0.2);
	param.emplace_back(sf::Color(0, 130, 0), sf::Color(0, 75, 0), 0.2);
	param.emplace_back(sf::Color(192, 82, 0), sf::Color(100, 40, 0), 0.15);
	param.emplace_back(sf::Color(64, 64, 64), sf::Color(255, 255, 255), 0.15);
	param.emplace_back(sf::Color::White, sf::Color::White, 1);
	return linearGradient(param);
}

function<sf::Color (double)> islandGradient() {
	vector<GradientParameters> param;
	param.emplace_back(sf::Color(0, 98, 145), sf::Color(0, 98, 145), 0.25);
	param.emplace_back(sf::Color(0, 98, 145), sf::Color(0, 162, 255), 0.25);
	param.emplace_back(sf::Color(200, 150, 0), sf::Color(200, 150, 0), 0.03);
	param.emplace_back(sf::Color(130, 130, 0), sf::Color(0, 75, 0), 0.2);
	param.emplace_back(sf::Color(192, 82, 0), sf::Color(100, 40, 0), 0.12);
	param.emplace_back(sf::Color(64, 64, 64), sf::Color(255, 255, 255), 0.18);
	param.emplace_back(sf::Color::White, sf::Color::White, 1);
	return linearGradient(param);
}

void makeIsland(VoronoiDiagram::VoronoiGraph& graph) {
	const double a = 2.;
	const double b = 0.4;
	Vector2d center(graph.width/2, graph.height/2);
	double rMax = distance2(center, Vector2d(0.,0.));
	for (const auto& node: graph.nodes) {
		double r = distance2(center, node->pos)/rMax; // between 0 and 1
		node->elevation *= (exp(-a*r)+b);
	}
}

void CudaPerlinNoise::PerlinNoise(
	VoronoiDiagram::VoronoiGraph& graph, 
	double fundamental, 
	unsigned int nbOctaves, 
	double persistence) {

	unique_ptr<float []> h_x(new float[graph.nodes.size()]);
	unique_ptr<float []> h_y(new float[graph.nodes.size()]);
	for (unsigned int node = 0; node < graph.nodes.size(); ++node) {
		h_x[node] = static_cast<float>(graph.nodes[node]->pos.x);
		h_y[node] = static_cast<float>(graph.nodes[node]->pos.y);
	}

	unique_ptr<float []> h_z(new float[graph.nodes.size()]);
	CUDA_PerlinNoise(
		float(graph.width), float(graph.height), graph.nodes.size(), 
		h_x.get(), h_y.get(), h_z.get(), 
		float(fundamental), nbOctaves, float(persistence));

	for (unsigned int node = 0; node < graph.nodes.size(); ++node)
		graph.nodes[node]->elevation = static_cast<double>(h_z[node]);
}

void CudaPerlinNoise::computeParallelMap(
	VoronoiDiagram::VoronoiGraph& graph,  
	double fundamental, 
	unsigned int nbOctaves,
	double persistence) {
		CudaPerlinNoise::PerlinNoise(graph, fundamental, nbOctaves, persistence);
		makeIsland(graph);
		computeAverageSiteElevation(graph);
}
