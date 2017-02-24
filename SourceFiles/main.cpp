#include <algorithm>
#include <iostream>
#include <functional>
#include <fstream>
#include <vector>

#include <SFML/Graphics.hpp>

#include "FortuneProcedure.h"
#include "PerlinNoise.h"
#include "PointGenerator.h"
#include "VoronoiDiagram.h"

using namespace std;
using namespace sf;

#define ISLAND
#define GPU_NOISE

void computeTests(unsigned int width, unsigned int height, unsigned int nbPoints, unsigned int nbTests) {
	ofstream out("tests.txt");
	for (unsigned int test = 0; test < nbTests; ++test) {
		vector<Vector2d> points = VoronoiDiagram::randomPoints(nbPoints, width, height);
		Fortune fortune(width, height, points);
		Clock clock;
		fortune.computeDiagram();
		out << clock.getElapsedTime().asMilliseconds() << "\n";
	}
}

int main() {
	const unsigned int width = 800;
	const unsigned int height = 800;

	const unsigned int nbPoints = 64*1000;

	vector<Vector2d> points = VoronoiDiagram::randomPoints(nbPoints, width, height);

	cout << "Calcul du diagramme... ";
	Clock clock;
	points = VoronoiDiagram::LloydRelaxation(width, height, points, 2);
	VoronoiDiagram::VoronoiGraph graph = VoronoiDiagram::optimisedProceduralGeneration(width, height, points);
	vector<CircleShape> sites = graph.getSites(1.5);
	cout << clock.getElapsedTime().asMilliseconds() << " ms\n";
	
#ifdef ISLAND
	double fundamental = 5.;
	unsigned int nbOctaves = 6;
	double persistence = 0.5;
	const auto colorGradient = islandGradient();
	const auto vertices = &VoronoiDiagram::VoronoiGraph::getMixedMap;

	cout << "Calcul de la carte... ";
	clock.restart();
#ifndef GPU_NOISE
	computeMap(graph, quinticInterpolate, fundamental, nbOctaves, persistence);
#else
	CudaPerlinNoise::computeParallelMap(graph, fundamental, nbOctaves, persistence);
#endif
	cout << clock.getElapsedTime().asMilliseconds() << " ms\n";

	VertexArray pixels = (graph.*vertices)(colorGradient);
#else
	VertexArray pixels = graph.getDelaunayTriangulation();
#endif

    RenderWindow window(VideoMode(width, height), "Map generator");
	
    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        window.clear(sf::Color::White);
		window.draw(pixels);
#ifndef ISLAND
		for (const auto& site : sites)
			window.draw(site);
#endif
        window.display();
    }

    return 0;
}
