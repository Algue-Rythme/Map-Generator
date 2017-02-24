#include "VoronoiDiagram.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <set>

#include "FortuneProcedure.h"
#include "GeometricUtility.h"
#include "VoronoiDiagram.h"
#include "VoronoiDiagram.cuh"

using namespace std;
using namespace sf;

namespace {
	const unsigned seed = static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count());
	default_random_engine generator(643227911);
}

vector<Vector2d> VoronoiDiagram::randomPoints(unsigned int nbPoints, unsigned int width, unsigned int height) {
	const unsigned factor = 1000*1000;
	vector<Vector2d> points(nbPoints);
	generate(begin(points), end(points),
		[=](){return Vector2d(double(generator()%(width*factor))/factor, double(generator()%(height*factor))/factor);});
	return points;
}

vector<Color> VoronoiDiagram::randomColors(unsigned int nbPoints) {
	vector<sf::Color> colors(nbPoints);
	generate(begin(colors), end(colors),
		[=](){return Color(generator()%255, generator()%255, generator()%255);});
	return colors;
}

VertexArray VoronoiDiagram::naiveProceduralGeneration
	(unsigned int width, unsigned int height, 
	const vector<Vector2d>& points, 
	const vector<Color>& colors)
{
	VertexArray pixels(PrimitiveType::Points, width*height);
	for (unsigned int y = 0; y < height; ++y) {
		for (unsigned int x = 0; x < width; ++x) {
			Vertex& currentPixel = pixels[y*width + x];
			currentPixel.position = Vector2f(float(x), float(y));
			double bestDistance = distance2(points[0], Vector2d(currentPixel.position));
			unsigned int bestNeighbour = 0;
			for (unsigned int neighbour = 1; neighbour < points.size(); ++neighbour) {
				double distance = distance2(points[neighbour], Vector2d(currentPixel.position));
				if (distance < bestDistance)
				{
					bestDistance = distance;
					bestNeighbour = neighbour;
				}
			}
			currentPixel.color = colors[bestNeighbour];
		}
	}

	return pixels;
}

VertexArray VoronoiDiagram::naiveParallelGeneration
	(unsigned int width, unsigned int height, 
	const vector<Vector2d>& points, 
	const vector<Color>& colors)
{
	VertexArray pixels(sf::PrimitiveType::Points, width*height);
	for (unsigned int y = 0; y < height; ++y)
		for (unsigned int x = 0; x < width; ++x)
			pixels[y*width + x].position = Vector2f(float(x), float(y));

	unique_ptr<unsigned int[]> h_index(new unsigned int[width*height]);
	unique_ptr<unsigned int[]> h_x(new unsigned int [points.size()]);
	unique_ptr<unsigned int[]> h_y(new unsigned int [points.size()]);
	for (unsigned int point = 0; point < points.size(); ++point) {
		h_x[point] = static_cast<unsigned int>(points[point].x);
		h_y[point] = static_cast<unsigned int>(points[point].y);
	}

	CUDA_VoronoiDiagram::naiveParallelGeneration(width, height, h_index.get(), points.size(), h_x.get(), h_y.get());

	for (unsigned int pixel = 0; pixel < pixels.getVertexCount(); ++pixel)
		pixels[pixel].color = colors[h_index[pixel]];
	return pixels;
}

VoronoiDiagram::VoronoiGraph VoronoiDiagram::optimisedProceduralGeneration (
	unsigned int width, 
	unsigned int height, 
	const std::vector<Vector2d>& points) {

	Fortune fortune(width, height, points);
	fortune.computeDiagram();
	return fortune.getVoronoiGraph();
}

vector<Vector2d> VoronoiDiagram::LloydRelaxation
	(unsigned int width, 
	unsigned int height,
	const vector<Vector2d>& sites, 
	unsigned int nbIterations)
{
	vector<Vector2d> diagram = sites;
	for (unsigned int iteration = 0; iteration < nbIterations; ++iteration) {
		Fortune fortune(width, height, diagram);
		fortune.computeDiagram();
		VoronoiGraph cells = fortune.getVoronoiGraph();
		vector<Vector2d> newDiagram; newDiagram.reserve(cells.sites.size());
		for (const auto& site : cells.sites) {
			double x = 0.;
			double y = 0.;
			for (const auto& vertice : site->nodes) {
				x += max(min(vertice->pos.x, double(width)), 0.);
				y += max(min(vertice->pos.y, double(height)), 0.);
			}
			newDiagram.emplace_back(x / site->nodes.size(), y / site->nodes.size());
		}
		diagram = newDiagram;
	}
	return diagram;
}

VoronoiDiagram::VoronoiGraph::DualEdge::DualEdge(
	unsigned int _v1,
	unsigned int _v2,
	unsigned int _s1, 
	unsigned int _s2):
nodeA(_v1),nodeB(_v2),siteA(_s1),siteB(_s2)
{}

VoronoiDiagram::VoronoiGraph::Site::Site(const Vector2d& _pos): pos(_pos), elevation(0.0)
{}

VoronoiDiagram::VoronoiGraph::Node::Node(const Vector2d& _pos): pos(_pos), elevation(0.0)
{}

VoronoiDiagram::VoronoiGraph::Edge::Edge(
	Node* _nodeA, 
	Node* _nodeB,
	Site* _siteA,
	Site* _siteB):
nodeA(_nodeA), nodeB(_nodeB), siteA(_siteA), siteB(_siteB)
{}

VoronoiDiagram::VoronoiGraph::VoronoiGraph(
	double _width,
	double _height,
	const std::vector<Vector2d>& sitesCoordinates, 
	const std::vector<Vector2d>& nodesCoordinates,
	const std::vector<DualEdge>& edgesData):
width(_width), height(_height)
{
	sites.reserve(sitesCoordinates.size());
	for (const auto& site : sitesCoordinates)
		sites.emplace_back(new Site(site));

	nodes.reserve(nodesCoordinates.size());
	for (const auto& vertex : nodesCoordinates)
		nodes.emplace_back(new Node(vertex));

	edges.reserve(edgesData.size());
	for (const auto& edge : edgesData) {
		Node* nodeA = nodes[edge.nodeA].get();
		Node* nodeB = nodes[edge.nodeB].get();
		Site* siteA = sites[edge.siteA].get();
		Site* siteB = sites[edge.siteB].get();
		Edge* graphEdge = new Edge(nodeA, nodeB, siteA, siteB);
		edges.emplace_back(graphEdge);

		nodeA->nodes.push_back(nodeB);
		nodeB->nodes.push_back(nodeA);

		nodeA->edges.push_back(graphEdge);
		nodeB->edges.push_back(graphEdge);

		siteA->borders.push_back(siteB);
		siteB->borders.push_back(siteA);

		siteA->edges.push_back(graphEdge);
		siteA->edges.push_back(graphEdge);

		siteA->nodes.insert(nodeA);
		siteA->nodes.insert(nodeB);
		siteB->nodes.insert(nodeA);
		siteB->nodes.insert(nodeB);
	}
}

sf::VertexArray VoronoiDiagram::VoronoiGraph::getEdges() const {
	sf::Color color = sf::Color::Black;
	sf::VertexArray lines(sf::PrimitiveType::Lines);
	for (const auto& edge : edges) {
		lines.append(sf::Vertex(sf::Vector2f(edge->nodeA->pos), color));
		lines.append(sf::Vertex(sf::Vector2f(edge->nodeB->pos), color));
	}
	return lines;
}

sf::VertexArray VoronoiDiagram::VoronoiGraph::getBeams() const {
	sf::Color color = sf::Color::Black;
	sf::VertexArray triangles = getEdges();
	auto addVertex = [&](const Vector2d& pos){ triangles.append(sf::Vertex(static_cast<sf::Vector2f>(pos), color));};
	for (const auto& edge : edges) {
		addVertex(edge->siteA->pos);
		addVertex(edge->nodeA->pos);

		addVertex(edge->siteA->pos);
		addVertex(edge->nodeB->pos);

		addVertex(edge->siteB->pos);
		addVertex(edge->nodeA->pos);

		addVertex(edge->siteB->pos);
		addVertex(edge->nodeA->pos);
	}
	return triangles;
}

sf::VertexArray VoronoiDiagram::VoronoiGraph::getDelaunayTriangulation() const {
	sf::Color color = sf::Color::Black;
	sf::VertexArray delaunay(sf::PrimitiveType::Lines);
	for (const auto& edge: edges) {
		delaunay.append(sf::Vertex(static_cast<sf::Vector2f>(edge->siteA->pos), color));
		delaunay.append(sf::Vertex(static_cast<sf::Vector2f>(edge->siteB->pos), color));
	}
	return delaunay;
}

vector<sf::CircleShape> VoronoiDiagram::VoronoiGraph::getSites(float radius) const {
	vector<sf::CircleShape> circles; circles.reserve(sites.size());
	for (const auto& site : sites) {
		sf::CircleShape circle(radius);
		circle.setOrigin(sf::Vector2f(radius/2, radius/2));
		circle.setPosition(static_cast<sf::Vector2f>(site->pos));
		circle.setFillColor(sf::Color::Black);
		circles.push_back(circle);
	}
	return circles;
}

sf::VertexArray VoronoiDiagram::VoronoiGraph::getCellMap(std::function<sf::Color (double)> colorGradient) const {
	sf::VertexArray ground(sf::PrimitiveType::Triangles);
	auto addVertex = [&](Vector2d pos, double elevation) {
		ground.append(sf::Vertex(static_cast<sf::Vector2f>(pos), colorGradient(elevation)));
	};
	for (const auto& edge : edges) {
		addVertex(edge->siteA->pos, edge->siteA->elevation);
		addVertex(edge->nodeA->pos, edge->siteA->elevation);
		addVertex(edge->nodeB->pos, edge->siteA->elevation);

		addVertex(edge->siteB->pos, edge->siteB->elevation);
		addVertex(edge->nodeA->pos, edge->siteB->elevation);
		addVertex(edge->nodeB->pos, edge->siteB->elevation);
	}
	return ground;
}

sf::VertexArray VoronoiDiagram::VoronoiGraph::getTriangulationMap(std::function<sf::Color (double)> colorGradient) const {
	sf::VertexArray ground(sf::PrimitiveType::Triangles);
	auto addVertex = [&](Vector2d pos, double elevation) {
		ground.append(sf::Vertex(static_cast<sf::Vector2f>(pos), colorGradient(elevation)));
	};
	for (const auto& edge : edges) {
		addVertex(edge->siteA->pos, edge->siteA->elevation);
		addVertex(edge->nodeA->pos, edge->nodeA->elevation);
		addVertex(edge->nodeB->pos, edge->nodeB->elevation);

		addVertex(edge->siteB->pos, edge->siteB->elevation);
		addVertex(edge->nodeA->pos, edge->nodeA->elevation);
		addVertex(edge->nodeB->pos, edge->nodeB->elevation);
	}
	return ground;
}

sf::VertexArray VoronoiDiagram::VoronoiGraph::getMixedMap(std::function<sf::Color (double)> colorGradient) const {
	sf::VertexArray ground(sf::PrimitiveType::Triangles);
	auto addVertex = [&](Vector2d pos, double elevation) {
		ground.append(sf::Vertex(static_cast<sf::Vector2f>(pos), colorGradient(elevation)));
	};
	for (const auto& edge : edges) {
		addVertex(edge->siteA->pos, edge->siteA->elevation);
		if (edge->siteA->elevation <= 0.5) {
			addVertex(edge->nodeA->pos, edge->nodeA->elevation);
			addVertex(edge->nodeB->pos, edge->nodeB->elevation);
		} else {
			addVertex(edge->nodeA->pos, edge->siteA->elevation);
			addVertex(edge->nodeB->pos, edge->siteA->elevation);
		}

		addVertex(edge->siteB->pos, edge->siteB->elevation);
		if (edge->siteB->elevation <= 0.5) {
			addVertex(edge->nodeA->pos, edge->nodeA->elevation);
			addVertex(edge->nodeB->pos, edge->nodeB->elevation);
		} else {
			addVertex(edge->nodeA->pos, edge->siteB->elevation);
			addVertex(edge->nodeB->pos, edge->siteB->elevation);
		}
	}
	return ground;
}