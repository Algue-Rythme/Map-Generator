#ifndef VORONOI_DIAGRAM_HPP
#define VORONOI_DIAGRAM_HPP

#include <functional>
#include <memory>
#include <set>

#include "SFML\Graphics.hpp"

typedef sf::Vector2<double> Vector2d;

namespace VoronoiDiagram {

	std::vector<Vector2d> randomPoints(unsigned int, unsigned int, unsigned int);
	std::vector<sf::Color> randomColors(unsigned int);

	sf::VertexArray naiveProceduralGeneration
		(unsigned int, unsigned int, const std::vector<Vector2d>&, const std::vector<sf::Color>&);

	sf::VertexArray naiveParallelGeneration
		(unsigned int, unsigned int, const std::vector<Vector2d>&, const std::vector<sf::Color>&);
	
	class VoronoiGraph;
	VoronoiDiagram::VoronoiGraph optimisedProceduralGeneration (unsigned int, unsigned int, const std::vector<Vector2d>&);

	std::vector<Vector2d> LloydRelaxation(unsigned int, unsigned int, const std::vector<Vector2d>&, unsigned int);

	class VoronoiGraph {
	public:

		struct DualEdge {
			DualEdge(unsigned int, unsigned int, unsigned int, unsigned);
			unsigned int nodeA;
			unsigned int nodeB;
			unsigned int siteA;
			unsigned int siteB;
		};

		struct Edge;
		struct Node;
		struct Site;

		struct Site {
			Site(const Vector2d&);

			Vector2d pos;
			std::vector<Site*> borders;
			std::vector<Edge*> edges;
			std::set<Node*> nodes;

			double elevation;
		};

		struct Node {
			Node(const Vector2d&);

			Vector2d pos;
			std::vector<Node*> nodes;
			std::vector<Edge*> edges;

			double elevation;
		};

		struct Edge {
			Edge(Node*,Node*,Site*,Site*);
			Node* nodeA;
			Node* nodeB;
			Site* siteA;
			Site* siteB;
		};

		VoronoiGraph(
			double,
			double, 
			const std::vector<Vector2d>&, 
			const std::vector<Vector2d>&,
			const std::vector<DualEdge>&);

		sf::VertexArray getEdges() const;
		sf::VertexArray getBeams() const;
		sf::VertexArray getDelaunayTriangulation() const;

		std::vector<sf::CircleShape> getSites(float) const;

		sf::VertexArray getCellMap(std::function<sf::Color (double)>) const;
		sf::VertexArray getTriangulationMap(std::function<sf::Color (double)>) const;
		sf::VertexArray getMixedMap(std::function<sf::Color (double)>) const;

		double width;
		double height;
		std::vector<std::unique_ptr<Site>> sites;
		std::vector<std::unique_ptr<Node>> nodes;
		std::vector<std::unique_ptr<Edge>> edges;
	};
}

#endif