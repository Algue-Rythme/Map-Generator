#ifndef FORTUNE_PROCEDURE_H
#define FORTUNE_PROCEDURE_H

#include <exception>
#include <memory>
#include <queue>
#include <set>

#include "SFML\Graphics.hpp"

#include "GeometricUtility.h"
#include "VoronoiDiagram.h"

class Fortune {
public:

	struct Edge;
	struct Event;
	struct Tree;

	Fortune(unsigned int, unsigned int, const std::vector<Vector2d>&);

	void computeDiagram();
	VoronoiDiagram::VoronoiGraph getVoronoiGraph() const;

	struct Edge
	{
		Edge();
		Edge(const Vector2d*, const Vector2d*, const Vector2d*);

		void print() const;

		const Vector2d* start;
		const Vector2d* end;
		const Vector2d* cellLeft;
		const Vector2d* cellRight;
		Edge* neighbour;
		Vector2d direction;
		double f;
		double g;
	};

	enum class EventType {
		None,
		Site,
		Circle
	};

	struct Event
	{
		Event(Vector2d*);
		Event(Vector2d*, Tree*);

		const Vector2d* const site;
		const EventType type;
		Tree* parabola;
		bool deprecated;
	};

	struct Tree
	{
		Tree();
		Tree(const Vector2d*);
		Tree(Tree*, Tree*, Edge*);

		bool isLeaf;
		const Vector2d* site;
		Tree* parent;
		Tree* left;
		Tree* right;
		Edge* edge;
		Event* event;
	};

private:

	Fortune(const Fortune&);

	void insertParabola(const Vector2d*);
	void removeParabola(Tree*, const Vector2d*);
	void checkCircleEvent(Tree*, double);
	void finishEdges(Tree*);
	void pickEdges();
	static Vector2d* const edgeIntersection(const Fortune::Edge&, const Fortune::Edge&);

	Tree* findParabolaIntersection(Tree*, const Vector2d*);
	Tree* leastCommonAncestry(Tree*, Tree*, Tree*);
	void replaceBy(Tree*, Tree*);
	Tree* leftChild(Tree*);
	Tree* rightChild(Tree*);
	Tree* leftParent(Tree*);
	Tree* rightParent(Tree*);

	void deprecateEvent(Tree*);
	struct maxHeapAxisY { bool operator()(const Event* const, const Event* const) const; };

	std::vector<std::unique_ptr<Vector2d>> vertices;
	std::vector<std::unique_ptr<Edge>> edges;

	double width;
	double height;
	std::set<std::unique_ptr<Vector2d>, compVector2dPtr<std::unique_ptr<Vector2d>>> sites;
	Vector2d* const sentinel;

	std::priority_queue <Event* const, std::vector<Event* const>, maxHeapAxisY> events;
	Tree* root;
	bool computed;
};

#endif
