#include <cassert>
#include <iostream>
#include <map>

#include "FortuneProcedure.h"
#include "GeometricUtility.h"

Fortune::Fortune(unsigned int _width, 
				 unsigned int _height, 
				 const std::vector<Vector2d>& points): 
width(static_cast<double>(_width)), height(static_cast<double>(_height)), sentinel(new Vector2d(width/2.0, height*100000)), root(nullptr), computed(false) {
	for (auto point: points)
		sites.insert(std::unique_ptr<Vector2d>(new Vector2d(point)));
	// sites.insert(std::unique_ptr<Vector2d>(sentinel));
	for (const auto& site : sites)
		events.push(new Event(site.get()));
}

void Fortune::computeDiagram() {
	if (computed)
		return ;
	computed = true;

	while (!events.empty()) {
		Event* event = events.top();
		events.pop();
		if (!event->deprecated) {
			switch (event->type) {
			case EventType::Site:
				insertParabola(event->site);
				break;
			case EventType::Circle:
				removeParabola(event->parabola, event->site);
				break;
			}
		}
		delete event;
	}
	finishEdges(root);
	pickEdges();
}

VoronoiDiagram::VoronoiGraph Fortune::getVoronoiGraph() const { 
	std::vector<Vector2d> sitesCoordinates; 
	sitesCoordinates.reserve(sites.size());
	std::map<const Vector2d*, unsigned> sitesIndices;
	unsigned int index = 0;
	for (const auto& site: sites) {
		sitesCoordinates.push_back(*site);
		sitesIndices[site.get()] = index;
		index += 1;
	}

	std::vector<Vector2d> nodesCoordinates; 
	nodesCoordinates.reserve(vertices.size());
	std::map<const Vector2d*, unsigned> nodesIndices;
	for (unsigned int node = 0; node < vertices.size(); ++node) {
		nodesCoordinates.push_back(*vertices[node]);
		nodesIndices[vertices[node].get()] = node;
	}

	std::vector<VoronoiDiagram::VoronoiGraph::DualEdge> edgesData; 
	edgesData.reserve(edgesData.size());
	for (const auto& edge : edges) {
		unsigned int v1 = nodesIndices[edge->start];
		unsigned int v2 = nodesIndices[edge->end];
		unsigned int s1 = sitesIndices[edge->cellLeft];
		unsigned int s2 = sitesIndices[edge->cellRight];
		edgesData.emplace_back(v1, v2, s1, s2);
	}

	return VoronoiDiagram::VoronoiGraph(width, height, sitesCoordinates, nodesCoordinates, edgesData);
}

void Fortune::insertParabola(const Vector2d* site) {
	if (root == nullptr) {
		root = new Tree(site);
		return ;
	}

	Tree* const parabola = findParabolaIntersection(root, site);
	deprecateEvent(parabola);

	Vector2d* vertex = degenerateIntersection(*parabola->site, *site);
	vertices.emplace_back(vertex);

	Edge* edgeLeft = new Edge(vertex, parabola->site, site);
	Edge* edgeRight = new Edge(vertex, site, parabola->site);
	edgeLeft->neighbour = edgeRight;
	edges.emplace_back(edgeLeft);

	Tree* pLeft = new Tree(parabola->site);
	Tree* pMid = new Tree(site);
	Tree* pRight = new Tree(parabola->site);
	
	replaceBy(parabola, new Tree(new Tree(pLeft, pMid, edgeLeft), pRight, edgeRight));

	checkCircleEvent(pLeft, site->y);
	checkCircleEvent(pRight, site->y);
}

void Fortune::removeParabola(Tree* parabola, const Vector2d* site) {
	Tree* const lp = leftParent(parabola);
	Tree* const rp = rightParent(parabola);
	Tree* const previousLeaf = leftChild(lp);
	Tree* const nextLeaf = rightChild(rp);

	deprecateEvent(previousLeaf);
	deprecateEvent(nextLeaf);

	Vector2d* vertex = edgeIntersection(*lp->edge, *rp->edge);
	vertices.emplace_back(vertex);

	lp->edge->end = vertex;
	rp->edge->end = vertex;

	Edge* edge = new Edge(vertex, previousLeaf->site, nextLeaf->site);
	edges.emplace_back(edge);

	Tree* const higher = leastCommonAncestry(parabola, lp, rp);
	higher->edge = edge;

	Tree* const father = parabola->parent;
	Tree* const brother = (father->left == parabola)? father->right : father->left;
	replaceBy(father, brother);
	delete parabola;

	checkCircleEvent(previousLeaf, site->y);
	checkCircleEvent(nextLeaf, site->y);
}

void Fortune::checkCircleEvent(Tree* parabola, double sweepLine) {
	Tree* const lp = leftParent(parabola);
	Tree* const rp = rightParent(parabola);

	Tree* const previousLeaf = leftChild(lp);
	Tree* const nextLeaf = rightChild(rp);

	if (previousLeaf == nullptr 
	 || nextLeaf == nullptr
	 || previousLeaf->site == nextLeaf->site)
	 return ;

	Vector2d* const intersection = edgeIntersection(*lp->edge, *rp->edge);
	if (intersection == nullptr)
		return ;

	double d = distance2(*parabola->site, *intersection);
	if (intersection->y - d >= sweepLine) {
		delete intersection;
		return ;
	}
	vertices.emplace_back(intersection);

	Vector2d* const circle = new Vector2d(intersection->x, intersection->y - d);
	vertices.emplace_back(circle);
	Event* circleEvent = new Event(circle, parabola);
	parabola->event = circleEvent;
	events.push(circleEvent);
}

void Fortune::finishEdges(Tree* node) {
	if (!node->isLeaf) {
		double mx = (node->edge->direction.x >= 0.)?
			std::max(width, node->edge->start->x + 10.):
			std::min(0., node->edge->start->x - 10.);

		Vector2d* end = new Vector2d(mx, mx*node->edge->f + node->edge->g);
		vertices.emplace_back(end);
		node->edge->end = end;

		finishEdges(node->left);
		finishEdges(node->right);
	}
	delete node;
}

void Fortune::pickEdges() {
	for (const auto& edge : edges) {
		if (edge->neighbour != nullptr) {
			edge->start = edge->neighbour->end;
			delete edge->neighbour;
			edge->neighbour = nullptr;
		}
	}
}

void Fortune::deprecateEvent(Tree* leaf) {
	if (leaf->event == nullptr)
		return ;
	leaf->event->deprecated = true;
	leaf->event = nullptr;
}

Vector2d* const Fortune::edgeIntersection(const Fortune::Edge& left, const Fortune::Edge& right) {
	const double epsilon = 1e-10;
	if (fabs(left.f-right.f) <= epsilon)
		return nullptr;

	double x = (right.g - left.g)/(left.f - right.f);
	double y = right.f*x + right.g;
	
	if (sgn(x - left.start->x)*sgn(left.direction.x) < 0
	 || sgn(y - left.start->y)*sgn(left.direction.y) < 0
	 || sgn(x - right.start->x)*sgn(right.direction.x) < 0
	 || sgn(y - right.start->y)*sgn(right.direction.y) < 0)
	 return nullptr;

	return new Vector2d(x, y);
}

bool Fortune::maxHeapAxisY::operator()(const Event* const left, const Event* const right) const {
	if (left->site->y != right->site->y)
		return left->site->y < right->site->y;
	return left->site->x < right->site->x;
}

void Fortune::replaceBy(Tree* old, Tree* subTree) {
	if (old == root) {
		delete root;
		root = subTree;
	} else {
		Tree* const father = old->parent;
		subTree->parent = father;
		Tree** const son = (father->left == old)? &father->left : &father->right;
		*son = subTree;
		delete old;
	}
}

Fortune::Tree* Fortune::findParabolaIntersection(Tree* node, const Vector2d* degenerate) {
	while (!node->isLeaf) {
		Tree* const left = leftChild(node);
		Tree* const right = rightChild(node);
		double xIntersection = xParabolaIntersection(degenerate->y, *left->site, *right->site);
		if (degenerate->x < xIntersection)
			node = node->left;
		else
			node = node->right;
	}

	return node;
}

Fortune::Tree* Fortune::leastCommonAncestry(Tree* start, Tree* candidate1, Tree* candidate2) {
	if (start == root)
		return root;

	if (start->parent == candidate1)
		return candidate2;
	else
		return candidate1;
}

Fortune::Tree* Fortune::leftChild(Tree* node) {
	if (node == nullptr)
		return nullptr;
	node = node->left;
	while (!node->isLeaf)
		node = node->right;
	return node;
}

Fortune::Tree* Fortune::rightChild(Tree* node) {
	if (node == nullptr)
		return nullptr;
	node = node->right;
	while (!node->isLeaf)
		node = node->left;
	return node;
}

Fortune::Tree* Fortune::leftParent(Tree* node) {
	Tree* parent = node->parent;
	while (parent != nullptr && parent->left == node) {
		node = parent;
		parent = parent->parent;
	}
	return parent;
}

Fortune::Tree* Fortune::rightParent(Tree* node) {
	Tree* parent = node->parent;
	while (parent != nullptr && parent->right == node) {
		node = parent;
		parent = parent->parent;
	}
	return parent;
}

Fortune::Edge::Edge(): 
	start(nullptr), 
	end(nullptr), 
	cellLeft(nullptr), 
	cellRight(nullptr), 
	neighbour(nullptr){}
Fortune::Edge::Edge(const Vector2d* _start, 
					const Vector2d* _cellLeft, 
					const Vector2d* _cellRight): 
	start(_start), 
	end(nullptr), 
	cellLeft(_cellLeft), 
	cellRight(_cellRight), 
	neighbour(nullptr),
	direction(cellRight->y - cellLeft->y, cellLeft->x - cellRight->x) {
	const double epsilon = 1e-10;
	if (fabs(cellLeft->y - cellRight->y) <= epsilon) {
		double verticality = sgn(cellRight->x - cellLeft->x);
		f = 1/epsilon;
		direction.x = -epsilon*verticality;
	} else {
		f = (cellRight->x - cellLeft->x)/(cellLeft->y - cellRight->y);
	}

	g = start->y - f*start->x;
}

void Fortune::Edge::print() const {
	std::cout << "(" << start->x << ", " << start->y << ") to ";
	if (end != nullptr)
		std::cout << "(" << end->x << ", " << end->y << ")\n";
	else
		std::cout << "NULL\n";
}

Fortune::Event::Event(Vector2d* _site): 
	site(_site),  
	type(EventType::Site), 
	parabola(nullptr),
	deprecated(false){}
Fortune::Event::Event(Vector2d* _site, Tree* _parabola): 
	site(_site), 
	type(EventType::Circle), 
	parabola(_parabola),
	deprecated(false){}

Fortune::Tree::Tree(): 
	isLeaf(false), 
	site(nullptr), 
	parent(nullptr), 
	left(nullptr), 
	right(nullptr), 
	edge(nullptr), 
	event(nullptr){}
Fortune::Tree::Tree(const Vector2d* _site): 
	isLeaf(true), 
	site(_site), 
	parent(nullptr), 
	left(nullptr), 
	right(nullptr), 
	edge(nullptr), 
	event(nullptr){}
Fortune::Tree::Tree(Tree* _left, Tree* _right, Edge* _edge): 
	isLeaf(false), 
	site(nullptr), 
	parent(nullptr), 
	left(_left), 
	right(_right), 
	edge(_edge), 
	event(nullptr) {
	if (left != nullptr)
		left->parent = this;
	if (right != nullptr)
		right->parent = this;
}
