#ifndef GEOMETRIC_UTILITY_HPP
#define GEOMETRIC_UTILITY_HPP

#include "SFML\Graphics.hpp"
// #include "FortuneProcedure.h"

template <typename T> int sgn(T val) {
    return (val >= T(0))? 1 : -1;
}

typedef sf::Vector2<double> Vector2d;

struct compVector2d {
	bool operator()(const Vector2d&, const Vector2d&) const;
};

template<typename T>
struct compVector2dPtr { 
	bool operator()(const T& left, const T& right) const {
		if (left->y != right->y)
			return left->y < right->y;
		return left->x < right->x;
	}
};

double norm2(Vector2d);
double distance2(Vector2d, Vector2d);
double dot(Vector2d, Vector2d);
Vector2d normalize(Vector2d);

Vector2d* degenerateIntersection(const Vector2d&, const Vector2d&);
double xParabolaIntersection(double, const Vector2d&, const Vector2d&);
double yParabolaIntersection(const Vector2d&, const Vector2d&);

#endif