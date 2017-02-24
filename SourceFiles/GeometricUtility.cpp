#include <algorithm>
#include <iostream>
#include <cmath>

#include "GeometricUtility.h"
#include "FortuneProcedure.h"

using namespace sf;
using namespace std;

bool compVector2d::operator()(const Vector2d& a, const Vector2d& b) const {
	return (a.y!=b.y)? a.y<b.y : a.x<b.x;
}

double distance2(Vector2d left, Vector2d right) { 
	return norm2(right - left); 
}

double norm2(Vector2d vect) { 
	return sqrt(dot(vect, vect)); 
}

double dot(Vector2d vect1, Vector2d vect2) {
	return vect1.x*vect2.x + vect1.y*vect2.y;
}

Vector2d normalize(Vector2d vect) {
	double n = norm2(vect);
	if (n <= 1e-11)
		return vect;
	return vect/n;
}

static double distanceParabola(const Vector2d& site, double sweepLine) 
{ return 2*(site.y - sweepLine); }

static double computeB(const Vector2d& site, double dp) 
{ return -2*site.x/dp; }

static double computeC(const Vector2d& site, double dp, double y) 
{ return y+dp/4+site.x*site.x/dp; }

double xParabolaIntersection(double sweepLine, const Vector2d& left, const Vector2d& right) {
	const double epsilon = 1e-10;
	if (fabs(left.y-right.y) <= epsilon)
		return (left.x+right.x)/2.;
	if (fabs(left.x-right.x) <= epsilon)
		return left.x;

	if (fabs(left.y-sweepLine) <= epsilon)
		return left.x;
	if (fabs(right.y-sweepLine) <= epsilon)
		return right.x;

	double dpLeft = distanceParabola(left, sweepLine);
	double dpRight = distanceParabola(right, sweepLine); 

	double a = 1/dpLeft - 1/dpRight;
	double b = computeB(left, dpLeft) -	computeB(right, dpRight);
	double c = computeC(left, dpLeft, sweepLine) -	computeC(right, dpRight, sweepLine);
	
	double delta = b*b - 4*a*c;
	double x1 = (-b + sqrt(delta))/(2*a);
	double x2 = (-b - sqrt(delta))/(2*a);

	if (left.y < right.y)
		return max(x1, x2);
	else
		return min(x1, x2);
}

double yParabolaIntersection(const Vector2d& parabola, const Vector2d& degenerate) {
	const double epsilon = 1e-10;
	if (fabs(parabola.y - degenerate.y) <= epsilon)
		return degenerate.y;

	const double x = degenerate.x;
	double dp = distanceParabola(parabola, degenerate.y);
	double a = 1/dp;
	double b = computeB(parabola, dp);
	double c = computeC(parabola, dp, degenerate.y);
	return a*x*x+b*x+c;
}

Vector2d* degenerateIntersection(const Vector2d& parabola, const Vector2d& degenerate) {
	const double epsilon = 1e-10;
	double x = (fabs(parabola.y - degenerate.y) <= epsilon)? 
		(degenerate.x+parabola.x)/2 : degenerate.x;
	double y = yParabolaIntersection(parabola, Vector2d(x, degenerate.y));

	return new Vector2d(x, y);
}
