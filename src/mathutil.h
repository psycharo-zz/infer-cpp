#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <cmath>

#include <armadillo>
using namespace arma;

// TODO: mathutil
template <typename T>
inline T sqr(const T &x) { return x * x; }

inline double norm_pdf(double x, double mean, double var) { return exp( - 0.5 * sqr(x - mean) / var) / sqrt(2 * M_PI * var); }

inline double log_norm_pdf(double x, double mean, double var) { return - 0.5 * sqr(x - mean) / var - log(sqrt(2 * M_PI * var)); }

const double EPS = 1e-3;


#endif // MATHUTIL_H
