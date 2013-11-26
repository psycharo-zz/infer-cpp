#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <cmath>

#include <armadillo>
using namespace arma;


namespace infer
{
const double LN_2PI = 1.8378770664093453;

template <typename T>
inline T sqr(const T &x) { return x * x; }

inline double norm_pdf(double x, double mean, double var)
{
    return exp( - 0.5 * sqr(x - mean) / var) / sqrt(2 * M_PI * var);
}

/// gaussian logpdf
inline double norm_log_pdf(double x, double mean, double var)
{
    return - 0.5 * sqr(x - mean) / var - 0.5 * log(var) - 0.5 * LN_2PI;
}

/// The digamma function is the derivative of gammaln.
double digamma(double x);

/// digamma vector type
vec digamma(const vec &v);

const double EPS = 1e-3;

} // namespace infer

#endif // MATHUTIL_H
