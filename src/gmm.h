#ifndef GMM_H
#define GMM_H

/**
  * Gaussian Mixture Models
  */
#include "mathutil.h"


/**
 * 1D gaussian mixture PDF
 */
double log_gmm_pdf(double data, const vec &means, const vec &vars, const vec &weights);

/**
 * generate_gmm create one-dimensional GMM dataset
 * @return a vector of normally-distributed
 */
vec generate_gmm(size_t N, const vec &mean, const vec &vars, const vec &weights);


/**
 * em_gmm EM-estimation one-dimensional Gaussian mixture model with a fixed number of comps
 * @return the number of iterations, max_iters if not converged
 */
size_t em_gmm(const vec &data, size_t num_comps, size_t max_iters, vec &means, vec &vars, vec &weights, double &log_l);


#endif // GMM_H
