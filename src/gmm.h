#ifndef GMM_H
#define GMM_H

/**
  * Gaussian Mixture Models
  */
#include "mathutil.h"

/**
 * generate_gmm create one-dimensional GMM dataset
 * @return a vector of normally-distributed
 */
vec generate_gmm(size_t N, const vec &mean, const vec &vars, const vec &weights);


/**
 * em_gmm EM-estimation one-dimensional Gaussian mixture model with a fixed number of comps
 * @return the number of iterations, max_iters if not converged
 */
size_t em_gmm(const vec &data, size_t num_comps, size_t max_iters, vec &mean, vec &vars, vec &weights, double &log_l);


/**
  * em_gmm_vector train D one-dimensional GMMs using EM
  * useful e.g. for training background model for images
  * @param data NxD matrix, input data
  * @param num_comps the fixed number of components for each model
  * @param max_iters the maximum number of iterations before convergence
  * @param mean DxK matrix, each row 1xK mean vector
  * @param vars DxK matrix, each row 1xK variance vector
  * @param weights DxK matrix, each row 1xK weight vector
  * @return number of iterations before convergence
  */
size_t em_gmm_vector(const mat &data, size_t num_comps, size_t max_iters, mat &mean, mat &vars, mat &weights);

#endif // GMM_H
