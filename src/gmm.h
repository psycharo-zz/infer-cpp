#ifndef GMM_H
#define GMM_H

/**
  * Gaussian Mixture Models
  */
#include "mathutil.h"


/**
 * 1D gaussian mixture PDF
 */
double gmm_log_pdf(double data, const vec &means, const vec &vars, const vec &weights);

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


/**
 * vb_gmm variational inference-based GMM
 * @return # of iterations
 */
size_t vb_gmm(const vec &data, size_t num_comps, size_t max_iters, vec &means, vec &vars, vec &weights, double &log_l,
              double prior_mean_m = 0, double prior_mean_b = 1e-3, double prior_prec_a = 1e-5, double prior_prec_b = 1e-5, double prior_dir_u = 5);



#endif // GMM_H
