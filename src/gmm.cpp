#include "gmm.h"


#include "kmeans.h"
using namespace infer;

#include <vector>
using namespace std;


double gmm_log_pdf(double x, const vec &means, const vec &vars, const vec &weights)
{
    vec logp = -0.5 * square(x - means) / vars - 0.5 * LN_2PI - 0.5 * log(vars) + log(weights);
    double logp_max = max(logp);
    return log(sum(exp(logp - logp_max))) + logp_max;
}

vec generate_gmm(size_t N, const vec &mean, const vec &stddev, const vec &weights)
{
    vec result(N);
    const size_t K = weights.size();
    size_t filled = 0;
    for (size_t k = 0; k < K; ++k)
    {
        size_t size_k = weights(k) * N;
        result.subvec(filled, filled + size_k - 1) =  mean(k) + randn(size_k) * stddev(k);
        filled += size_k;
    }
    if (filled < N)
        result.subvec(filled, N - 1) = mean(K-1) + randn(N - filled) * stddev(K-1);
    return result;
}


size_t em_gmm(const vec &data, size_t K, size_t max_iters, vec &means, vec &vars, vec &weights, double &log_l)
{
    const size_t N = data.size();

    // ad-hoc to have non-zero variances
    const double MIN_VAR = 1e-6;

    means.resize(K);
    vars.resize(K);
    weights = ones(K) / K;

    // responsibilities
    mat resps = zeros(N, K);
    vec resp_count = EPS * ones(K);

    // initialisation
    ivec assigns;
    kmeans(data, K, max_iters, means, assigns);
    vars.fill(0.0);
    for (size_t p = 0; p < N; ++p)
    {
        size_t k = assigns(p);
        vars(k) += sqr(data(p) - means(k));
        ++resp_count(k);
    }
    vars = MIN_VAR + vars / resp_count;
    weights = resp_count / N;

    size_t iters = 0;
    log_l = 0;
    for (iters = 0; iters < max_iters; ++iters)
    {
        // recomputing responsibilities
        for (size_t p = 0; p < N; ++p)
        {
            for (size_t k = 0; k < K; ++k)
                resps(p, k) = weights(k) * norm_pdf(data(p), means(k), vars(k));
            resps.row(p) /= sum(resps.row(p));
        }

        for (size_t k = 0; k < K; ++k)
            resp_count(k) = sum(resps.col(k));

        // recomputing mixture parameters
        for (size_t k = 0; k < K; ++k)
        {
            if (resp_count(k) > EPS)
            {
                means(k) = dot(data, resps.col(k)) / resp_count(k);
                vars(k) = MIN_VAR + dot(resps.col(k), square(data - means(k))) / resp_count(k);
            }
            weights(k) = resp_count(k) / N;
        }

        // evaluating log-likelihood
        double curr_log_l = 0.0;
        for (size_t p = 0; p < N; ++p)
            curr_log_l += gmm_log_pdf(data(p), means, vars, weights);

        if (fabs(curr_log_l - log_l) < EPS)
            break;
        log_l = curr_log_l;
    }

    return iters;
}




size_t vb_gmm(const vec &data, size_t num_comps, size_t max_iters, vec &means, vec &vars, vec &weights, double &log_l,
              double prior_mean_m, double prior_mean_b, double prior_prec_a, double prior_prec_b, double prior_dir_u)
{
    const vec data2 = square(data);

    const size_t N = data.size();
    const size_t K = num_comps;

    // expectations
    vec exp_mean(K);
    vec exp_mean2(K);
    vec exp_prec(K);
    vec exp_log_prec(K);
    vec exp_log_weights(K);

    // responsibilities
    mat resps = zeros(N, K);
    mat log_resps(N, K);

    // initialising exp_mean using k-means
    ivec assigns;
    kmeans(data, K, max_iters, exp_mean, assigns);
    for (size_t p = 0; p < N; ++p)
        resps(p, assigns(p)) = 1.0;
    exp_mean2 = square(exp_mean);

    // mean hyperparameters
    double prior_mean_mb = prior_mean_m * prior_mean_b;
    vec mean_mb(K);
    vec mean_b(K);
    // precision hyperparameters
    vec prec_a(K);
    vec prec_b(K);
    // mixture weights hyperparameters
    vec dir_u(K);

    size_t iters = 0;
    log_l = 0;
    max_iters = 1;
    for (iters = 0; iters < max_iters; ++iters)
    {
        // updating hyperparams: prec
        for (size_t k = 0; k < K; ++k)
        {
            prec_a(k) = prior_prec_a + 0.5 * sum(resps.col(k));
            prec_b(k) = prior_prec_b + 0.5 * dot(resps.col(k), data2 - 2 * data * exp_mean(k) + exp_mean2(k));
        }
        // updating expectations
        exp_prec = prec_a / prec_b;
        exp_log_prec = digamma(prec_a) - log(prec_b);

        // updating hyperparams: mean
        for (size_t k = 0; k < K; ++k)
        {
            mean_mb(k) = prior_mean_mb + exp_prec(k) * dot(data, resps.col(k));
            mean_b(k) = prior_mean_b + exp_prec(k) * sum(resps.col(k));
        }
        // updating expectations
        exp_mean = mean_mb / mean_b;
        exp_mean2 = exp_mean + 1.0 / exp_prec;

        // updating hyperparams: assignment
        for (size_t k = 0; k < K; ++k)
            dir_u(k) = prior_dir_u + sum(resps.col(k));
        // updating expectations
        exp_log_weights = digamma(dir_u) - digamma(sum(dir_u));

        // updating assignment variables
        for (size_t k = 0; k < K; ++k)
            log_resps.col(k) = exp_log_weights(k)
                    + 0.5 * exp_log_prec(k)
                    - 0.5 * exp_prec(k) * (data2 - 2 * data * exp_mean(k) + exp_mean2(k));

        for (size_t p = 0; p < N; ++p)
        {
            resps.row(p) = exp(log_resps.row(p) - max(log_resps.row(p)));
            resps /= sum(resps.row(p));
        }

        // TODO: compute LB instead
        double curr_log_l = 0.0;
        for (size_t p = 0; p < N; ++p)
            curr_log_l += gmm_log_pdf(data(p), exp_mean, 1.0 / exp_prec, exp(exp_log_weights));
        if (fabs(curr_log_l - log_l) < EPS)
            break;
        log_l = curr_log_l;
    }

    means = exp_mean;
    vars = 1.0 / exp_prec;
    weights = exp(exp_log_weights);

    return iters;
}




