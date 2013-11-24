#include "gmm.h"


#include "kmeans.h"

#include <vector>
using namespace std;


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
        result.subvec(filled, N - 1) =  mean(K-1) + randn(N - filled) * stddev(K-1);
    return result;
}


size_t em_gmm(const vec &data, size_t K, size_t max_iters, vec &means, vec &vars, vec &weights, double &log_l)
{
    const size_t N = data.size();

    means.resize(K);
    vars.resize(K);
    weights = ones(K) / K;

    // responsibilities
    mat resps = zeros(N, K);
    vec resp_count = zeros(K);

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
    vars /= resp_count;
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
            means(k) = dot(data, resps.col(k)) / resp_count(k);
            vars(k) = dot(resps.col(k), square(data - means(k))) / resp_count(k);
            weights(k) = resp_count(k) / N;
        }

        // evaluating log-likelihood
        double curr_log_l = 0.0;
        for (size_t p = 0; p < N; ++p)
            for (size_t k = 0; k < K; ++k)
                curr_log_l += resps(p, k) * (log(weights(k)) + log(norm_pdf(data(p), means(k), weights(k))));
        if (fabs(curr_log_l - log_l) < EPS)
            break;
        log_l = curr_log_l;
    }

    return iters;
}



size_t em_gmm_vector(const mat &data, size_t num_comps, size_t max_iters, mat &mean, mat &vars, mat &weights)
{
    size_t iters = 0;

    return iters;
}
