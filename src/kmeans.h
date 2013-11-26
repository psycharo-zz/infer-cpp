#ifndef KMEANS_H
#define KMEANS_H

#include "mathutil.h"


#include <cfloat>

/**
 * K-means algorithm
 * @param K the number of clusters
 * @param means cluster centers
 * @param counts sizes of clusters
 * @param resps assignments to clusters
 * @return the number of iterations till convergence
 */
inline size_t kmeans(const vec &data, size_t K, size_t max_iters, vec &means, ivec &assigns, double EPS = 1.0e-5)
{
    const size_t N = data.size();

    ivec counts(K);
    vec new_means(K);

    means.resize(K);
    assigns.resize(N);

    // random initialisation
    for (size_t k = 0; k < K; ++k)
        means[k] = data[rand() % N];

    size_t iters = 0;
    bool converged = false;
    for (iters = 0; (iters < max_iters) && !converged; ++iters)
    {
        new_means.fill(0.0);
        counts.fill(0);
        for (size_t p = 0; p < N; ++p)
        {
            size_t cluster = 0;
            double cluster_dist = DBL_MAX;
            for (size_t k = 0; k < K; ++k)
            {
                double dist = fabs(data[p] - means[k]);
                if (dist <= cluster_dist)
                {
                    cluster = k;
                    cluster_dist = dist;
                }
            }
            assigns[p] = cluster;
            ++counts[cluster];
            new_means[cluster] += data[p];
        }
        converged = true;
        for (size_t k = 0; k < K; ++k)
        {
            if (counts[k] == 0)
                continue;
            double m = new_means[k] / counts[k];
            converged &= fabs(m - means[k]) < EPS;
            means[k] = m;
        }
   }
   return iters;
}

#endif // KMEANS_H
