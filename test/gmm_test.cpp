
#include "gtest/gtest.h"

#include <vector>
#include <algorithm>
using namespace std;

#include "gmm.h"

TEST(GMMTest, EM1D)
{
    // TODO: separate errors for variance and mean?
    const double MAX_INFERENCE_ERROR = 0.5;

    vec INPUT_MEANS = {-5,5,8};
    vec INPUT_VARS = {0.2, 0.5, 0.7};
    vec INPUT_WEIGHTS = {0.2, 0.5, 0.3};
    auto data = generate_gmm(200, INPUT_MEANS, INPUT_VARS, INPUT_WEIGHTS);
    vec means, vars, weights;
    double log_l;

    means = sort(means);

    em_gmm(data, 3, 100, means, vars, weights, log_l);

    vector<pair<double, int>> idx;
    for (size_t i = 0; i < means.size(); ++i)
        idx.push_back(make_pair(means(i), i));
    std::sort(idx.begin(), idx.end(), [](pair<double,int> a, pair<double,int> b) { return a.first < b.first; });

    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(means(idx[i].second), INPUT_MEANS(i), MAX_INFERENCE_ERROR);
        EXPECT_NEAR(vars(idx[i].second), INPUT_VARS(i), MAX_INFERENCE_ERROR);
        EXPECT_NEAR(weights(idx[i].second), INPUT_WEIGHTS(i), MAX_INFERENCE_ERROR);

    }
}
