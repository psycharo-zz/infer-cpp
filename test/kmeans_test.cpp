#include "gtest/gtest.h"

#include "kmeans.h"
#include "gmm.h"


TEST(KMeansTest, 1D)
{
    auto data = generate_gmm(100, {-5,5}, {0.1, 0.1},  {0.5, 0.5});
    vec means;
    ivec assigns;
    kmeans(data, 2, 100, means, assigns);

    means = sort(means);
    EXPECT_NEAR(means(0), -5, 0.5);
    EXPECT_NEAR(means(1), 5, 0.5);
}
