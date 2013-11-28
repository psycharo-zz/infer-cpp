
#include "gtest/gtest.h"

#include <vector>
#include <algorithm>
using namespace std;

#include "gmm.h"

TEST(GMMTest, LogPDF)
{
    const vec INPUT_MEANS = {-5,5,8};
    const vec INPUT_VARS = {0.2, 0.5, 0.7};
    const vec INPUT_WEIGHTS = {0.2, 0.5, 0.3};

    // log pdf -10:1:10
    const vec EXPECTED_LOG_PDF = {  -64.223657489421726,
                                    -41.723657489421733,
                                    -24.223657489421729,
                                    -11.723657489421724,
                                     -4.223657489421724,
                                     -1.723657489421723,
                                     -4.223657489421724,
                                    -11.723657489421724,
                                    -24.223657489416830,
                                    -37.253994875045613,
                                    -26.265512122972979,
                                    -17.265512120643507,
                                    -10.265512095548582,
                                     -5.265511637070215,
                                     -2.265497126189088,
                                     -1.264693627810608,
                                     -2.189323326616069,
                                     -2.587673412824504,
                                     -1.944330527754624,
                                     -2.658859126520253,
                                     -4.801716722227239};

    for (int i = 0, data = -10; i < EXPECTED_LOG_PDF.size(); ++i, data += 1)
        EXPECT_FLOAT_EQ(gmm_log_pdf(data, INPUT_MEANS, INPUT_VARS, INPUT_WEIGHTS), EXPECTED_LOG_PDF[i]);
}


TEST(GMMTest, LogPDFKinect)
{
    double X = 4771;
    const vec MEANS = "4.9060e+03   4.8380e+03        0e+00";
    const vec VARS = "5.0000e-06   4.0000e-06   9.9999e-06";
    const vec WEIGHTS = "0.3335   0.3727   0.2551";

    EXPECT_FLOAT_EQ(-5.61124996e+08, gmm_log_pdf(X, MEANS, VARS, WEIGHTS));
}


TEST(GMMTest, EM1D)
{
    // TODO: separate errors for variance and mean?
    const double MAX_INFERENCE_ERROR = 0.5;
    const size_t MAX_ITERS = 20;
    const size_t NUM_POINTS = 100;

    vec INPUT_MEANS = {-5,5,8};
    vec INPUT_VARS = {0.2, 0.5, 0.7};
    vec INPUT_WEIGHTS = {0.2, 0.5, 0.3};
    auto data = generate_gmm(NUM_POINTS, INPUT_MEANS, INPUT_VARS, INPUT_WEIGHTS);
    vec means, vars, weights;

    double log_l;
    em_gmm(data, INPUT_MEANS.size(), MAX_ITERS, means, vars, weights, log_l);

    vector<pair<double, int>> idx(means.size());
    for (size_t i = 0; i < means.size(); ++i)
        idx[i] = make_pair(means(i), i);
    std::sort(idx.begin(), idx.end(), [](pair<double,int> a, pair<double,int> b) { return a.first < b.first; });

    for (size_t i = 0; i < INPUT_MEANS.size(); ++i)
    {
        EXPECT_NEAR(means(idx[i].second), INPUT_MEANS(i), MAX_INFERENCE_ERROR);
        EXPECT_NEAR(vars(idx[i].second), INPUT_VARS(i), MAX_INFERENCE_ERROR);
        EXPECT_NEAR(weights(idx[i].second), INPUT_WEIGHTS(i), MAX_INFERENCE_ERROR);
    }
}


TEST(GMMTest, EMKinect)
{
    vec DATA = "4.9770e+03   4.9770e+03   5.1250e+03   5.2020e+03   5.2020e+03   5.1250e+03   4.9770e+03   5.2020e+03   5.2020e+03   5.2020e+03   5.0500e+03";
    vec means, vars, weights;
    double log_l;
    em_gmm(DATA, 3, 10, means, vars, weights, log_l);
    EXPECT_LE(log_l, DBL_MAX);
}



TEST(GMMTest, VB1D)
{
    // TODO: separate errors for variance and mean?
    const double MAX_INFERENCE_ERROR = 0.5;
    const size_t MAX_ITERS = 40;
    const size_t NUM_POINTS = 1000;

    vec INPUT_MEANS = {-5,5,8};
    vec INPUT_VARS = {0.5, 0.5, 1};
    vec INPUT_WEIGHTS = {0.2, 0.5, 0.3};
    auto data = generate_gmm(NUM_POINTS, INPUT_MEANS, INPUT_VARS, INPUT_WEIGHTS);
    vec means, vars, weights;

    double log_l;
    vb_gmm(data, INPUT_MEANS.size(), MAX_ITERS, means, vars, weights, log_l);

    vector<pair<double, int>> idx(means.size());
    for (size_t i = 0; i < means.size(); ++i)
        idx[i] = make_pair(means(i), i);
    std::sort(idx.begin(), idx.end(), [](pair<double,int> a, pair<double,int> b) { return a.first < b.first; });

    for (size_t i = 0; i < INPUT_MEANS.size(); ++i)
    {
        EXPECT_NEAR(means(idx[i].second), INPUT_MEANS(i), MAX_INFERENCE_ERROR);
        EXPECT_NEAR(vars(idx[i].second), INPUT_VARS(i), MAX_INFERENCE_ERROR);
        EXPECT_NEAR(weights(idx[i].second), INPUT_WEIGHTS(i), MAX_INFERENCE_ERROR);
    }
}


TEST(GMMTest, VBKinect)
{
    vec DATA = "4.9770e+03   4.9770e+03   5.1250e+03   5.2020e+03   5.2020e+03   5.1250e+03   4.9770e+03   5.2020e+03   5.2020e+03   5.2020e+03   5.0500e+03";
    vec means, vars, weights;
    double log_l;
    vb_gmm(DATA, 3, 20, means, vars, weights, log_l);
    EXPECT_LE(log_l, DBL_MAX);
}


