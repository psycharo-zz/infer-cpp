#include "gtest/gtest.h"

#include "mathutil.h"
using namespace infer;

TEST(MathUtilTest, LogPDF)
{
    const double MEAN = 4;   // mu
    const double VAR = 0.25; // sigma^2

    // logpdf([0;10), mu=4,sigma^2=0.25)
    const vec EXPECTED = {-32.2257913526,-18.2257913526,-8.22579135264, -2.22579135264,
                          -0.225791352645,-2.22579135264,-8.22579135264,-18.2257913526,
                          -32.2257913526,-50.2257913526};
    for (int i = 0; i < 10; ++i)
        EXPECT_FLOAT_EQ(norm_log_pdf(i, MEAN, VAR), EXPECTED[i]);
}


TEST(MathUtilTest, DiGamma)
{
    const vec DATA = {1,2,3,4,5};
    const vec EXPECTED = "-0.57721566  0.42278434  0.92278434  1.25611767  1.50611767";

    vec RESULT = digamma(DATA);
    for (size_t i = 0; i < DATA.size(); ++i)
        EXPECT_FLOAT_EQ(EXPECTED(i), RESULT(i));
}
