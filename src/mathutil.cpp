#include "mathutil.h"

using namespace infer;


/* The digamma function is the derivative of gammaln.

   Reference:
    J Bernardo,
    Psi ( Digamma ) Function,
    Algorithm AS 103,
    Applied Statistics,
    Volume 25, Number 3, pages 315-317, 1976.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modifications for negative numbers and extra precision)
*/
double infer::digamma(double x)
{
    double neginf = -INFINITY;
    static const double c = 12,
            digamma1 = -0.57721566490153286,
            trigamma1 = 1.6449340668482264365, /* pi^2/6 */
            s = 1e-6,
            s3 = 1./12,
            s4 = 1./120,
            s5 = 1./252,
            s6 = 1./240,
            s7 = 1./132,
            s8 = 691./32760,
            s9 = 1./12,
            s10 = 3617./8160;
    double result;
    /* Illegal arguments */
    if((x == neginf) || std::isnan(x)) {
        return NAN;
    }
    /* Singularities */
    if((x <= 0) && (floor(x) == x)) {
        return neginf;
    }
    /* Negative values */
    /* Use the reflection formula (Jeffrey 11.1.6):
   * digamma(-x) = digamma(x+1) + pi*cot(pi*x)
   *
   * This is related to the identity
   * digamma(-x) = digamma(x+1) - digamma(z) + digamma(1-z)
   * where z is the fractional part of x
   * For example:
   * digamma(-3.1) = 1/3.1 + 1/2.1 + 1/1.1 + 1/0.1 + digamma(1-0.1)
   *               = digamma(4.1) - digamma(0.1) + digamma(1-0.1)
   * Then we use
   * digamma(1-z) - digamma(z) = pi*cot(pi*z)
   */
    if(x < 0) {
        return digamma(1-x) + M_PI/tan(-M_PI*x);
    }
    /* Use Taylor series if argument <= S */
    if(x <= s) return digamma1 - 1/x + trigamma1*x;
    /* Reduce to digamma(X + N) where (X + N) >= C */
    result = 0;
    while(x < c) {
        result -= 1/x;
        x++;
    }
    /* Use de Moivre's expansion if argument >= C */
    /* This expansion can be computed in Maple via asympt(Psi(x),x) */
    if(x >= c) {
        double r = 1/x, t;
        result += log(x) - 0.5*r;
        r *= r;
#if 0
        result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
#else
        /* this version for lame compilers */
        t = (s5 - r * (s6 - r * s7));
        result -= r * (s3 - r * (s4 - r * t));
#endif
    }
    return result;
}


vec infer::digamma(const vec &v)
{
    vec res = v;
    for (double &el : res)
        el = digamma(el);
    return res;
}
