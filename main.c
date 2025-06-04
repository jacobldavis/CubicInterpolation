#include "c_frame/cubic_interp.h"
#include <gsl/gsl_test.h>
#include <gsl/gsl_math.h>
#include <stdio.h>
#include <assert.h>

int cubic_interp_test(void)
{
    {
        static const double data[] = {1, 0, 1, 4};
        cubic_interp *interp = cubic_interp_init(data, 4, -1, 1);
        assert(interp);
        for (double t = 0; t <= 1; t += 0.01)
        {
            const double result = cubic_interp_eval(interp, t);
            const double expected = gsl_pow_2(t);
            gsl_test_abs(result, expected, 10 * GSL_DBL_EPSILON,
                "testing cubic interpolant for quadratic input");
        }
        cubic_interp_free(interp);      
    }

        {
        static const double data[] = {1, 1, 1, 1};
        cubic_interp *interp = cubic_interp_init(data, 4, -1, 1);
        assert(interp);
        for (double t = -10; t <= 10; t += 0.01)
        {
            const double result = cubic_interp_eval(interp, t);
            const double expected = 1;
            gsl_test_abs(result, expected, 0,
                "testing cubic interpolant for unit input");
        }
        cubic_interp_free(interp);
    }

    return gsl_test_summary();
}

int main() {
    printf("Groovy %d", cubic_interp_test());
}