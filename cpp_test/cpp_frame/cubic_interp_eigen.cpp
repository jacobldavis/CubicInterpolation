/*
 * Copyright (C) 2015-2024  Leo Singer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "cubic_interp_eigen.h"
#include "branch_prediction.h"
#include "vmath_eigen.h"
#include <math.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>

/* Allow contraction of a * b + c to a faster fused multiply-add operation.
 * This pragma is supposedly standard C, but only clang seems to support it.
 * On other compilers, floating point contraction is ON by default at -O3. */
#if defined(__clang__) || defined(__llvm__)
#pragma STDC FP_CONTRACT ON
#endif

#define VCUBIC(a, t) (t * (t * (t * a[0] + a[1]) + a[2]) + a[3])


struct cubic_interp {
    double f, t0, length;
    xt::xtensor<double, 2> a;
};


/*
 * Calculate coefficients of the interpolating polynomial in the form
 *      a[0] * t^3 + a[1] * t^2 + a[2] * t + a[3]
 */
static void cubic_interp_init_coefficients(
    double *a, const double *z, const double *z1)
{
    if (UNLIKELY(!isfinite(z1[1] + z1[2])))
    {
        /* If either of the inner grid points are NaN or infinite,
         * then fall back to nearest-neighbor interpolation. */
        a[0] = 0;
        a[1] = 0;
        a[2] = 0;
        a[3] = z[1];
    } else if (UNLIKELY(!isfinite(z1[0] + z1[3]))) {
        /* If either of the outer grid points are NaN or infinite,
         * then fall back to linear interpolation. */
        a[0] = 0;
        a[1] = 0;
        a[2] = z[2] - z[1];
        a[3] = z[1];
    } else {
        /* Otherwise, all of the grid points are finite.
         * Use cubic interpolation. */
        a[0] = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]);
        a[1] = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3];
        a[2] = 0.5 * (z[2] - z[0]);
        a[3] = z[1];
    }
}

// Returns an array of cubic_interp structs based on the input data array
cubic_interp *cubic_interp_init_eigen(
    const double *data, int n, double tmin, double dt)
{
    const int length = n + 6;
    cubic_interp *interp = new cubic_interp;
    if (LIKELY(interp))
    {
        interp->f = 1 / dt;
        interp->t0 = 3 - interp->f * tmin;
        interp->length = length;
        interp->a = xt::eval(xt::zeros<double>({length, 4}));
        for (int i = 0; i < length; i ++)
        {
            double z[4];
            for (int j = 0; j < 4; j ++)
            {
                z[j] = data[VCLIP(i + j - 4, 0, n - 1)];
            }
            if (UNLIKELY(!isfinite(z[1] + z[2]))) {
                xt::row(interp->a, i) = xt::eval(xt::xtensor<double, 1>{0, 0, 0, z[1]});
            } else if (UNLIKELY(!isfinite(z[0] + z[3]))) {
                xt::row(interp->a, i) = xt::eval(xt::xtensor<double, 1>{0, 0, z[2]-z[1], z[1]});
            } else {
                xt::row(interp->a, i) = xt::eval(xt::xtensor<double, 1>{1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]),
                                                               z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3], 
                                                               0.5 * (z[2] - z[0]), z[1]});
            }
        }
    }
    return interp;
}


void cubic_interp_free_eigen(cubic_interp *interp)
{
    free(interp);
}


xt::xtensor<double, 1> cubic_interp_eval_eigen(const cubic_interp *interp, xt::xtensor<double, 1> t)
{
    double xmin = 0.0, xmax = interp->length - 1.0;

    t *= interp->f;
    t += interp->t0;

    t = xt::eval(xt::minimum(xt::maximum(t, xmin), xmax));

    xt::xtensor<int, 1> ix = xt::floor(t);
    t -= ix;
    
    return (t * (t * 
           (t * xt::view(interp->a, xt::keep(ix), 0)
            + xt::view(interp->a, xt::keep(ix), 1))
            + xt::view(interp->a, xt::keep(ix), 2))
            + xt::view(interp->a, xt::keep(ix), 3));
}
