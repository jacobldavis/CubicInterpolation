/*
 * Copyright (C) 2015-2024  Leo Singer & Jacob Davis
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

#include "cubic_interp_xtensor.h"
#include "branch_prediction.h"
#include "vmath_xtensor.h"
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
    vd2f a;
};


struct bicubic_interp {
    vdf fx, x0, xlength;
    vd2f a;
};

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

cubic_interp *cubic_interp_init(
    const std::vector<double> &data, int n, double tmin, double dt)
{
    const int length = n + 6;
    cubic_interp *interp = new cubic_interp;
    if (LIKELY(interp))
    {
        interp->f = 1 / dt;
        interp->t0 = 3 - interp->f * tmin;
        interp->length = length;
        interp->a = xt::zeros<double>({length, 4});
        for (int i = 0; i < length; i ++)
        {
            double z[4];
            for (int j = 0; j < 4; j ++)
            {
                z[j] = data[clip(i + j - 4, 0, n - 1)];
            }
            if (UNLIKELY(!isfinite(z[1] + z[2]))) {
                xt::row(interp->a, i) = xt::xtensor<double, 1>{0, 0, 0, z[1]};
            } else if (UNLIKELY(!isfinite(z[0] + z[3]))) {
                xt::row(interp->a, i) = xt::xtensor<double, 1>{0, 0, z[2]-z[1], z[1]};
            } else {
                xt::row(interp->a, i) = xt::xtensor<double, 1>{1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]),
                                                               z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3], 
                                                               0.5 * (z[2] - z[0]), z[1]};
            }
        }
    }
    return interp;
}

void cubic_interp_free(cubic_interp *interp)
{
    free(interp);
}

double cubic_interp_eval(const cubic_interp *interp, double t)
{
    if (UNLIKELY(isnan(t)))
        return t;

    double x = t, xmin = 0.0, xmax = interp->length - 1.0;
    x *= interp->f;
    x += interp->t0;
    x = clip(x, xmin, xmax);

    double ix = double_floor(x);
    x -= ix;

    auto a = xt::row(interp->a, static_cast<int>(ix));
    return x * (x * (x * a(0) + a(1)) + a(2)) + a(3);
}

// --------------
// Bicubic Interp
// --------------

bicubic_interp *bicubic_interp_init(
    const double *data, int ns, int nt,
    double smin, double tmin, double ds, double dt)
{
    const int slength = ns + 6;
    const int tlength = nt + 6;
    bicubic_interp *interp = new bicubic_interp;
    if (LIKELY(interp))
    {
        interp->fx = {1/ds, 1/dt};
        interp->x0= {3 - interp->fx[0] * smin, 3 - interp->fx[1] * tmin};
        interp->xlength = {1.0 * slength, 1.0 * tlength};
        interp->a = xt::zeros<double>({slength * tlength, 4});

        for (int is = 0; is < slength; is ++)
        {
            for (int it = 0; it < tlength; it ++)
            {
                double a[4][4], a1[4][4];
                for (int js = 0; js < 4; js ++)
                {
                    double z[4];
                    int ks = clip(is + js - 4, 0, ns - 1);
                    for (int jt = 0; jt < 4; jt ++)
                    {
                        int kt = clip(it + jt - 4, 0, nt - 1);
                        z[jt] = data[ks * ns + kt];
                    }
                    cubic_interp_init_coefficients(a[js], z, z);
                }
                for (int js = 0; js < 4; js ++)
                {
                    for (int jt = 0; jt < 4; jt ++)
                    {
                        a1[js][jt] = a[jt][js];
                    }
                }
                for (int js = 0; js < 4; js ++)
                {
                    cubic_interp_init_coefficients(a[js], a1[js], a1[3]);
                }
                xt::row(interp->a, is * tlength + it) = xt::xtensor<double, 1>{a[0][0], a[0][1], a[0][2], a[0][3]};
            }
        }
    }
    return interp;
}


void bicubic_interp_free(bicubic_interp *interp)
{
    free(interp);
}


double bicubic_interp_eval(const bicubic_interp *interp, double s, double t)
{
    if (UNLIKELY(isnan(s) || isnan(t)))
        return s + t;

    vdf x = {s, t}, xmin = {0.0, 0.0}, xmax = interp->xlength - 1.0;
    x *= interp->fx;
    x += interp->x0;
    x = clip(x, xmin, xmax);

    vdf ix = vdf_floor(x);
    x -= ix;

    int idx = static_cast<int>(ix[0] * interp->xlength[1] + ix[1]);
    auto coeffs = xt::row(interp->a, idx);

    xt::xtensor<double, 1> b = xt::zeros<double>({4});
    for (int i = 0; i < 4; ++i) {
        const double* row = &coeffs(i * 4);  
        b(i) = VCUBIC(row, x[1]);            
    }

    return VCUBIC(b.data(), x[0]);
}
