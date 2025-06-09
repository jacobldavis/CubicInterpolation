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
#include "vmath.h"
#include <math.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

/* Allow contraction of a * b + c to a faster fused multiply-add operation.
 * This pragma is supposedly standard C, but only clang seems to support it.
 * On other compilers, floating point contraction is ON by default at -O3. */
#if defined(__clang__) || defined(__llvm__)
#pragma STDC FP_CONTRACT ON
#endif

#define VCUBIC(a, t) (t * (t * (t * a[0] + a[1]) + a[2]) + a[3])


struct cubic_interp {
    double f, t0, length;
    Eigen::MatrixXd a;
};

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
        interp->a.resize(length, 4);
        for (int i = 0; i < length; i ++)
        {
            double z[4];
            for (int j = 0; j < 4; j ++)
            {
                z[j] = data[VCLIP(i + j - 4, 0, n - 1)];
            }
            if (UNLIKELY(!isfinite(z[1] + z[2]))) {
                interp->a.row(i) = Eigen::RowVector4d(0, 0, 0, z[1]);
            } else if (UNLIKELY(!isfinite(z[0] + z[3]))) {
                interp->a.row(i) = Eigen::RowVector4d(0, 0, z[2]-z[1], z[1]);
            } else {
                interp->a.row(i) = Eigen::RowVector4d(1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]), 
                                                      z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3], 
                                                      0.5 * (z[2] - z[0]), z[1]);
            }
        }
    }
    return interp;
}


void cubic_interp_free_eigen(cubic_interp *interp)
{
    free(interp);
}

Eigen::VectorXd cubic_interp_eval_eigen(const cubic_interp* interp, Eigen::VectorXd t)
{
    double xmin = 0.0, xmax = static_cast<double>(interp->length - 1);

    t *= interp->f;
    t = t.array() + interp->t0;

    t = t.array().min(xmax).max(xmin);

    Eigen::VectorXd ix = t.array().floor();
    t -= ix;

    Eigen::MatrixXd ai = interp->a(ix, Eigen::all);

    return (t.array() * (t.array() * (t.array() * ai.col(0).array() + ai.col(1).array()) + ai.col(2).array()) + ai.col(3).array()).matrix();
}
