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
 *
 *
 *      Accelerated 1D and 2D cubic interpolation
 *
 *      1. Constant boundary conditions
 *
 *      2. Robust to invalid data: drops to linear or nearest-neighbor
 *         when the input data contains NaNs or infinities
 *
 *      3. Bounds and invalid value checks are precomputed:
 *         minimal branch instructions needed in evaluate function
 *
 *      4. Interpolating polynomial is precomputed.
 *
 *         For 1D interpolation, direct evaluation of the inteprolating
 *         polynomial from the data takes 9 multiplications and 10 additions;
 *         with precomputed coefficients we need only 3 multiplications and 3
 *         additions at the cost of 4x the memory footprint.
 *
 *         For 2D interpolation, direct evaluation of the inteprolating
 *         polynomial from the data takes 18 multiplications and 20 additions;
 *         with precomputed coefficients we need only 6 multiplications and 6
 *         additions at the cost of 16x the memory footprint.
 */


#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xsimd/xsimd.hpp>

typedef struct cubic_interp cubic_interp;
typedef struct bicubic_interp bicubic_interp;

__attribute__ ((malloc))
cubic_interp *cubic_interp_init_xtensor(
    const double *data, int n, double tmin, double dt);

void cubic_interp_free_xtensor(cubic_interp *interp);

__attribute__ ((pure))
xt::xtensor<double, 1> cubic_interp_eval_xtensor(const cubic_interp *interp, xt::xtensor<double, 1>& t_in);
