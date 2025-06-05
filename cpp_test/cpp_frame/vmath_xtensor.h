/*
 * Copyright (C) 2025  Leo Singer & Jacob Davis
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

#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <math.h>
#include <algorithm>

// Define vector types using xtensor
using vdf = xt::xtensor<double, 1>;
using vd2f = xt::xtensor<double, 2>;

// Vectorized math operations for vdf
inline vdf vdf_min(const vdf& a, const vdf& b) {
    return xt::minimum(a, b);
}

inline vdf vdf_max(const vdf& a, const vdf& b) {
    return xt::maximum(a, b);
}

inline vdf vdf_floor(const vdf& a) {
    return xt::floor(a);
}

// Scalar operations for int and double
inline int int_min(int a, int b) {
    return std::min(a, b);
}

inline int int_max(int a, int b) {
    return std::max(a, b);
}

inline double double_min(double a, double b) {
    return std::fmin(a, b);
}

inline double double_max(double a, double b) {
    return std::fmax(a, b);
}

inline double double_floor(double a) {
    return std::floor(a);
}

// Clip operations
inline vdf clip(const vdf& x, const vdf& a, const vdf& b) {
    return vdf_min(vdf_max(x, a), b);
}

inline int clip(int x, int a, int b) {
    return int_min(int_max(x, a), b);
}

inline double clip(double x, double a, double b) {
    return double_min(double_max(x, a), b);
}


