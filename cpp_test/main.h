/*
 * Copyright (C) 2025 Jacob Davis
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
 * The purpose of this program is to control profiling
 * different frameworks for a cubic interpolation.
 */

#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>

static const int n_values[5] = {4, 10, 100, 400, 1000};
static const int n_values_size = 5;

/* 
 * Collects vectors of the sizes in n_values from 1dvalues.txt.
 */
std::vector<std::vector<double>> read_1dvalues();

/* 
 * Collects arrays of the sizes in n_values from 2dvalues.txt.
 */
double ***read_2dvalues();

/*
 * Runs the cubic interpolation at a particular value of n.
 */
void test_xtensor_cubic(int i, std::vector<double> &values);

/*
 * Runs the cubic interpolation at every value of n in n_values.
 */
void test_all_xtensor_cubic(std::vector<std::vector<double>> &values);

/*
 * Runs the bicubic interpolation at a particular value of n.
 */
void test_xtensor_bicubic(int i, double** values);

/*
 * Runs the bicubic interpolation at every value of n in n_values.
 */
void test_all_xtensor_bicubic(double*** values);

/*
 * Frees the 2d values
 */
void free2d(double*** values);
