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

#ifndef MAIN_H
#define MAIN_H

#ifndef __cplusplus

static const int n_values[6] = {10, 100, 1000, 10000, 100000, 1000000};

 /* 
  * Collects arrays of the sizes in n_values from values.txt.
  */
double **read_values();

/*
 * Runs the cubic interpolation at a particular value of n.
 */
void test_n(int n, double* values);

/*
 * Runs the cubic interpolation at every value of n in n_values.
 */
void test_all_n(double** values);

/*
 * Tests a particular implementation of cubic interpolation.
 * Plans include the following frameworks: xtensor, Eigen, CUDA, and OpenCL.
 */
double run_interpolate(int c);

#endif /* __cplusplus */

#endif /* CUBIC_INTERP_H */