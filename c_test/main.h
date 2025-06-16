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

#include <stdio.h>

static const int n_values[5] = {4, 9, 100, 400, 1024};
static const int n_values_size = 5;
static const int iteration_values[4] = {10000, 100000, 1000000, 10000000};
static const int iteration_values_size = 4;

/* 
 * Collects arrays of the sizes in n_values from 1dvalues.txt.
 */
double **read_1dvalues();

/*
 * Frees the 1d values
 */
void free1d(double** values);

/*
 * Runs the cubic interpolation at a particular value of n.
 */
void test_cubic(int i, double* values, FILE* fp);

/*
 * Runs the cubic interpolation at every value of n in n_values.
 */
void test_all_cubic(double** values, FILE* fp);

/*
 * Runs the cubic interpolation at a particular value of n.
 */
void test_cubic_openmp(int i, double* values, FILE* fp);

/*
 * Runs the cubic interpolation at every value of n in n_values.
 */
void test_all_cubic_openacc(double** values, FILE* fp);

/*
 * Runs the cubic interpolation at a particular value of n.
 */
void test_cubic_openacc(int i, double* values, FILE* fp);

/*
 * Runs the cubic interpolation at every value of n in n_values.
 */
void test_all_cubic_openmp(double** values, FILE* fp);

/*
 * Runs the bicubic interpolation at a particular value of n.
 */
void test_bicubic(int i, double* values, FILE* fp);

/*
 * Runs the bicubic interpolation at every value of n in n_values.
 */
void test_all_bicubic(double** values, FILE* fp);

#endif 