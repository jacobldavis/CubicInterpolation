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

#include "c_frame/cubic_interp.h"
#include <gsl/gsl_test.h>
#include <gsl/gsl_math.h>
#include <assert.h>
#include <stdio.h>

int main() {
    // Welcomes the profiler.
    printf("\n--- Welcome to profiling! ---\n");

    // Reads in the cubic interpolation values from values.txt.

    // Selects to run the cubic interpolation at a particular value or all values of n.
    int c = 0;
    printf("\nWhat value(s) of n do you want to test?\n");
    printf("0: 10, 1: 100, 2: 1000, 3: 10000, 4: 100000, 5: 1000000 6: All\n");

}


double **read_values() {

}


void test_n(int n, double* values) {

}


void test_all_n(double** values) {

}

double run_interpolate(int c) {
    return 0.0;
}