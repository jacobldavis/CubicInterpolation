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

#include "cpp_frame/cubic_interp_xtensor.h"
#include "main.h"
#include <gsl/gsl_test.h>
#include <gsl/gsl_math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    // Reads the files for input values
    double **onevalues = read_1dvalues();
    double ***twovalues = read_2dvalues();

    // Executes the tests for onevalues and two values
    srand(time(NULL));
    test_all_cubic_xtensor(onevalues);
    test_all_bicubic_xtensor(twovalues);

    // Frees onevalues and twovalues
    free1d(onevalues);
    free2d(twovalues);

    return 0;
}

double **read_1dvalues() {
    // Allocates memory for values based on n_values.
    double** values = (double**)malloc(n_values_size * sizeof(double*));
    for (int i = 0; i < n_values_size; i++) {
        values[i] = (double*)malloc(n_values[i] * sizeof(double));
    }

    // Reads and returns the values from 1dvalues.txt into values.
    FILE *file = fopen("../1dvalues.txt", "r");
    for (int i = 0; i < n_values_size; i++) {
        for (int j = 0; j < n_values[i]; j++) {
            int x = fscanf(file, "%lf, ", &values[i][j]);
        }
    }

    return values;
}

double ***read_2dvalues() {
    // Allocates memory for values based on n_values.
    double*** values = (double***)malloc(n_values_size * sizeof(double**));
    for (int i = 0; i < n_values_size; i++) {
        values[i] = (double**)malloc(n_values[i] * sizeof(double*));
        for (int j = 0; j < n_values[i]; j++) {
            values[i][j] = (double*)malloc(n_values[i] * sizeof(double));
        }
    }

    // Reads and returns the values from 1dvalues.txt into values.
    FILE *file = fopen("../2dvalues.txt", "r");
    for (int i = 0; i < n_values_size; i++) {
        for (int j = 0; j < n_values[i]; j++) {
            for (int k = 0; k < n_values[i]; k++) {
                int x = fscanf(file, "%lf, ", &values[i][j][k]);
            }
        }
    }

    return values;
}

void test_cubic_xtensor(int i, double* values) {
    // Initializes time recording variables and cubic_interp
    clock_t start, end;
    double cpu_time_used;
    cubic_interp *interp = cubic_interp_init(values, n_values[i], -1, 1);

    // Iterates through the interpolation with varying loop operation counts
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        // Precomputes random values
        xt::xtensor<double, 1> random = xt::random::rand(c, 0, c);

        // Performs benchmark
        start = clock();
        volatile const double result = cubic_interp_eval(interp, random);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);

        c *= 10;
    }
    printf("\n");
    cubic_interp_free(interp);
}

void test_all_cubic_xtensor(double** values) {
    // Runs the test for all values of n
    printf("\nTesting cubic:\n");
    for (int i = 0; i < n_values_size; i++) {
        test_cubic_xtensor(i, values[i]);
    }
}

void test_bicubic_xtensor(int i, double** values) {
    // Initializes time recording variables and bicubic_interp
    clock_t start, end;
    double cpu_time_used;
    bicubic_interp *interp = bicubic_interp_init(*values, n_values[i], n_values[i], -1, -1, 1, 1);

    // Iterates through the interpolation with varying loop operation counts
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        // Precomputes random values
        xt::xtensor<double, 1> randomu = xt::random::rand(c, 0, c);
        xt::xtensor<double, 1> randomv = xt::random::rand(c, 0, c);

        // Performs benchmark
        start = clock();
        volatile const double result = bicubic_interp_eval(interp, randomu, randomv);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);

        c *= 10;
    }
    printf("\n");
    bicubic_interp_free(interp);
}

void test_all_bicubic_xtensor(double*** values) {
    // Runs the test for all values of n
    printf("\nTesting bicubic:\n");
    for (int i = 0; i < n_values_size; i++) {
        test_bicubic_xtensor(i, values[i]);
    }
}

void free1d(double** values) {
    for (int i = 0; i < n_values_size; i++) {
        free(values[i]);
    }
    free(values);
}

void free2d(double*** values) {
    for (int i = 0; i < n_values_size; i++) {
        for (int j = 0; j < n_values[i]; j++) {
            free(values[i][j]);
        }
        free(values[i]);
    }
    free(values);
}