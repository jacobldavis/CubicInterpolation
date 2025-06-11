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
#include "cpp_frame/cubic_interp_eigen.h"
#include "main.h"
#include <gsl/gsl_test.h>
#include <gsl/gsl_math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    // Reads the files for input values
    double **onevalues = read_1dvalues();
    FILE *xfp = fopen("xtensor_data.csv", "w");
    FILE *efp = fopen("eigen_data.csv", "w");

    // Executes the tests for onevalues
    srand(time(NULL));
    test_all_cubic_xtensor(onevalues, xfp);
    test_all_cubic_eigen(onevalues, efp);

    // Frees onevalues
    free1d(onevalues);
    fclose(xfp);
    fclose(efp);

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

void test_cubic_xtensor(int i, double* values, FILE* fp) {
    // Initializes time recording variables and cubic_interp
    clock_t start, end;
    double cpu_time_used;
    cubic_interp *interp = cubic_interp_init_xtensor(values, n_values[i], -1, 1);

    // Iterates through the interpolation with varying xtensor sizes
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        // Precomputes random values
        xt::xtensor<double, 1> random = xt::eval(xt::random::rand<double>({c}, 0, 100));

        // Performs benchmark
        start = clock();
        volatile const xt::xtensor<double, 1> result = cubic_interp_eval_xtensor(interp, random);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], c, cpu_time_used);

        c = c * 10;
    }
    printf("\n");
    cubic_interp_free_xtensor(interp);
}

void test_all_cubic_xtensor(double** values, FILE* fp) {
    // Runs the test for all values of n
    printf("\nTesting cubic:\n");
    fprintf(fp, "Data,Iterations,Time\n");
    for (int i = 0; i < n_values_size; i++) {
        test_cubic_xtensor(i, values[i], fp);
    }
}

void test_cubic_eigen(int i, double* values, FILE* fp) {
    // Initializes time recording variables and cubic_interp
    clock_t start, end;
    double cpu_time_used;
    cubic_interp *interp = cubic_interp_init_eigen(values, n_values[i], -1, 1);

    // Iterates through the interpolation with varying xtensor sizes
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        // Precomputes random values
        Eigen::VectorXd random = (Eigen::VectorXd::Random(c) + Eigen::VectorXd::Ones(c)) * 50;

        // Performs benchmark
        start = clock();
        volatile const Eigen::VectorXd result = cubic_interp_eval_eigen(interp, random);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], c, cpu_time_used);

        c *= 10;
    }
    printf("\n");
    cubic_interp_free_xtensor(interp);
}

void test_all_cubic_eigen(double** values, FILE* fp) {
    // Runs the test for all values of n
    printf("\nTesting cubic:\n");
    fprintf(fp, "Data,Iterations,Time\n");
    for (int i = 0; i < n_values_size; i++) {
        test_cubic_eigen(i, values[i], fp);
    }
}

void free1d(double** values) {
    for (int i = 0; i < n_values_size; i++) {
        free(values[i]);
    }
    free(values);
}
