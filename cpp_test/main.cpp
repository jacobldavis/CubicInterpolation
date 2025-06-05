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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    // Read the files for input values
    std::vector<std::vector<double>> onevalues = read_1dvalues();
    std::vector<std::vector<std::vector<double>>> twovalues = read_2dvalues();

    // Execute the tests for onevalues and two values
    srand(time(NULL));
    test_all_xtensor_cubic(onevalues);
    test_all_xtensor_bicubic(twovalues);

    return 0;
}

std::vector<std::vector<double>> read_1dvalues() {
    // Reads and returns the values from 1dvalues.txt into values.
    std::vector<std::vector<double>> values;
    FILE *file = fopen("../1dvalues.txt", "r");
    for (int i = 0; i < n_values_size; i++) {
        values.push_back({});
        for (int j = 0; j < n_values[i]; j++) {
            int x = fscanf(file, "%lf, ", &values[i][j]);
        }
    }

    return values;
}

std::vector<std::vector<std::vector<double>>> read_2dvalues() {
    // Reads and returns the values from 1dvalues.txt into values.
    std::vector<std::vector<std::vector<double>>> values;
    FILE *file = fopen("../2dvalues.txt", "r");
    for (int i = 0; i < n_values_size; i++) {
        values.push_back({});
        for (int j = 0; j < n_values[i]; j++) {
            values[i].push_back({});
            for (int k = 0; k < n_values[i]; k++) {
                int x = fscanf(file, "%lf, ", &values[i][j][k]);
            }
        }
    }

    return values;
}

void test_xtensor_cubic(int i, xt::xtensor<double, 1> values) {
    // Initialize time recording variables and cubic_interp
    clock_t start, end;
    double cpu_time_used;
    cubic_interp *interp = cubic_interp_init(values, n_values[i], -1, 1);

    // Iterate through the interpolation with varying loop operation counts
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        start = clock();
        for (int t = 0; t <= c; t += 1) {
            double u = rand() * c - c/2;
            const double result = cubic_interp_eval(interp, u);
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);
        c *= 10;
    }
    printf("\n");
    cubic_interp_free(interp);
}

void test_all_xtensor_cubic(const std::vector<std::vector<double>> &values) {
    // Runs the test for all values of n
    printf("\nTesting cubic:\n");
    for (int i = 0; i < n_values_size; i++) {
        test_xtensor_cubic(i, from_vector_1d_adapt(values[i]));
    }
}

void test_xtensor_bicubic(int i, xt::xtensor<double, 3> values) {
    // Initialize time recording variables and bicubic_interp
    clock_t start, end;
    double cpu_time_used;
    bicubic_interp *interp = bicubic_interp_init(*values, n_values[i], n_values[i], -1, -1, 1, 1);

    // Iterate through the interpolation with varying loop operation counts
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        int iter = sqrt(c);
        start = clock();
        for (double s = 0; s <= iter; s += 1) {
            for (double t = 0; t <= iter; t += 1) {
                double u = rand() * c - c/2;
                double v = rand() * c - c/2;
                const double result = bicubic_interp_eval(interp, u, v);
            }
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);
        c *= 10;
    }
    printf("\n");
    bicubic_interp_free(interp);
}

void test_all_xtensor_bicubic(const std::vector<std::vector<std::vector<double>>> &values) {
    // Runs the test for all values of n
    printf("\nTesting bicubic:\n");
    for (int i = 0; i < n_values_size; i++) {
        test_xtensor_bicubic(i, from_vector_2d(values[i]));
    }
}

xt::xtensor<double, 2> from_vector_2d(const std::vector<std::vector<double>>& vec) {
    size_t rows = vec.size();
    size_t cols = rows > 0 ? vec[0].size() : 0;
    xt::xtensor<double, 2> result({rows, cols});

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = vec[i][j];
        }
    }

    return result;
}

xt::xtensor<double, 1> from_vector_1d_adapt(const std::vector<double>& vec) {
    return xt::adapt(vec.data(), {vec.size()});
}