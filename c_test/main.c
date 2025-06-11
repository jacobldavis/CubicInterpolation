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
#include "c_frame/kernel.h"
#include "c_frame/cubic_interp_cl.h"
#include "main.h"
#include <gsl/gsl_test.h>
#include <gsl/gsl_math.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    // Reads the files for input values and creates a csv
    double **onevalues = read_1dvalues();
    double ***twovalues = read_2dvalues();
    FILE* cfp = fopen("c_data.csv", "w");
    FILE* cudafp = fopen("cuda_data.csv", "w");
    FILE* clfp = fopen("cl_data.csv", "w");

    // Executes the tests for onevalues and two values
    srand(time(NULL));
    test_all_cubic(onevalues, cfp);
    test_all_bicubic(twovalues, cfp);
    test_all_cubic_cuda(onevalues, cudafp);
    test_all_cubic_cl(onevalues, clfp);

    // Frees onevalues and twovalues
    free1d(onevalues);
    free2d(twovalues);
    fclose(cfp);
    fclose(cudafp);
    fclose(clfp);

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

void test_cubic(int i, double* values, FILE* fp) {
    // Initializes time recording variables and cubic_interp
    clock_t start, end;
    double cpu_time_used;
    cubic_interp *interp = cubic_interp_init(values, n_values[i], -1, 1);

    // Iterates through the interpolation with varying loop operation counts
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        // Precomputes random values
        double* random = (double*)malloc(c * sizeof(double));
        for (int k = 0; k < c; k++) {
            random[k] = rand() * 100;
        }

        // Performs benchmark
        start = clock();
        for (int t = 0; t <= c; t += 1) {
            volatile const double result = cubic_interp_eval(interp, random[t]);
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], c, cpu_time_used);

        c *= 10;
        free(random);
    }
    printf("\n");
    cubic_interp_free(interp);
}

void test_all_cubic(double** values, FILE* fp) {
    // Runs the test for all values of n
    printf("\nTesting cubic:\n");
    fprintf(fp, "Data,Iterations,Time\n");
    for (int i = 0; i < n_values_size; i++) {
        test_cubic(i, values[i], fp);
    }
}

void test_bicubic(int i, double** values, FILE* fp) {
    // Initializes time recording variables and bicubic_interp
    clock_t start, end;
    double cpu_time_used;
    bicubic_interp *interp = bicubic_interp_init(*values, n_values[i], n_values[i], -1, -1, 1, 1);

    // Iterates through the interpolation with varying loop operation counts
    int c = 10000;
    for (int m = 1; m < 5; m++) {
        // Precomputes random values
        double* randomu = (double*)malloc(c * sizeof(double));
        double* randomv = (double*)malloc(c * sizeof(double));
        for (int k = 0; k < c; k++) {
            randomu[k] = rand() * 100;
            randomv[k] = rand() * 100;
        }
        
        // Performs benchmark
        int iter = sqrt(c);
        start = clock();
        for (int s = 0; s <= iter; s += 1) {
            for (int t = 0; t <= iter; t += 1) {
                volatile const double result = bicubic_interp_eval(interp, randomu[s * iter + t], randomv[s * iter + t]);
            }
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, cpu_time_used);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], c, cpu_time_used);

        c *= 10;
        free(randomu);
        free(randomv);
    }
    printf("\n");
    bicubic_interp_free(interp);
}

void test_all_bicubic(double*** values, FILE* fp) {
    // Runs the test for all values of n
    printf("\nTesting bicubic:\n");
    for (int i = 0; i < n_values_size; i++) {
        test_bicubic(i, values[i], fp);
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