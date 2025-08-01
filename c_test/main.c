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
#include "c_frame/cubic_interp_cuda.h"
#include "c_frame/cubic_interp_cl.h"
#include "main.h"
#include "c_frame/vmath.h"
#include <gsl/gsl_test.h>
#include <gsl/gsl_math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define VCUBIC(a, t) (t * (t * (t * a[0] + a[1]) + a[2]) + a[3])

int main(int argc, char **argv) {
    // Reads the files for input values and creates a csv
    double **onevalues = read_1dvalues();
    FILE* cfp = fopen("../results/c_data.csv", "w");
    FILE* bicfp = fopen("../results/bi_c_data.csv", "w");
    FILE* cudafp = fopen("../results/cuda_data.csv", "w");
    FILE* clfp = fopen("../results/opencl_data.csv", "w");
    FILE* ompfp = fopen("../results/openmp_data.csv", "w");
    FILE* ompcpufp = fopen("../results/cpu_openmp_data.csv", "w");

    // Executes the tests for onevalues and two values
    srand(time(NULL));
    test_all_cubic(onevalues, cfp);
    test_all_bicubic(onevalues, bicfp);
    test_all_cubic_cuda(onevalues, cudafp);
    test_all_cubic_cl(onevalues, clfp);
    test_all_cubic_openmp(onevalues, ompfp);
    test_all_cubic_openmp_cpu(onevalues, ompcpufp);

    // Frees onevalues and twovalues and closes files
    free1d(onevalues);
    fclose(cfp);
    fclose(bicfp);
    fclose(cudafp);
    fclose(clfp);
    fclose(ompfp);
    fclose(ompcpufp);

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

void test_cubic(int i, double* values, FILE* fp) {
    // Initializes time recording variables and cubic_interp
    clock_t start, end;
    double cpu_time_used = 0.0;
    cubic_interp *interp = cubic_interp_init(values, n_values[i], -1, 1);

    // Iterates through the interpolation with varying loop operation counts
    for (int m = 0; m < iteration_values_size; m++) {
        // Precomputes random values
        double* random = (double*)malloc(iteration_values[m] * sizeof(double));
        for (int k = 0; k < iteration_values[m]; k++) {
            random[k] = rand() * 100;
        }

        // Performs benchmark
        int a = iteration_values[m];
        for (int l = 0; l < trials; l++) {
            start = clock();
            for (int t = 0; t < a; t += 1) {
                volatile const double result = cubic_interp_eval(interp, random[t]);
            }
            end = clock();
            cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
        }
        cpu_time_used /= trials;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], iteration_values[m], cpu_time_used);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], iteration_values[m], cpu_time_used);

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

void test_cubic_openmp(int i, double* values, FILE* fp) {
    // Initializes cubic_interp and elapsed_time
    cubic_interp *interp = cubic_interp_init(values, n_values[i], -1, 1);
    double elapsed_time = 0.0;

    // Iterates through the interpolation with varying loop operation counts
    for (int m = 0; m < iteration_values_size; m++) {
        // Precomputes random values
        double* random = (double*)malloc(iteration_values[m] * sizeof(double));
        for (int k = 0; k < iteration_values[m]; k++) {
            random[k] = rand() * 100;
        }

        // Sets necessary values
        int f = interp->f;
        int t0 = interp->t0;
        int length = interp->length;
        double (*a)[4] = interp->a;
        int n = iteration_values[m];
        double* coeffs;
        double xmin = 0.0, xmax = interp->length - 1.0;

        // Performs benchmark
        for (int l = 0; l < trials; l++) {
            double start = omp_get_wtime();
            #pragma omp target
            for (int t = 0; t < n; t++) {
                double x = random[t] * f + t0;
                x = VCLIP(x, xmin, xmax);            
                int ix = VFLOOR(x);
                x -= ix;
                coeffs = a[ix];
                volatile double result = VCUBIC(coeffs, x);
            }
            #pragma omp barrier
            double end = omp_get_wtime();
            elapsed_time += end - start;
        }
        elapsed_time /= trials;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], iteration_values[m], elapsed_time);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], iteration_values[m], elapsed_time);

        free(random);
    }
    printf("\n");
    cubic_interp_free(interp);
}

void test_all_cubic_openmp(double** values, FILE* fp) {
    // Runs the test for all values of n
    printf("\nTesting openmp cubic:\n");
    fprintf(fp, "Data,Iterations,Time\n");
    for (int i = 0; i < n_values_size; i++) {
        test_cubic_openmp(i, values[i], fp);
    }
}

void test_cubic_openmp_cpu(int i, double* values, FILE* fp) {
    // Initializes cubic_interp and elapsed_time
    cubic_interp *interp = cubic_interp_init(values, n_values[i], -1, 1);
    double elapsed_time = 0.0;

    // Iterates through the interpolation with varying loop operation counts
    for (int m = 0; m < iteration_values_size; m++) {
        // Precomputes random values
        double* random = (double*)malloc(iteration_values[m] * sizeof(double));
        for (int k = 0; k < iteration_values[m]; k++) {
            random[k] = rand() * 100;
        }

        // Sets necessary values
        int f = interp->f;
        int t0 = interp->t0;
        int length = interp->length;
        double (*a)[4] = interp->a;
        int n = iteration_values[m];
        double* coeffs;
        double xmin = 0.0, xmax = interp->length - 1.0;

        // Performs benchmark
        for (int l = 0; l < trials; l++) {
            double start = omp_get_wtime();
            #pragma omp parallel for
            for (int t = 0; t < n; t++) {
                double x = random[t] * f + t0;
                x = VCLIP(x, xmin, xmax);            
                int ix = VFLOOR(x);
                x -= ix;
                coeffs = a[ix];
                volatile double result = VCUBIC(coeffs, x);
            }
            #pragma omp barrier
            double end = omp_get_wtime();
            elapsed_time += end - start;
        }
        elapsed_time /= trials;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], iteration_values[m], elapsed_time);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], iteration_values[m], elapsed_time);

        free(random);
    }
    printf("\n");
    cubic_interp_free(interp);
}

void test_all_cubic_openmp_cpu(double** values, FILE* fp) {
    // Runs the test for all values of n
    printf("\nTesting openmp cpu cubic:\n");
    fprintf(fp, "Data,Iterations,Time\n");
    for (int i = 0; i < n_values_size; i++) {
        test_cubic_openmp_cpu(i, values[i], fp);
    }
}

void test_bicubic(int i, double* values, FILE* fp) {
    // Initializes time recording variables and bicubic_interp
    clock_t start, end;
    double cpu_time_used = 0.0;
    int n_value = sqrt(n_values[i]);
    bicubic_interp *interp = bicubic_interp_init(values, n_value, n_value, -1, -1, 1, 1);

    // Iterates through the interpolation with varying loop operation counts
    for (int m = 0; m < iteration_values_size; m++) {
        // Precomputes random values
        double* randomu = (double*)malloc(iteration_values[m] * sizeof(double));
        double* randomv = (double*)malloc(iteration_values[m] * sizeof(double));
        for (int k = 0; k < iteration_values[m]; k++) {
            randomu[k] = rand() * 100;
            randomv[k] = rand() * 100;
        }
        
        // Performs benchmark
        int iter = sqrt(iteration_values[m]);
        for (int l = 0; l < trials; l++) {
            start = clock();
            for (int s = 0; s <= iter; s += 1) {
                for (int t = 0; t <= iter; t += 1) {
                    volatile const double result = bicubic_interp_eval(interp, randomu[s * iter + t], randomv[s * iter + t]);
                }
            }
            end = clock();
            cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
        }
        cpu_time_used /= trials;
        printf("Time for size %d and iterations %d is %lf\n", n_values[i], iteration_values[m], cpu_time_used);
        fprintf(fp, "%d,%d,%lf\n", n_values[i], iteration_values[m], cpu_time_used);

        free(randomu);
        free(randomv);
    }
    printf("\n");
    bicubic_interp_free(interp);
}

void test_all_bicubic(double** values, FILE* fp) {
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
