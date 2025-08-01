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
#include <cuda_runtime.h>
#include "cubic_interp_cuda.h"

__global__ void cubic_interp_eval(int c, cubic_interp* dev_interp, double* dev_t) {
    // Sets initial index and other values to compute
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double f = dev_interp->f;
    double t0 = dev_interp->t0;
    double xmin = 0.0, xmax = dev_interp->length - 1.0;

    // Performs cubic interpolation
    while (idx < c) {
        double x = dev_t[idx];
        x = f * x + t0;
        x = fmin(fmax(x, xmin), xmax);

        int ix = (int)floor(x);
        x -= ix;

        const double* a = (const double*)&(dev_interp->a[ix][0]);
        dev_t[idx] = (x * (x * (x * a[0] + a[1]) + a[2]) + a[3]);

        idx += stride;
    }
}

extern void test_all_cubic_cuda(double **values, FILE *fp) 
{
    printf("Testing CUDA cubic\n");
    fprintf(fp, "Data,Iterations,Time\n");

    // Check if CUDA is available
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA initialization error or no CUDA GPU found");
        return;
    }

    // Iterates through the test for each size in n_values
    for (int i = 0; i < n_values_size; i++) {
        // Initializes cubic_interp
        cubic_interp *interp = cubic_interp_init(values[i], n_values[i], -1, 1);

        // Creates device copy of interp->a
        double (*dev_a)[4];
        cudaMalloc((void**)&dev_a, interp->length * sizeof(double[4]));
        cudaMemcpy(dev_a, interp->a, interp->length * sizeof(double[4]), cudaMemcpyHostToDevice);

        // Creates device copy of interp
        cubic_interp h_interp_dev;
        h_interp_dev.length = interp->length;
        h_interp_dev.f = interp->f;
        h_interp_dev.t0 = interp->t0;
        h_interp_dev.a = dev_a;
        cubic_interp *dev_interp;
        cudaMalloc((void**)&dev_interp, sizeof(cubic_interp));
        cudaMemcpy(dev_interp, &h_interp_dev, sizeof(cubic_interp), cudaMemcpyHostToDevice);

        // Iterates through the interpolation with varying evaluation counts
        for (int m = 0; m < iteration_values_size; m++) {
            // Precomputes random values and other relevant vars for CUDA
            double* t = (double*)malloc(iteration_values[m] * sizeof(double));
            for (int k = 0; k < iteration_values[m]; k++) {
                t[k] = rand() * 100;
            }
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            int threadsPerBlock = min(prop.maxThreadsPerBlock, 128);
            int blocksPerGrid = int((iteration_values[m]+threadsPerBlock-1)/threadsPerBlock);
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            float elapsedTime;
            double averageTime = 0.0;

            // Copies t to the device
            double* dev_t;
            cudaMalloc( (void**)&dev_t, iteration_values[m] * sizeof(double));
            cudaMemcpy(dev_t, t, iteration_values[m] * sizeof(double), cudaMemcpyHostToDevice);

            // Performs benchmark and records time
            for (int l = 0; l < trials; l++) {
                cudaEventRecord(start, 0);
                cubic_interp_eval<<<blocksPerGrid,threadsPerBlock>>>(iteration_values[m], dev_interp, dev_t);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                averageTime += elapsedTime;
            }
            averageTime /= trials;
            cudaMemcpy(t, dev_t, iteration_values[m]*sizeof(double), cudaMemcpyDeviceToHost); // include or exclude from timing?
            printf("Time for size %d and iterations %d is %lf\n", n_values[i], iteration_values[m], averageTime / 1000.0);
            fprintf(fp, "%d,%d,%lf\n", n_values[i], iteration_values[m], averageTime / 1000.0);

            // Frees t, dev_t, and events
            cudaFree(dev_t);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            free(t);
        }
        // Frees interp related variables
        cudaFree(dev_interp);
        cudaFree(dev_a);
        cubic_interp_free(interp);
        printf("\n");
    }
}