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
#include "kernel.h"

__global__ void cubic_interp_eval(int c, cubic_interp* dev_interp, double* dev_t) {
    // Sets initial index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Performs cubic interpolation
    while (idx < c) {
        double x = dev_t[idx], xmin = 0.0, xmax = dev_interp->length - 1.0;
        x *= dev_interp->f;
        x += dev_interp->t0;
        x = min(max(x, xmin), xmax);

        double ix = floor(x);
        x -= ix;

        const double *a = dev_interp->a[(int) ix];
        dev_t[idx] = (x * (x * (x * a[0] + a[1]) + a[2]) + a[3]);
        idx += blockDim.x * gridDim.x;
    }
}

extern void test_all_cubic_cuda(double **values, FILE *fp) 
{
    printf("Testing CUDA cubic\n");
    fprintf(fp, "Data,Iterations,Time\n");

    // Iterates through the test for each size in n_values
    for (int i = 0; i < n_values_size; i++) {
        // Initializes cubic_interp and copies to the GPU
        cubic_interp *interp = cubic_interp_init(values[i], n_values[i], -1, 1);
        cubic_interp *dev_interp;
        cudaMalloc((void**)&dev_interp, sizeof(cubic_interp));
        cudaMemcpy(dev_interp, interp, sizeof(cubic_interp), cudaMemcpyHostToDevice);

        // Iterates through the interpolation with varying evaluation counts
        int c = 10000;
        for (int m = 1; m < 5; m++) {
            // Precomputes random values and other relevant vars for CUDA
            double* t = (double*)malloc(c * sizeof(double));
            for (int k = 0; k < c; k++) {
                t[k] = rand() * 100;
            }
            int threadsPerBlock = 256;
            int blocksPerGrid = int((c+threadsPerBlock-1)/threadsPerBlock);
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            float elapsedTime;

            // Copies t to the GPU
            double* dev_t;
            cudaMalloc( (void**)&dev_t, c * sizeof(double));
            cudaMemcpy(dev_t, t, c * sizeof(double), cudaMemcpyHostToDevice);

            // Performs benchmark and records time
            cudaEventRecord(start, 0);
            cubic_interp_eval<<<blocksPerGrid,threadsPerBlock>>>(c, dev_interp, dev_t);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            cudaMemcpy(t, dev_t, c*sizeof(double), cudaMemcpyDeviceToHost); // include or exclude from timing?
            printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, elapsedTime);
            fprintf(fp, "%d,%d,%lf\n", n_values[i], c, elapsedTime);

            // Frees t, dev_t, and events
            cudaFree(dev_t);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            free(t);

            c *= 10;
        }
        // Frees dev_interp and interp
        cudaFree(dev_interp);
        cubic_interp_free(interp);
        printf("\n");
    }
}