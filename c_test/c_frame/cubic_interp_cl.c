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
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cubic_interp_cl.h"

// Kernel for cubic interpolation
const char *kernelSource = 
"__kernel void cubic_interp_eval(int c, int length, double f, double t0, __global double* dev_t, __global double* dev_a) {\n"
"    int idx = get_global_id(0);\n"
"    int stride = get_global_size(0);\n"
"    double xmin = 0.0, xmax = length - 1.0;\n"
"    while (idx < c) {\n"
"        double x = dev_t[idx];\n"
"        x = f * x + t0;\n"
"        x = fmin(fmax(x, xmin), xmax);\n"
"        int ix = (int)floor(x);\n"
"        x -= ix;\n"
"        __global double *a = dev_a + ix * 4;\n"
"        dev_t[idx] = (x * (x * (x * a[0] + a[1]) + a[2]) + a[3]);\n"
"        idx += stride;\n"
"    }\n"
"}\n";

void test_all_cubic_cl(double **values, FILE *fp) 
{
    printf("Testing OpenCL cubic\n");
    fprintf(fp, "Data,Iterations,Time\n");

    // Initializes OpenCL 
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    // Creates and builds program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "cubic_interp_eval", &err);

    // Iterates through test sizes
    for (int i = 0; i < n_values_size; i++) {
        cubic_interp *interp = cubic_interp_init(values[i], n_values[i], -1, 1);

        // Creates device buffer for interp->a
        cl_mem dev_a = clCreateBuffer(context, CL_MEM_READ_ONLY, interp->length * sizeof(double[4]), NULL, &err);
        clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, interp->length * sizeof(double[4]), interp->a, 0, NULL, NULL);

        // Iterates through evaluation counts
        int c = 10000;
        for (int m = 1; m < 5; m++) {
            // Precomputes random values
            double* t = (double*)malloc(c * sizeof(double));
            for (int k = 0; k < c; k++) {
                t[k] = rand() * 100;
            }

            // Gets device properties
            size_t maxWorkGroupSize;
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
            size_t threadsPerBlock = maxWorkGroupSize < 128 ? maxWorkGroupSize : 128;
            size_t blocksPerGrid = (c + threadsPerBlock - 1) / threadsPerBlock;
            size_t globalSize = blocksPerGrid * threadsPerBlock;

            // Creates device buffer for t
            cl_mem dev_t = clCreateBuffer(context, CL_MEM_READ_WRITE, c * sizeof(double), NULL, &err);
            clEnqueueWriteBuffer(queue, dev_t, CL_TRUE, 0, c * sizeof(double), t, 0, NULL, NULL);

            // Sets kernel arguments
            clSetKernelArg(kernel, 0, sizeof(int), &c);
            clSetKernelArg(kernel, 1, sizeof(int), &interp->length);
            clSetKernelArg(kernel, 2, sizeof(double), &interp->f);
            clSetKernelArg(kernel, 3, sizeof(double), &interp->t0);
            clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_t);
            clSetKernelArg(kernel, 5, sizeof(cl_mem), &dev_a);

            // Creates event for timing
            cl_event event;
            cl_ulong start, end;

            // Executes kernel and times it
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &threadsPerBlock, 0, NULL, &event);
            clWaitForEvents(1, &event);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            double elapsedTime = (end - start) / 1e6; // time is returned in nanoseconds => milliseconds
            clEnqueueReadBuffer(queue, dev_t, CL_TRUE, 0, c * sizeof(double), t, 0, NULL, NULL);
            printf("Time for size %d and iterations %d is %lf\n", n_values[i], c, elapsedTime / 1000.0);
            fprintf(fp, "%d,%d,%lf\n", n_values[i], c, elapsedTime / 1000.0);

            // Frees dev_t/t and events
            clReleaseMemObject(dev_t);
            clReleaseEvent(event);
            free(t);

            c *= 10;
        }

        // Frees interp-related resources
        clReleaseMemObject(dev_a);
        cubic_interp_free(interp);
        printf("\n");
    }

    // Cleans OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}