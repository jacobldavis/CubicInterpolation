'''
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
 *
'''
import numpy as np
import cupy as cp
import jax.numpy as jnp
from jax import jit
from jax import vmap
import jax
import torch
import pyopencl as cl
import time
from cubic_interp import *
n_values = [4, 9, 100, 400, 1024]
iteration_counts = [10000, 100000, 1000000, 10000000]
# Modify trials to change the number of runs per test; averaged result is returned
trials = 10

# Collects the 1d randomly generated values
onevalues = []
with open('../1dvalues.txt', 'r') as f:
    for line in f:
        str_values = line.strip().split(', ')
        float_values = np.array([float(val) for val in str_values])
        onevalues.append(float_values)

def test_all_cubic_np():
    f = open('../results/numpy_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing np cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = np.array(interp.a)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            elapsed_time = 0
            random = np.random.uniform(0, 100, iterations)
            for trial in range(trials):
                start = time.perf_counter()
                result = interp.cubic_interp_eval_np(random)
                end = time.perf_counter()
                elapsed_time += end - start
            elapsed_time /= trials
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_torch():
    f = open('../results/torch_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing torch cubic:")
    # Iterates through the test for each size of data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = torch.tensor(interp.a).to(device)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            elapsed_time = 0
            random = np.random.uniform(0, 100, iterations)
            random = torch.from_numpy(random).to(device)
            for trial in range(trials):
                start = time.perf_counter()
                result = interp.cubic_interp_eval_torch(device, random)
                torch.cuda.synchronize()
                end = time.perf_counter()
                elapsed_time += end - start
            elapsed_time /= trials
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_torch_cpu():
    f = open('../results/cpu_torch_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing torch cpu cubic:")
    # Iterates through the test for each size of data
    device = torch.device("cpu")
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = torch.tensor(interp.a).to(device)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            elapsed_time = 0
            random = np.random.uniform(0, 100, iterations)
            random = torch.from_numpy(random).to(device)
            for trial in range(trials):
                start = time.perf_counter()
                result = interp.cubic_interp_eval_torch(device, random)
                end = time.perf_counter()
                elapsed_time += end - start
            elapsed_time /= trials
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_cupy():
    f = open('../results/cupy_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing cupy cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = cp.array(interp.a)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            elapsed_time = 0
            random = np.random.uniform(0, 100, iterations)
            random = cp.asarray(random)
            for trial in range(trials):
                start = time.perf_counter()
                result = interp.cubic_interp_eval_cp(random)
                cp.cuda.Device().synchronize()
                end = time.perf_counter()
                elapsed_time += end - start
            elapsed_time /= trials
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_jax():
    f = open('../results/jax_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing jax cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = jnp.array(interp.a)
        interp.batch_eval = jit(vmap(interp.cubic_interp_eval_jax))
        # Iterates through the test for each iteration count
        with jax.default_device(jax.devices('gpu')[0]):
            for iterations in iteration_counts:
                elapsed_time = 0
                random = jnp.array(np.random.uniform(0, 100, iterations))
                # for the precompilation step, the array passed in must be the 
                # same shape as ones passed in later on for the performance bump
                # for the purposes of the benchmark, this is fine
                _ = interp.batch_eval(jnp.zeros_like(random))
                _.block_until_ready()
                for trial in range(trials):
                    start = time.perf_counter()
                    result = interp.batch_eval(random)
                    result.block_until_ready()
                    end = time.perf_counter()
                    elapsed_time += end - start
                elapsed_time /= trials
                print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
                f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_jax_cpu():
    f = open('../results/cpu_jax_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing jax cpu cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = jnp.array(interp.a)
        interp.batch_eval = jit(vmap(interp.cubic_interp_eval_jax))
        # Iterates through the test for each iteration count
        with jax.default_device(jax.devices('cpu')[0]):
            for iterations in iteration_counts:
                elapsed_time = 0
                random = jnp.array(np.random.uniform(0, 100, iterations))
                # for the precompilation step, the array passed in must be the 
                # same shape as ones passed in later on for the performance bump
                # for the purposes of the benchmark, this is fine
                _ = interp.batch_eval(jnp.zeros_like(random))
                _.block_until_ready()
                for trial in range(trials):
                    start = time.perf_counter()
                    result = interp.batch_eval(random)
                    result.block_until_ready()
                    end = time.perf_counter()
                    elapsed_time += end - start
                elapsed_time /= trials
                print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
                f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

kernelsource = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void cubic_interp_eval(const uint c, const uint length, const double f, const double t0, __global double* dev_t, __global double* dev_a) {
    int idx = get_global_id(0);
    int stride = get_global_size(0);
    double xmin = 0.0, xmax = length - 1.0;
    while (idx < c) {
        double x = dev_t[idx];
        x = f * x + t0;
        x = fmin(fmax(x, xmin), xmax);
        int ix = floor(x);
        x -= ix;
        __global double *a = dev_a + ix * 4;
        dev_t[idx] = (x * (x * (x * a[0] + a[1]) + a[2]) + a[3]);
        idx += stride;
    }
};
"""

def test_all_cubic_opencl():
    f = open('../results/pyopencl_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing pyopencl cubic:")

    # Initializes OpenCL and builds program
    context = cl.create_some_context(0)
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(context, kernelsource).build()
    cubic = program.cubic_interp_eval
    cubic.set_scalar_arg_dtypes([np.uint32, np.uint32, np.float64, np.float64, None, None])

    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = np.array(interp.a).astype(np.float64)
        d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=interp.a)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            elapsed_ns = 0
            random = np.random.uniform(0, 100, iterations).astype(np.float64)
            d_random = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=random)
            for trial in range(trials):
                event = cubic(queue, (iterations,), None,
                    np.uint32(iterations), np.uint32(interp.length),
                    np.float64(interp.f), np.float64(interp.t0),
                    d_random, d_a)
                event.wait()
                start_time = event.profile.start  # In nanoseconds
                end_time = event.profile.end
                elapsed_ns += end_time - start_time
            elapsed_s = (elapsed_ns * 1e-9) / trials
            cl.enqueue_copy(queue, random, d_random)
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_s:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_s:.4g}\n")
        print()
    f.close()

test_all_cubic_np()
test_all_cubic_torch_cpu()
test_all_cubic_jax_cpu()

if torch.cuda.is_available():
    test_all_cubic_torch()
if cp.cuda.is_available():
    test_all_cubic_cupy()
if jax.devices()[0].platform != "cpu":
    test_all_cubic_jax()

platforms = cl.get_platforms()
devices = [device for platform in platforms for device in platform.get_devices()]
gpu_available = any(device.type == cl.device_type.GPU for device in devices)
if gpu_available:
    test_all_cubic_opencl()