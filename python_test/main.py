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
import time
from cubic_interp import *
n_values = [4, 9, 100, 400, 1024]
iteration_counts = [10000, 100000, 1000000, 10000000]

# Collects the 1d randomly generated values
onevalues = []
with open('../1dvalues.txt', 'r') as f:
    for line in f:
        str_values = line.strip().split(', ')
        float_values = np.array([float(val) for val in str_values])
        onevalues.append(float_values)

def test_all_cubic_np():
    f = open('numpy_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing np cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = np.array(interp.a)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            random = np.random.uniform(0, 100, iterations)
            start = time.perf_counter()
            result = interp.cubic_interp_eval_np(random)
            end = time.perf_counter()
            elapsed_time = end - start
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_torch():
    f = open('torch_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing torch cubic:")
    # Iterates through the test for each size of data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = torch.tensor(interp.a).to(device)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            random = np.random.uniform(0, 100, iterations)
            random = torch.from_numpy(random).to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = interp.cubic_interp_eval_torch(device, random)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_torch_cpu():
    f = open('cpu_torch_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing torch cubic:")
    # Iterates through the test for each size of data
    device = torch.device("cpu")
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = torch.tensor(interp.a).to(device)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            random = np.random.uniform(0, 100, iterations)
            random = torch.from_numpy(random).to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = interp.cubic_interp_eval_torch(device, random)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_cupy():
    f = open('cupy_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing cupy cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = cp.array(interp.a)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            random = np.random.uniform(0, 100, iterations)
            random = cp.asarray(random)
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
            result = interp.cubic_interp_eval_cp(random)
            end_gpu.record()
            end_gpu.synchronize()
            elapsed_time = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000.0
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_jax():
    f = open('jax_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing jax cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = jnp.array(interp.a)
        # Iterates through the test for each iteration count
        for iterations in iteration_counts:
            random = np.random.uniform(0, 100, iterations)
            random = jnp.array(random)
            interp.batch_eval = jit(vmap(interp.cubic_interp_eval_jax))
            _ = interp.batch_eval(random)
            jax.block_until_ready(_)
            start = time.perf_counter()
            result = interp.batch_eval(random)
            result.block_until_ready()
            end = time.perf_counter()
            elapsed_time = end - start
            print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
            f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

def test_all_cubic_jax_cpu():
    f = open('cpu_jax_data.csv', 'w')
    f.write("Data,Iterations,Time\n")
    print("Testing jax cpu cubic:")
    # Iterates through the test for each size of data
    for i, n_value in enumerate(n_values):
        interp = cubic_interp(onevalues[i], n_value, -1, 1)
        interp.a = jnp.array(interp.a)
        # Iterates through the test for each iteration count
        with jax.default_device(jax.devices('cpu')[0]):
            for iterations in iteration_counts:
                random = np.random.uniform(0, 100, iterations)
                random = jnp.array(random)
                interp.batch_eval = jit(vmap(interp.cubic_interp_eval_jax))
                _ = interp.batch_eval(random)
                jax.block_until_ready(_)
                start = time.perf_counter()
                result = interp.batch_eval(random)
                result.block_until_ready()
                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Time for size {n_value} and iterations {iterations} is {elapsed_time:.4g}")
                f.write(f"{n_value},{iterations},{elapsed_time:.4g}\n")
        print()
    f.close()

test_all_cubic_np()
test_all_cubic_torch()
test_all_cubic_torch_cpu()
test_all_cubic_cupy()
test_all_cubic_jax()
test_all_cubic_jax_cpu()