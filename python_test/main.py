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
n_values = [4, 10, 100, 400, 1000]

# Collects the 1d randomly generated values
onevalues = []
with open('../1dvalues.txt', 'r') as f:
    for line in f:
        str_values = line.strip().split(', ')
        float_values = np.array([float(val) for val in str_values])
        onevalues.append(float_values)

# Collects the 2d randomly generated values
twovalues = []
with open('../2dvalues.txt', 'r') as f:
    for line in f:
        str_values = line.strip().split(', ')
        float_values = np.array([float(val) for val in str_values])
        onevalues.append(float_values)
