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
 * The purpose of this program is to generate a series of random values
 * for testing cubic interpolation at different array sizes.
 * 
'''

import numpy as np
n_values = [4, 9, 100, 400, 1024]

with open('1dvalues.txt', 'w') as f:
    for n in n_values:
        # generates a randomized array of n floats
        initial = np.random.randint(0, 100, n)
        offset = np.random.rand(n)
        values = initial * offset

        # prints the array
        values = np.round(values, 2)
        print(", ".join(map(str, values)), file=f)


