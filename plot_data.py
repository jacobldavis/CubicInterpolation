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
 * The purpose of this program is to plot the csv data generated from tests
 * 
'''
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Collects the cubic interpolation files
csv_files = glob.glob('results/*.csv')
csv_files.remove('results/bi_c_data.csv')

# Plots the data
plt.figure(figsize=(100,100))
for file in csv_files:
    df = pd.read_csv(file)
    df = df[df['Data'] == 400]
    df = df[df['Iterations'] != 10000]
    plt.plot(df['Iterations'], df['Time'], label=file.split('/')[-1])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Time')
plt.title('Loop Iterations vs. Time (|Data| == 400)')
plt.legend()
plt.grid(True)
plt.show()
