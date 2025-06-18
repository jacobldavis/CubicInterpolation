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
from pathlib import Path

# Collects the cubic interpolation files
csv_files = glob.glob('results/*.csv')
csv_files.remove('results/bi_c_data.csv')
csv_files.remove('results/cpu_torch_data.csv')

# Plots the data
labels = []
times = []

labels = []
times = []

for file in csv_files:
    df = pd.read_csv(file)
    df = df[(df['Data'] == 400) & (df['Iterations'] == 10000000)]
    
    if not df.empty:
        label = Path(file).stem.replace('_data', '')
        labels.append(label)
        times.append(df['Time'].iloc[0])

# Sorts by time
sorted_pairs = sorted(zip(times, labels)) 
times, labels = zip(*sorted_pairs)

# Normalizes time values for colormap
cmap = plt.get_cmap('tab20')
colors = [cmap(i % 20) for i in range(len(labels))]

# Plots
plt.figure(figsize=(12, 6))
plt.bar(labels, times, color=colors)
plt.yscale('log')
plt.xlabel('Framework')
plt.ylabel('Time')
plt.title('Execution Time by Framework (|Data| = 400, Iterations = 10⁷)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
