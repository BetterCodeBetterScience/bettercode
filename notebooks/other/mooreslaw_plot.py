# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bettercode
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Moore's Law Visualization
#
# This notebook loads processor transistor count data and visualizes the exponential growth over time (Moore's Law).

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

FIGURE_DIR = Path("../../book/book/images")
assert FIGURE_DIR.exists(), f"Figure directory {FIGURE_DIR} does not exist"

# %% [markdown]
# ## Load the Data
#
# Load the processor transistor count data from the CSV file.

# %%
# Load the data
data_path = '../src/bettercode/data/mooreslaw/processor_transistor_counts.csv'
df = pd.read_csv(data_path)

# Display first few rows
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ## Data Preprocessing
#
# Clean the transistor count column by removing commas and converting to numeric values.

# %%
# Clean the transistor count column
# Remove commas and extract the first number if there are multiple values
df['Transistor count'] = df['Transistor count'].astype(str).str.replace(',', '')
df['Transistor count'] = df['Transistor count'].str.extract(r'(\d+)')[0].astype(float)

# Remove rows with missing transistor counts or years
df_clean = df.dropna(subset=['Transistor count', 'Year'])

print(f"Clean dataset shape: {df_clean.shape}")
df_clean.head()

# %% [markdown]
# ## Plot Log Transistor Count vs Year
#
# Create a scatter plot showing the logarithmic relationship between transistor count and year, demonstrating Moore's Law.

# %%
# Create the plot
plt.figure(figsize=(12, 6))
plt.scatter(df_clean['Year'], np.log10(df_clean['Transistor count']), alpha=0.6, s=50)

# Add labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Log₁₀(Transistor Count)', fontsize=12)
plt.title('Moore\'s Law: Transistor Count Growth Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add a trend line
z = np.polyfit(df_clean['Year'], np.log10(df_clean['Transistor count']), 1)
p = np.poly1d(z)
plt.plot(df_clean['Year'], p(df_clean['Year']), 'r--', alpha=0.8, linewidth=2, label=f'Trend line (slope={z[0]:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'moores_law_plot.png', dpi=300)
plt.show()

# %% [markdown]
# ## Calculate Doubling Time
#
# Moore's Law predicts that transistor counts double approximately every 2 years. Let's verify this from our data.

# %%
# Calculate doubling time from the slope
# The slope is in log10 units per year
# Doubling means log10(2) = 0.301 increase
doubling_time = 0.301 / z[0]

print(f"Slope of log₁₀(transistor count) vs year: {z[0]:.4f}")
print(f"Estimated doubling time: {doubling_time:.2f} years")
print(f"\nMoore's Law prediction: ~2 years")
print(f"Observed from data: ~{doubling_time:.2f} years")

# %% [markdown]
# ### Top500 performance

# %%
top500_file = '../src/bettercode/data/mooreslaw/top500_performance.csv'
top500_df = pd.read_csv(top500_file)
drop_cols = ['Power efficiency (GFLOPS per Watt)', 'Unnamed: 5',
       'Unnamed: 6']
top500_df = top500_df.drop(columns=drop_cols)

top500_df.head()

# %%
top500_df.columns

# %%
# Check unique units in the Peak speed column
units = top500_df['Peak speed (Rmax)'].str.extract(r'(GFLOPS|TFLOPS|PFLOPS|EFLOPS)')[0].unique()
print("Units found:", units)
print("\nSample values:")
print(top500_df['Peak speed (Rmax)'].head(10))


# %%
# Convert Peak speed to GFLOPS
def convert_to_gflops(speed_str):
    """Convert speed string to GFLOPS"""
    # Extract numeric value and unit
    parts = speed_str.strip().split()
    value = float(parts[0])
    unit = parts[1]
    
    # Conversion factors to GFLOPS
    conversions = {
        'GFLOPS': 1,
        'TFLOPS': 1e3,      # 1 TFLOPS = 1000 GFLOPS
        'PFLOPS': 1e6,      # 1 PFLOPS = 1,000,000 GFLOPS
        'EFLOPS': 1e9       # 1 EFLOPS = 1,000,000,000 GFLOPS
    }
    
    return value * conversions[unit]

# Create new column with values in GFLOPS
top500_df['Peak speed (GFLOPS)'] = top500_df['Peak speed (Rmax)'].apply(convert_to_gflops)

# Display the results
print("Conversion complete!")
print("\nSample conversions:")
comparison = top500_df[['Year', 'Supercomputer', 'Peak speed (Rmax)', 'Peak speed (GFLOPS)']].head(10)
print(comparison.to_string())

# %%
# Create plot of log(GFLOPS) vs Year
plt.figure(figsize=(12, 6))
plt.scatter(top500_df['Year'], np.log10(top500_df['Peak speed (GFLOPS)']), alpha=0.6, s=50)

# Add labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Log₁₀(GFLOPS)', fontsize=12)
plt.title('Top500 Supercomputer Performance Growth Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add a trend line
z_top500 = np.polyfit(top500_df['Year'], np.log10(top500_df['Peak speed (GFLOPS)']), 1)
p_top500 = np.poly1d(z_top500)
plt.plot(top500_df['Year'], p_top500(top500_df['Year']), 'r--', alpha=0.8, linewidth=2, 
         label=f'Trend line (slope={z_top500[0]:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'top500_performance_plot.png', dpi=300)
plt.show()

# Calculate doubling time for supercomputer performance
doubling_time_top500 = 0.301 / z_top500[0]
print(f"\nSlope of log₁₀(GFLOPS) vs year: {z_top500[0]:.4f}")
print(f"Estimated doubling time: {doubling_time_top500:.2f} years")

# %%
# Create two-panel plot comparing Moore's Law and Top500 performance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: Transistor counts (Moore's Law)
ax1.scatter(df_clean['Year'], np.log10(df_clean['Transistor count']), alpha=0.6, s=50)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Log₁₀(Transistor Count)', fontsize=12)
ax1.set_title('Moore\'s Law: Transistor Count Growth', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add trend line for transistors
z = np.polyfit(df_clean['Year'], np.log10(df_clean['Transistor count']), 1)
p = np.poly1d(z)
ax1.plot(df_clean['Year'], p(df_clean['Year']), 'r--', alpha=0.8, linewidth=2, 
         label=f'Trend line (slope={z[0]:.3f})')
ax1.legend()

# Right panel: Top500 supercomputer performance
ax2.scatter(top500_df['Year'], np.log10(top500_df['Peak speed (GFLOPS)']), alpha=0.6, s=50)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Log₁₀(GFLOPS)', fontsize=12)
ax2.set_title('Top500 Supercomputer Performance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add trend line for Top500
z_top500 = np.polyfit(top500_df['Year'], np.log10(top500_df['Peak speed (GFLOPS)']), 1)
p_top500 = np.poly1d(z_top500)
ax2.plot(top500_df['Year'], p_top500(top500_df['Year']), 'r--', alpha=0.8, linewidth=2, 
         label=f'Trend line (slope={z_top500[0]:.3f})')
ax2.legend()

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'moores_law_comparison.png', dpi=300)
plt.show()

# Print comparison
print("Comparison of Growth Rates:")
print(f"\nTransistor Count (Moore's Law):")
print(f"  Slope: {z[0]:.4f}")
print(f"  Doubling time: {0.301 / z[0]:.2f} years")
print(f"\nTop500 Performance:")
print(f"  Slope: {z_top500[0]:.4f}")
print(f"  Doubling time: {0.301 / z_top500[0]:.2f} years")

# %%
# Calculate how long a 1-hour simulation on the latest supercomputer would take in 1994

# Get the most recent supercomputer performance
latest_year = top500_df['Year'].max()
latest_performance = top500_df[top500_df['Year'] == latest_year]['Peak speed (GFLOPS)'].max()

# Get the 1994 supercomputer performance
perf_1994 = top500_df[top500_df['Year'] == 1994]['Peak speed (GFLOPS)'].max()

# Calculate the ratio
performance_ratio = latest_performance / perf_1994

# Calculate time for 1994 (assuming linear relationship with performance)
simtime = 1/60 # one minute on latest
time_1994_hours = simtime * performance_ratio
time_1994_days = time_1994_hours / 24
time_1994_years = time_1994_days / 365.25

print(f"Most recent supercomputer ({latest_year}):")
print(f"  Performance: {latest_performance:,.0f} GFLOPS")
print(f"\n1994 supercomputer:")
print(f"  Performance: {perf_1994:,.0f} GFLOPS")
print(f"\nPerformance ratio: {performance_ratio:,.0f}x")
print(f"\nIf a simulation takes {simtime*60} minutes on the {latest_year} supercomputer,")
print(f"it would have taken on the 1994 supercomputer:")
print(f"  {time_1994_hours:,.0f} hours")
print(f"  {time_1994_days:,.1f} days")
print(f"  {time_1994_years:,.2f} years")

# %%
