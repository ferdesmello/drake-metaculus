import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Sample Data
data = np.random.lognormal(mean=1.0, sigma=1.5, size=1000)

# Step 2: Define Logarithmic Bins
bins = np.logspace(np.log10(min(data)), np.log10(max(data)), 20)

# Step 3: Calculate Bin Widths
bin_widths = np.diff(bins)

# Step 4: Assign Weights to Data Points
# Find indices of the bins to which each value belongs.
indices = np.digitize(data, bins) - 1

# Create an array of weights for each data point, inversely proportional to the bin width.
weights = np.zeros_like(data)
for i in range(len(bins)-1):
    # Assign weights inversely proportional to bin width
    weights[indices == i] = 1. / bin_widths[i]

# Normalize weights so the integral of the histogram equals 1
weights /= (weights.sum() * np.log(10))  # Multiplying by np.log(10) because of log scale

# Step 5: Plot the Histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=bins, weights=weights, alpha=0.6, color='b', edgecolor='black')
plt.xscale('log')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram with Logarithmic Bins')
plt.show()