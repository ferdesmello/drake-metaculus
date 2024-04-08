import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Assuming original CDF data is appropriately scaled
x_values_log = np.logspace(np.log10(0.01), np.log10(1000), 100)  # Logarithmic scale for x
cdf_values = np.linspace(0, 1, 100)  # CDF values from 0 to 1

# Interpolate to create an inverse CDF function in log space
inverse_cdf_function = interp1d(cdf_values, np.log(x_values_log), kind='linear')

# Generate uniform samples representing cumulative probabilities
uniform_samples = np.random.rand(1000)

# Apply the inverse CDF (in log space) to these samples to get log-scaled x values
sampled_x_values_log = inverse_cdf_function(uniform_samples)

# Exponentiate to return to the original scale
sampled_x_values = np.exp(sampled_x_values_log)

# Visualize the sampled x values on a logarithmic scale
plt.hist(sampled_x_values, bins=np.logspace(np.log10(0.01), np.log10(1000), 50), alpha=0.7, density=True)
plt.xscale('log')  # Set the x-axis to a log scale
plt.xlabel('X Values')
plt.ylabel('Frequency')
plt.title('Histogram of Sampled X Values')
plt.show()
