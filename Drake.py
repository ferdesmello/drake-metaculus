import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import gaussian_kde
from scipy.stats import loguniform
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


# ------------------------
"""# What is the average rate of formation of suitable stars (stars/year) in our galaxy?
(mean = 0.5, sigma = 0.5, size = 100)
# What fraction of stars form planets?
(mean = 0.5, sigma = 0.2, size = 100)
# What is the average number of habitable planets per star?
(mean = 0.05, sigma = 0.05, size = 100)
# On what fraction of habitable planets does any form of life emerge?
(mean = 5*10**-4, sigma = 5*10**-4, size = 100)
# On what fraction of habitable planets with life does intelligence evolve?
(mean = 2*10**-4, sigma = 5*10**-4, size = 100)
# What fraction of planets with intelligent life are capable of interstellar communication?
(mean = 0.1, sigma = 0.5, size = 100)
# For how many years does a civilization remain detectable?
(mean = 500, sigma = 250, size = 100)"""



# What is the average rate of formation of suitable stars (stars/year) in our galaxy?
xf1 = np.linspace(0.1, 5, 1000)
f1 = pd.Series(loguniform(0.1, 5).pdf(xf1))
# What fraction of stars form planets?
xf2 = np.linspace(0.1, 1, 1000)
f2 = pd.Series(loguniform(0.1, 1).pdf(xf2))
# What is the average number of habitable planets per star?
xf3 = np.linspace(0.001, 0.01, 1000)
f3 = pd.Series(loguniform(0.001, 0.01).pdf(xf3))
# On what fraction of habitable planets does any form of life emerge?
xf4 = np.linspace(0.0001, 0.001, 1000)
f4 = pd.Series(loguniform(0.0001, 0.001).pdf(xf4))
# On what fraction of habitable planets with life does intelligence evolve?
xf5 = np.linspace(0.0001, 0.001, 1000)
f5 = pd.Series(loguniform(0.0001, 0.001).pdf(xf5))
# What fraction of planets with intelligent life are capable of interstellar communication?
xf6 = np.linspace(0.01, 1, 1000)
f6 = pd.Series(loguniform(0.01, 1).pdf(xf6))
# For how many years does a civilization remain detectable?
xf7 = np.linspace(100, 1000, 1000)
f7 = pd.Series(loguniform(100, 1000).pdf(xf7))

print("Sampling")
# What is the average rate of formation of suitable stars (stars/year) in our galaxy?
f1_sampled = f1.sample(n = 1000, replace = True).reset_index(drop=True)
# What fraction of stars form planets?
f2_sampled = f2.sample(n = 1000, replace = True).reset_index(drop=True)
# What is the average number of habitable planets per star?
f3_sampled = f3.sample(n = 1000, replace = True).reset_index(drop=True)
# On what fraction of habitable planets does any form of life emerge?
f4_sampled = f4.sample(n = 1000, replace = True).reset_index(drop=True)
# On what fraction of habitable planets with life does intelligence evolve?
f5_sampled = f5.sample(n = 1000, replace = True).reset_index(drop=True)
# What fraction of planets with intelligent life are capable of interstellar communication?
f6_sampled = f6.sample(n = 1000, replace = True).reset_index(drop=True)
# For how many years does a civilization remain detectable?
f7_sampled = f7.sample(n = 1000, replace = True).reset_index(drop=True)

"""print(f1_sampled)
print(f2_sampled)
print(f3_sampled)
print(f4_sampled)
print(f5_sampled)
print(f6_sampled)
print(f7_sampled)"""

print("Multiplying.")

f_samples = (f1_sampled * 
             f2_sampled * 
             f3_sampled * 
             f4_sampled * 
             f5_sampled * 
             f6_sampled * 
             f7_sampled)

"""print(f_samples)
print(len(f_samples))
print(f_samples.hasnans)

f_samples = f_samples.dropna()"""

print("Monte Carling.")

# Estimate the PDF of x using KDE
f_kde = gaussian_kde(f_samples)
x_values = np.linspace(min(f_samples), max(f_samples), 10000)
f_pdf = f_kde(x_values)

# Normalize the PDF values so that the sum equals 1
total_area = simps(f_pdf, x_values)
f_pdf = f_pdf / total_area

print("Together done!")





#-------------------------------------------

print("Making Figures.")

# Figures

fig1, ax1 = plt.subplots(figsize = (10, 6))

ax1.plot(x_values, f_pdf,
         color = "red",
         linewidth = 2,
         label = "pdf")

"""ax1.hist(y1, 
         #range = (45, 200), 
         histtype = "step", 
         fill = True, 
         density = True, 
         #alpha = 0.2, 
         #bins = 50, 
         color = (1.0, 0.0, 0.0, 0.1), 
         edgecolor = "red", 
         linewidth = 2,
         label = "y1")"""

#plt.legend()

# Design
ax1.set_xlabel("X", fontsize = 12)
ax1.set_ylabel("probability", fontsize = 12)
ax1.set_title("PDF Metaculus", fontsize = 16, pad = 20)

ax1.grid(True, linestyle = ":", linewidth = "1")

# Axes
#ax1.set_xlim(45, 200)
#ax1.set_ylim(0, 300)
plt.xscale("log")

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "inout", length = 7)
ax1.tick_params(which = "minor", direction = "in", length = 2)
ax1.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

plt.savefig("pdf Metaculus.png", bbox_inches = "tight")

print("All done!")

#-------------------------------------------
# Calling plt.show() to make graphics appear.
plt.show()