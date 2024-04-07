import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import gaussian_kde
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from random import choices
from scipy.interpolate import interp1d

# reading csv data from the API

url_p1 = "https://www.metaculus.com/api2/questions/1337/download_csv/"
url_p2 = "https://www.metaculus.com/api2/questions/1338/download_csv/"
url_p3 = "https://www.metaculus.com/api2/questions/1339/download_csv/"
url_p4 = "https://www.metaculus.com/api2/questions/1340/download_csv/"
url_p5 = "https://www.metaculus.com/api2/questions/1341/download_csv/"
url_p6 = "https://www.metaculus.com/api2/questions/1342/download_csv/"
url_p7 = "https://www.metaculus.com/api2/questions/1343/download_csv/"

# Reading and processing data

print("Reading...")

def cdf_setter(url, min, max, q = 10000):

    cdf_p = pd.DataFrame()

    # reading data
    df_p = pd.read_csv(url)

    # reducing data to just the CDF distribution
    df_p = df_p.iloc[-2, 211:412].reset_index(drop = False)
    df_p.columns = ['x_norm', 'CDF']
    df_p["x_norm"].replace(to_replace = "cdf_at_", value = "", regex = True, inplace = True)
    df_p["x_norm"] = pd.to_numeric(df_p["x_norm"], errors = 'coerce')#.astype("float")
    df_p["CDF"] = df_p["CDF"].astype("float")

    # Step 1: Adjust the CDF to start at 0
    df_p['CDF'] -= df_p['CDF'].min()

    # Step 2: Normalize to end at 1
    df_p['CDF'] /= df_p['CDF'].max()

    # inverse transform sampling
    uniform_samples = np.random.uniform(0, 1, q)

    # transforming
    xmin = np.log10(min)
    xmax = np.log10(max)
    #df_p["x_log"] = pd.Series(df_p["x_norm"]).apply(lambda x : min + x * ((max - min)))
    df_p["x_log"] = pd.Series(df_p["x_norm"]).apply(lambda x : 10**(xmin + x * (xmax - xmin)))

    # interpolating
    inverse_cdf = interp1d(df_p["CDF"], 
                            df_p["x_log"],#df_p["x_norm"], 
                            kind = 'linear', 
                            bounds_error = False, 
                            fill_value = "extrapolate")
    # sampling
    sampled_normalized_x = inverse_cdf(uniform_samples)

    #cdf_p["x_log"] = pd.Series(sampled_normalized_x).apply(lambda x : min + x * ((max - min)))
    #cdf_p["x_log"] = pd.Series(sampled_normalized_x).apply(lambda x : 10**(xmin + x * (xmax - xmin)))
    cdf_p["x_log"] = pd.Series(sampled_normalized_x)
    cdf_p["cdf_values"] = pd.Series(inverse_cdf(uniform_samples))

    #print(sampled_normalized_x)
    #print(cdf_p["x_log"])
    print(cdf_p.head())
    print(cdf_p.tail())

    return cdf_p.copy()

#--------------------------------
#P1
df_p1 = cdf_setter(url = url_p1, min = 0.01, max = 1000, q = 10000)
print("p1 done!")

#--------------------------------
#P2
df_p2 = cdf_setter(url = url_p2, min = 0.01, max = 1, q = 10000)
print("p2 done!")

#--------------------------------
#P3
df_p3 = cdf_setter(url = url_p3, min = 10**-6, max = 100, q = 10000)
print("p3 done!")

#--------------------------------
#P4
df_p4 = cdf_setter(url = url_p4, min = 10**-31, max = 1, q = 10000)
print("p4 done!")

#--------------------------------
#P5
df_p5 = cdf_setter(url = url_p5, min = 10**-20, max = 1, q = 10000)
print("p5 done!")

#--------------------------------
#P6
df_p6 = cdf_setter(url = url_p6, min = 10**-5, max = 1, q = 10000)
print("p6 done!")

#--------------------------------
#P7
df_p7 = cdf_setter(url = url_p7, min = 10, max = 10**10, q = 10000)
print("p7 done!")

#-------------------------------------------
# Multiplying the factors

print("Sampling")

p1_sampled = df_p1["x_log"]
p2_sampled = df_p2["x_log"]
p3_sampled = df_p3["x_log"]
p4_sampled = df_p4["x_log"]
p5_sampled = df_p5["x_log"]
p6_sampled = df_p6["x_log"]
p7_sampled = df_p7["x_log"]

print(p1_sampled)
print(p2_sampled)
print(p3_sampled)
print(p4_sampled)
print(p5_sampled)
print(p6_sampled)
print(p7_sampled)

print("Multiplying.")

p_samples = (p1_sampled * 
             p2_sampled * 
             p3_sampled * 
             p4_sampled * 
             p5_sampled * 
             p6_sampled * 
             p7_sampled)

print("Monte Carling.")

# Estimate the CDF of x using KDE
p_kde = gaussian_kde(p_samples)
p_values = np.linspace(min(p_samples), max(p_samples), 10000)
p_cdf = p_kde(p_values)

# Normalize the CDF values so that the sum equals 1
total_area = simps(p_cdf, p_values)
p_cdf = p_cdf / total_area

print("Together done!")

#-------------------------------------------

print("Making Figures.")

# Figures
#-------------------------------------------
fig2, ax2 = plt.subplots(figsize = (10, 6))

ax2.plot(p_values, p_cdf,
         color = "red",
         linewidth = 2,
         label = "fermi")

# Design
ax2.set_xlabel("Civilizations in our Galaxy", fontsize = 12)
ax2.set_ylabel("probability", fontsize = 12)
ax2.set_title("Fermi Metaculus", fontsize = 16, pad = 20)

ax2.grid(True, linestyle = ":", linewidth = "1")

# Axes
#ax2.set_xlim(45, 200)
#ax2.set_ylim(0, 300)
plt.xscale("log")

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "inout", length = 7)
ax2.tick_params(which = "minor", direction = "in", length = 2)
ax2.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

plt.savefig("Fermi Metaculus CDF.png", bbox_inches = "tight")

#-------------------------------------------
# Create a figure object with size 14x8 inches and 6 subfigs
fig3, axes = plt.subplots(nrows = 3,
                          ncols = 3,
                          figsize = (14, 8))

# axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

#----------------------
#P1
ax1.plot(df_p1['x_log'], df_p1['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p1")

# Design
ax1.set_xlabel("average rate of formation of suitable stars (stars/year)", fontsize = 8)
ax1.set_ylabel("probability", fontsize = 12)
ax1.set_title("P1", fontsize = 16, pad = 20)
ax1.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax1.set_xscale("log")

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "inout", length = 7)
ax1.tick_params(which = "minor", direction = "in", length = 2)
ax1.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#P2
ax2.plot(df_p2['x_log'], df_p2['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p2")

# Design
ax2.set_xlabel("fraction of stars that form planets", fontsize = 8)
ax2.set_ylabel("probability", fontsize = 12)
ax2.set_title("p2", fontsize = 16, pad = 20)
ax2.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax2.set_xscale("log")

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "inout", length = 7)
ax2.tick_params(which = "minor", direction = "in", length = 2)
ax2.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#P3
ax3.plot(df_p3['x_log'], df_p3['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p3")

# Design
ax3.set_xlabel("average number of habitable planets per star", fontsize = 8)
ax3.set_ylabel("probability", fontsize = 12)
ax3.set_title("p3", fontsize = 16, pad = 20)
ax3.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax3.set_xscale("log")

ax3.minorticks_on()
ax3.tick_params(which = "major", direction = "inout", length = 7)
ax3.tick_params(which = "minor", direction = "in", length = 2)
ax3.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax3.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#P4
ax4.plot(df_p4['x_log'], df_p4['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p4")

# Design
ax4.set_xlabel("fraction of habitable planets in which life emerges", fontsize = 8)
ax4.set_ylabel("probability", fontsize = 12)
ax4.set_title("p4", fontsize = 16, pad = 20)
ax4.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax4.set_xscale("log")

ax4.minorticks_on()
ax4.tick_params(which = "major", direction = "inout", length = 7)
ax4.tick_params(which = "minor", direction = "in", length = 2)
ax4.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax4.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#P5
ax5.plot(df_p5['x_log'], df_p5['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p5")

# Design
ax5.set_xlabel("fraction of habitable planets where intelligence life emerges", fontsize = 8)
ax5.set_ylabel("probability", fontsize = 12)
ax5.set_title("p5", fontsize = 16, pad = 20)
ax5.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax5.set_xscale("log")

ax5.minorticks_on()
ax5.tick_params(which = "major", direction = "inout", length = 7)
ax5.tick_params(which = "minor", direction = "in", length = 2)
ax5.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax5.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#P6
ax6.plot(df_p6['x_log'], df_p6['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p6")

# Design
ax6.set_xlabel("fraction of planets where life is capable of interstellar communication", fontsize = 8)
ax6.set_ylabel("probability", fontsize = 12)
ax6.set_title("p6", fontsize = 16, pad = 20)
ax6.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax6.set_xscale("log")

ax6.minorticks_on()
ax6.tick_params(which = "major", direction = "inout", length = 7)
ax6.tick_params(which = "minor", direction = "in", length = 2)
ax6.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax6.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#P7
ax7.plot(df_p7['x_log'], df_p7['cdf_values'],
         color = "red",
         linewidth = 2,
         label = "p7")

# Design
ax7.set_xlabel("years a civilization remains detectable", fontsize = 8)
ax7.set_ylabel("probability", fontsize = 12)
ax7.set_title("p7", fontsize = 16, pad = 20)
ax7.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax7.set_xscale("log")

ax7.minorticks_on()
ax7.tick_params(which = "major", direction = "inout", length = 7)
ax7.tick_params(which = "minor", direction = "in", length = 2)
ax7.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax7.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#Histo P3

logbins = np.logspace(np.log10(min(p3_sampled)), np.log10(max(p3_sampled)), 50)
#print(logbins)

ax8.hist(p3_sampled, 
         #range = (45, 200), 
         histtype = "step", 
         fill = True, 
         density = True, 
         #alpha = 0.2, 
         bins = logbins,
         color = (0.0, 0.0, 1.0, 0.1), 
         edgecolor = "blue", 
         linewidth = 2,
         label = "Histo P3")

# Design
ax8.set_xlabel("X", fontsize = 8)
ax8.set_ylabel("Frequency", fontsize = 12)
ax8.set_title("Histo P3", fontsize = 16, pad = 20)
ax8.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax8.set_xscale("log")

ax8.minorticks_on()
ax8.tick_params(which = "major", direction = "inout", length = 7)
ax8.tick_params(which = "minor", direction = "in", length = 2)
ax8.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax8.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#----------------------
#Histo p_samples

logbins = np.logspace(np.log10(min(p_samples)), np.log10(max(p_samples)), 50)
print(logbins)
#p1_cdf, p1_values
ax9.hist(p_samples,#df_p1["x_log"],#p1_cdf,#p1_sampled, 
         #range = (45, 200), 
         histtype = "step", 
         fill = True, 
         density = True, 
         #alpha = 0.2, 
         bins = logbins,
         color = (0.0, 0.0, 1.0, 0.1), 
         edgecolor = "blue", 
         linewidth = 2,
         label = "Histo p_samples")

# Design
ax9.set_xlabel("X", fontsize = 8)
ax9.set_ylabel("Frequency", fontsize = 12)
ax9.set_title("Histo P1", fontsize = 16, pad = 20)
ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "inout", length = 7)
ax9.tick_params(which = "minor", direction = "in", length = 2)
ax9.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

# Adjusting the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Fermi Metaculus Parameters CDF.png", bbox_inches = "tight")

print("All done!")

#-------------------------------------------
# Calling plt.show() to make graphics appear.
plt.show()