import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.stats import gaussian_kde
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats

# reading csv data from the API
url_Rs = "https://www.metaculus.com/api2/questions/1337/download_csv/"
url_fp = "https://www.metaculus.com/api2/questions/1338/download_csv/"
url_ne = "https://www.metaculus.com/api2/questions/1339/download_csv/"
url_fl = "https://www.metaculus.com/api2/questions/1340/download_csv/"
url_fi = "https://www.metaculus.com/api2/questions/1341/download_csv/"
url_fc = "https://www.metaculus.com/api2/questions/1342/download_csv/"
url_L = "https://www.metaculus.com/api2/questions/1343/download_csv/"

# Reading and processing data
print("Reading...")

# function to read, format, reduce, and transform data
def cdf_setter(url, xmin, xmax, q = 10000):

    cdf_f = pd.DataFrame()

    # reading data
    df_f = pd.read_csv(url)

    # reducing data to just the PDF distribution
    df_pdf = df_f.iloc[-2, 11:211].reset_index(drop = False)
    df_pdf.columns = ['x_norm', 'PDF']
    df_pdf["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
    df_pdf["x_norm"] = df_pdf["x_norm"].astype("float")
    df_pdf["PDF"] = df_pdf["PDF"].astype("float")

    # reducing data to just the CDF distribution
    df_cdf = df_f.iloc[-2, 211:412].reset_index(drop = False)
    df_cdf.columns = ['x_norm', 'CDF']
    df_cdf["x_norm"].replace(to_replace = "cdf_at_", value = "", regex = True, inplace = True)
    df_cdf["x_norm"] = pd.to_numeric(df_cdf["x_norm"], errors = 'coerce')
    df_cdf["CDF"] = df_cdf["CDF"].astype("float")

    # Step 1: Adjust the CDF to start at 0
    df_cdf['CDF'] -= df_cdf['CDF'].min()
    # Step 2: Normalize the CDF to end at 1
    df_cdf['CDF'] /= df_cdf['CDF'].max()

    # random number generator
    uniform_samples = np.random.uniform(0, 1, q)

    # transforming x range
    xmin_log = np.log10(xmin)
    xmax_log = np.log10(xmax)
    df_cdf["x_log"] = pd.DataFrame(df_cdf["x_norm"]).apply(lambda x : (xmin_log + x * (xmax_log - xmin_log)))
    df_pdf["x_log"] = pd.DataFrame(df_pdf["x_norm"]).apply(lambda x : 10**(xmin_log + x * (xmax_log - xmin_log)))

    # Calculate the area under the curve
    total_area = simps(df_pdf["PDF"], df_pdf["x_log"])
    # Normalize the PDF values so that the sum equals 1
    df_pdf['PDF'] = df_pdf["PDF"] / total_area
    #df_pdf['PDF'] = df_pdf["PDF"] / df_pdf["PDF"].sum()

    # interpolating
    inverse_cdf = CubicSpline(df_cdf["CDF"], df_cdf["x_log"])

    # sampling
    sampled_normalized = inverse_cdf(uniform_samples)

    cdf_f["x_log"] = pd.DataFrame(sampled_normalized).apply(lambda x : 10**x )
    cdf_f["cdf_values"] = pd.DataFrame(uniform_samples)

    return cdf_f.copy(), df_pdf.copy()

#--------------------------------

quantity = 100000

df_Rs_cdf, df_Rs_pdf = cdf_setter(url = url_Rs, xmin = 0.01, xmax = 1000, q = quantity)
print("Rs done!")
df_fp_cdf, df_fp_pdf = cdf_setter(url = url_fp, xmin = 0.01, xmax = 1, q = quantity)
print("fp done!")
df_ne_cdf, df_ne_pdf = cdf_setter(url = url_ne, xmin = 10**-6, xmax = 100, q = quantity)
print("ne done!")
df_fl_cdf, df_fl_pdf = cdf_setter(url = url_fl, xmin = 10**-31, xmax = 1, q = quantity)
print("fl done!")
df_fi_cdf, df_fi_pdf = cdf_setter(url = url_fi, xmin = 10**-20, xmax = 1, q = quantity)
print("fi done!")
df_fc_cdf, df_fc_pdf = cdf_setter(url = url_fc, xmin = 10**-5, xmax = 1, q = quantity)
print("fc done!")
df_L_cdf, df_L_pdf = cdf_setter(url = url_L, xmin = 10, xmax = 10**10, q = quantity)
print("L done!")

#-------------------------------------------
# Multiplying the factors
print("Multiplying...")

N_cdf = (df_Rs_cdf["x_log"] * 
         df_fp_cdf["x_log"] * 
         df_ne_cdf["x_log"] * 
         df_fl_cdf["x_log"] * 
         df_fi_cdf["x_log"] * 
         df_fc_cdf["x_log"] * 
         df_L_cdf["x_log"])

#-------------------------------------------
print("Making Figures...")
# Figures

#----------------------
# Function to make the histograms sub figures
def histogram(ax, df_f, color, edgecolor, label, xlabel, title):

    logbins = np.logspace(np.log10(min(df_f)), np.log10(max(df_f)), 1000)

    ax.hist(df_f,
            histtype = "step",
            fill = True,
            bins = logbins,
            color = color, 
            edgecolor = edgecolor, 
            linewidth = 2,
            label = label)

    # Design
    ax.set_xlabel(xlabel, fontsize = 8)
    ax.set_ylabel("Counts", fontsize = 10)
    ax.set_title(title, fontsize = 12, pad = 10)
    ax.grid(True, linestyle = ":", linewidth = "1")

    # Axes
    ax.set_xscale("log")

    ax.minorticks_on()
    ax.tick_params(which = "major", direction = "inout", length = 7)
    ax.tick_params(which = "minor", direction = "in", length = 2)
    ax.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
    ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#-------------------------------------------
# Create a figure object with size 14x8 inches and 6 subfigs
fig1, axes = plt.subplots(nrows = 3,
                          ncols = 3,
                          figsize = (14, 8))

# axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

fig1.suptitle("Counts by CDF", fontsize = 14)

#----------------------
#HISTOGRAMS
#F1
histogram(ax = ax1, 
          df_f = df_Rs_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f1", 
          xlabel = "Average SFR of suitable stars [stars/year]", 
          title = "$R_{*}$")
#F2
histogram(ax = ax2, 
          df_f = df_fp_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f2", 
          xlabel = "Fraction of stars that form planets", 
          title = "$f_{p}$")
#F3
histogram(ax = ax3, 
          df_f = df_ne_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f3", 
          xlabel = "Average number of habitable planets per star", 
          title = "$n_{e}$")
#F4
histogram(ax = ax4, 
          df_f = df_fl_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f4", 
          xlabel = "Fraction of habitable planets in which life emerges", 
          title = "$f_{l}$")
#F5
histogram(ax = ax5, 
          df_f = df_fi_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f5", 
          xlabel = "Fraction of habitable planets where intelligence life emerges", 
          title = "$f_{i}$")
#F6
histogram(ax = ax6, 
          df_f = df_fc_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f6", 
          xlabel = "Fraction of planets where life is capable of interstellar communication", 
          title = "$f_{c}$")
#F7
histogram(ax = ax7, 
          df_f = df_L_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f7", 
          xlabel = "Years a civilization remains detectable", 
          title = "$L$")
#F final PDF
histogram(ax = ax8, 
          df_f = N_cdf,
          color = (1.0, 0.0, 0.0, 0.1),
          edgecolor = "red",
          label = "Histo N", 
          xlabel = "Number of civilizations in our Galaxy", 
          title = "$N$, number of civilizations in our galaxy")

#----------------------
#F final CDF
logbins = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), 50)

ax9.ecdf(N_cdf, 
         color = "red", 
         linewidth = 2,
         label = "Histo N ecdf")

# Design
ax9.set_xlabel("Number of civilizations in our Galaxy", fontsize = 8)
ax9.set_ylabel("Cumulative probability", fontsize = 10)
ax9.set_title("CDF of $N$", fontsize = 12, pad = 10)
ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "inout", length = 7)
ax9.tick_params(which = "minor", direction = "in", length = 2)
ax9.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

ecdft = ECDF(N_cdf)
points = np.logspace(start = -49, stop = 12, num = 1000)
ax9.fill_between(points, ecdft(points), color = "red", alpha = 0.1)
                 
#----------------------
# Adjusting the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus CDF.png", bbox_inches = "tight")

#-------------------------------------------
# Function to make the plot sub figures
def plot(ax, df_f, color, label, xlabel, title):

    ax.plot(df_f['x_log'], 
            df_f['PDF'], 
            color = color, 
            linewidth = 2, 
            label = label)
    
    # Design
    ax.set_xlabel(xlabel, fontsize = 8)
    ax.set_ylabel("Probability", fontsize = 10)
    ax.set_title(title, fontsize = 12, pad = 10)
    ax.grid(True, linestyle = ":", linewidth = "1")

    # Axes
    ax.set_xscale("log")

    ax.minorticks_on()
    ax.tick_params(which = "major", direction = "inout", length = 7)
    ax.tick_params(which = "minor", direction = "in", length = 2)
    ax.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
    ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

    ax.fill_between(df_f['x_log'], df_f['PDF'], color = "blue", alpha = 0.1)
#-------------------------------------------
# PLOTS
# Create a figure object with size 14x8 inches and 9 subfigs
fig2, axes = plt.subplots(nrows = 3,
                          ncols = 3,
                          figsize = (14, 8))

# axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

fig2.suptitle("Factors by Metaculus", fontsize = 14)

#----------------------
#PLOTS
#F1
plot(ax = ax1, 
     df_f = df_Rs_pdf,
     color = "blue",
     label = "Plot f1",
     xlabel = "Average SFR of suitable stars [stars/year]",
     title = "$R_{*}$")
#F2
plot(ax = ax2,
     df_f = df_fp_pdf,
     color = "blue",
     label = "Plot f2", 
     xlabel = "Fraction of stars that form planets",
     title = "$f_{p}$")
#F3
plot(ax = ax3, 
     df_f = df_ne_pdf,
     color = "blue",
     label = "Plot f3",
     xlabel = "Average number of habitable planets per star", 
          title = "$n_{e}$")
#F4
plot(ax = ax4, 
     df_f = df_fl_pdf,
     color = "blue",
     label = "Plot f4", 
     xlabel = "Fraction of habitable planets in which life emerges", 
     title = "$f_{l}$")
#F5
plot(ax = ax5, 
     df_f = df_fi_pdf,
     color = "blue",
     label = "Plot f5", 
     xlabel = "Fraction of habitable planets where intelligence life emerges", 
     title = "$f_{i}$")
#F6
plot(ax = ax6, 
     df_f = df_fc_pdf,
     color = "blue",
     label = "Plot f6", 
     xlabel = "Fraction of planets where life is capable of interstellar communication", 
     title = "$f_{c}$")
#F7
plot(ax = ax7, 
     df_f = df_L_pdf,
     color = "blue",
     label = "Plot f7", 
     xlabel = "Years a civilization remains detectable", 
     title = "$L$")
#F final
histogram(ax = ax8, 
          df_f = N_cdf,
          color = (1.0, 0.0, 0.0, 0.1),
          edgecolor = "red",
          label = "Histo N", 
          xlabel = "Number of civilizations in our Galaxy", 
          title = "$N$, number of civilizations in our galaxy")

#----------------------
#F final CDF
logbins = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), 50)

ax9.ecdf(N_cdf, 
         color = "red", 
         linewidth = 2,
         label = "Histo N ecdf")

# Design
ax9.set_xlabel("Number of civilizations in our Galaxy", fontsize = 8)
ax9.set_ylabel("Cumulative probability", fontsize = 10)
ax9.set_title("CDF of $N$", fontsize = 12, pad = 10)
ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "inout", length = 7)
ax9.tick_params(which = "minor", direction = "in", length = 2)
ax9.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#ecdft = ECDF(N_cdf)
#points = np.logspace(start = -40, stop = 11, num = 1000)
ax9.fill_between(points, ecdft(points), color = "red", alpha = 0.1)

#----------------------
# Adjusting the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDF.png", bbox_inches = "tight")







#-------------------------------------------
# To normalize the N plot
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (0.1, 0.1))

_, bins, _ = ax.hist(N_cdf, bins = 1000) # n, bins, patches

logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
#logbins = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), 50)

weights = np.zeros_like(N_cdf) + 1./10000

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))

ax.hist(N_cdf,
         histtype = "step",
         fill = True,
         bins = logbins,
         weights = weights, 
         color = (0.1, 0.0, 0.0, 0.1),
         edgecolor = "red", 
         linewidth = 2,
         label = "Histo N")

# Design
ax.set_xlabel("Number of civilizations in our Galaxy", fontsize = 8)
ax.set_ylabel("Probability", fontsize = 10)
ax.set_title("$N$", fontsize = 12, pad = 10)
ax.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax.set_xscale("log")

ax.minorticks_on()
ax.tick_params(which = "major", direction = "inout", length = 7)
ax.tick_params(which = "minor", direction = "in", length = 2)
ax.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
#-------------------------------------------





#-------------------------------------------
print("All done!")

#-------------------------------------------
# Calling plt.show() to make graphics appear.
plt.show()