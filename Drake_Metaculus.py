import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy import interpolate as inter
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from statsmodels.distributions.empirical_distribution import ECDF

# urls of the csv data from the API
url_Rs = "https://www.metaculus.com/api2/questions/1337/download_csv/"
url_fp = "https://www.metaculus.com/api2/questions/1338/download_csv/"
url_ne = "https://www.metaculus.com/api2/questions/1339/download_csv/"
url_fl = "https://www.metaculus.com/api2/questions/1340/download_csv/"
url_fi = "https://www.metaculus.com/api2/questions/1341/download_csv/"
url_fc = "https://www.metaculus.com/api2/questions/1342/download_csv/"
url_L = "https://www.metaculus.com/api2/questions/1343/download_csv/"


quantity = 10**6 # quantity of data in the simulation

# function to read, format, reduce, and transform data
def cdf_setter(url, xmin, xmax, q = 1000):

     # dataframe for use later
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

     # transforming x range from 0-1 to the real xmin-xmax
     xmin_log = np.log10(xmin)
     xmax_log = np.log10(xmax)
     df_cdf["x_log"] = pd.DataFrame(df_cdf["x_norm"]).apply(lambda x : (xmin_log + x * (xmax_log - xmin_log)))
     df_pdf["x_log"] = pd.DataFrame(df_pdf["x_norm"]).apply(lambda x : 10**(xmin_log + x * (xmax_log - xmin_log)))

     # Calculating the area under the curve
     total_area = simps(df_pdf["PDF"], df_pdf["x_log"])
     # Normalizing the PDF values so that the sum equals 1
     df_pdf['PDF'] = df_pdf["PDF"] / total_area
     #df_pdf['PDF'] = df_pdf["PDF"] / df_pdf["PDF"].sum()

     # inverse interpolation
     inverse_cdf = CubicSpline(df_cdf["CDF"], df_cdf["x_log"])

     # sampling
     sampled_normalized = inverse_cdf(uniform_samples)

     # filling cdf dataframe to be returned
     cdf_f["x_log"] = pd.DataFrame(sampled_normalized).apply(lambda x : 10**x )
     cdf_f["cdf_values"] = pd.DataFrame(uniform_samples)

     return cdf_f.copy(), df_pdf.copy()

#--------------------------------
# executing the function on the data
print("Reading and reducing the data...")

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
print("Multiplying the factors...")

N_cdf = (df_Rs_cdf["x_log"] * 
         df_fp_cdf["x_log"] * 
         df_ne_cdf["x_log"] * 
         df_fl_cdf["x_log"] * 
         df_fi_cdf["x_log"] * 
         df_fc_cdf["x_log"] * 
         df_L_cdf["x_log"])

#-------------------------------------------
# Making the figures
print("Making the histograms...")

#----------------------
# Function to make the histograms sub figures
def histogram(ax, df_f, color, edgecolor, label, xlabel, title, q, nbins):

     logbins = np.logspace(np.log10(min(df_f)), np.log10(max(df_f)), nbins)
     weights = np.zeros_like(df_f) + 1./q

     ax.hist(df_f,
             histtype = "step",
             fill = True,
             bins = logbins,
             weights = weights,
             color = color,
             edgecolor = edgecolor, 
             linewidth = 2,
             label = label)

     # Design
     ax.set_xlabel(xlabel, fontsize = 8)
     ax.set_ylabel("Frequency", fontsize = 10)
     ax.set_title(title, fontsize = 12, pad = 10)
     #ax.grid(True, linestyle = ":", linewidth = "1")

     # Axes
     ax.set_xscale("log")

     ax.minorticks_on()
     ax.tick_params(which = "major", direction = "inout", length = 7)
     ax.tick_params(which = "minor", direction = "in", length = 2)
     ax.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
     ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#-------------------------------------------
# Creating a figure object with size 14x8 inches and 9 subfigs
fig1, axes = plt.subplots(nrows = 3,
                          ncols = 3,
                          figsize = (14, 8))

# axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

fig1.suptitle("Counts by the CDF", fontsize = 14)

#----------------------
# HISTOGRAMS
nbins = 100

# Rs
histogram(ax = ax1, 
          df_f = df_Rs_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f1", 
          xlabel = "Average SFR of suitable stars [stars/year]", 
          title = "$R_{*}$", 
          q = quantity,
          nbins = nbins)
# fp
histogram(ax = ax2, 
          df_f = df_fp_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f2", 
          xlabel = "Fraction of stars that form planets", 
          title = "$f_{p}$", 
          q = quantity,
          nbins = nbins)
# ne
histogram(ax = ax3, 
          df_f = df_ne_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f3", 
          xlabel = "Average number of habitable planets per star", 
          title = "$n_{e}$", 
          q = quantity,
          nbins = nbins)
# fl
histogram(ax = ax4, 
          df_f = df_fl_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f4", 
          xlabel = "Fraction of habitable planets in which life emerges", 
          title = "$f_{l}$", 
          q = quantity,
          nbins = nbins)
# fi
histogram(ax = ax5, 
          df_f = df_fi_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f5", 
          xlabel = "Fraction of habitable planets where intelligence life emerges", 
          title = "$f_{i}$", 
          q = quantity,
          nbins = nbins)
# fc
histogram(ax = ax6, 
          df_f = df_fc_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f6", 
          xlabel = "Fraction of planets where life is capable of interstellar communication", 
          title = "$f_{c}$", 
          q = quantity,
          nbins = nbins)
# L
histogram(ax = ax7, 
          df_f = df_L_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f7", 
          xlabel = "Years a civilization remains detectable", 
          title = "$L$", 
          q = quantity,
          nbins = nbins)
# N PDF
histogram(ax = ax8, 
          df_f = N_cdf,
          color = (1.0, 0.0, 0.0, 0.1),
          edgecolor = "red",
          label = "Histo N", 
          xlabel = "Number of civilizations in our Galaxy", 
          title = "$N$, number of civilizations in our galaxy", 
          q = quantity,
          nbins = nbins)

#----------------------
# N CDF
logbins = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), nbins)
weights = np.zeros_like(N_cdf) + 1./quantity

ax9.hist(N_cdf, 
         bins = logbins, 
         histtype = "step",
         fill = True,
         weights = weights,
         cumulative = True, 
         color = (1.0, 0.0, 0.0, 0.1),
         edgecolor = "red",
         linewidth = 2, 
         label = "Histo N cumulative")

# Design
ax9.set_xlabel("Number of civilizations in our Galaxy", fontsize = 8)
ax9.set_ylabel("Cumulative frequency", fontsize = 10)
ax9.set_title("CDF of $N$", fontsize = 12, pad = 10)
#ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "inout", length = 7)
ax9.tick_params(which = "minor", direction = "in", length = 2)
ax9.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

ecdft = ECDF(N_cdf)
ax9.fill_between(logbins, ecdft(logbins), color = "red", alpha = 0.1)
                 
#----------------------
# Adjusting the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus histos.png", bbox_inches = "tight")

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
    #ax.grid(True, linestyle = ":", linewidth = "1")

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

fig2.suptitle("Drake equation factors by Metaculus", fontsize = 14)

#----------------------
# PLOTS
print("Making the plots...")

# Rs
plot(ax = ax1, 
     df_f = df_Rs_pdf,
     color = "blue",
     label = "Plot f1",
     xlabel = "Average SFR of suitable stars [stars/year]",
     title = "$R_{*}$")
# fp
plot(ax = ax2,
     df_f = df_fp_pdf,
     color = "blue",
     label = "Plot f2", 
     xlabel = "Fraction of stars that form planets",
     title = "$f_{p}$")
# ne
plot(ax = ax3, 
     df_f = df_ne_pdf,
     color = "blue",
     label = "Plot f3",
     xlabel = "Average number of habitable planets per star", 
          title = "$n_{e}$")
# fl
plot(ax = ax4, 
     df_f = df_fl_pdf,
     color = "blue",
     label = "Plot f4", 
     xlabel = "Fraction of habitable planets in which life emerges", 
     title = "$f_{l}$")
# fi
plot(ax = ax5, 
     df_f = df_fi_pdf,
     color = "blue",
     label = "Plot f5", 
     xlabel = "Fraction of habitable planets where intelligence life emerges", 
     title = "$f_{i}$")
# fc
plot(ax = ax6, 
     df_f = df_fc_pdf,
     color = "blue",
     label = "Plot f6", 
     xlabel = "Fraction of planets where life is capable of interstellar communication", 
     title = "$f_{c}$")
# L
plot(ax = ax7, 
     df_f = df_L_pdf,
     color = "blue",
     label = "Plot f7", 
     xlabel = "Years a civilization remains detectable", 
     title = "$L$")

#----------------------
# N smooth

# Normalizing histogram counts
log_bin_edges = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), 101)

counts, _ = np.histogram(N_cdf, bins = log_bin_edges) #counts, bins

log_bin_widths = np.diff(log_bin_edges)
total_observations = counts.sum()
density = counts / total_observations#(log_bin_widths * total_observations)

# Interpolating on the data from the histogram
log_bin_midpoints = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2
bin_midpoints = np.log10(log_bin_midpoints)
interpolation = inter.UnivariateSpline(bin_midpoints, density, s = 10**-6)

# Data for the smooth curve
bin_edges = np.log10(log_bin_edges[1:])
N_PDF = interpolation(bin_edges)

#verification = simps(N_PDF, log_bin_edges[1:])
#print(verification)
#-----------
ax8.plot(log_bin_midpoints, N_PDF, 
         color = "red", 
         linewidth = 2, 
         label = "Plot N")

# Design
ax8.set_xlabel("Number of civilizations in our Galaxy", fontsize = 8)
ax8.set_ylabel("Probability", fontsize = 10)
ax8.set_title("$N$, number of civilizations in our galaxy", fontsize = 12, pad = 10)
#ax8.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax8.set_xscale("log")

ax8.minorticks_on()
ax8.tick_params(which = "major", direction = "inout", length = 7)
ax8.tick_params(which = "minor", direction = "in", length = 2)
ax8.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax8.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

ax8.fill_between(log_bin_midpoints, N_PDF, color = "red", alpha = 0.1)

#----------------------
# N CDF
logbins = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), 1000)

ax9.ecdf(N_cdf, 
         color = "red", 
         linewidth = 2,
         label = "Plot N ecdf")

# Design
ax9.set_xlabel("Number of civilizations in our Galaxy", fontsize = 8)
ax9.set_ylabel("Cumulative probability", fontsize = 10)
ax9.set_title("CDF of $N$", fontsize = 12, pad = 10)
#ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "inout", length = 7)
ax9.tick_params(which = "minor", direction = "in", length = 2)
ax9.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

#ecdft = ECDF(N_cdf)
#points = np.logspace(start = -40, stop = 11, num = 1000)
ax9.fill_between(logbins, ecdft(logbins), color = "red", alpha = 0.1)

#----------------------
# Adjusting the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDFs.png", bbox_inches = "tight")

#-------------------------------------------
# Figure double
# Creating a figure object with size 6x8 inches and 2 subfigs
fig3, axes = plt.subplots(nrows = 2,
                          ncols = 1,
                          figsize = (6, 8))

# axes is a 1D numpy array of AxesSubplot objects
ax1, ax2 = axes[0], axes[1]

logbins = np.logspace(np.log10(min(N_cdf)), np.log10(max(N_cdf)), 1000)
#----------------------
# N smooth
ax1.plot(log_bin_midpoints, N_PDF, 
         color = "red", 
         linewidth = 2, 
         label = "Plot N",
         zorder = 5)

# Design
ax1.set_xlabel("Number of civilizations in our Galaxy", fontsize = 10)
ax1.set_ylabel("Probability", fontsize = 10)
ax1.set_title("$N$, number of civilizations in our galaxy", fontsize = 12, pad = 10)
#ax1.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax1.set_xscale("log")

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "inout", length = 7)
ax1.tick_params(which = "minor", direction = "in", length = 2)
ax1.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

# Color filling
ax1.fill_between(log_bin_midpoints, N_PDF, color = "red", alpha = 0.1)

#---------
# Alone in the Galaxy marker filling
points = np.logspace(np.log10(min(N_cdf)), np.log10(1), 1000)
ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch = '\\\\', 
                 edgecolor = (0,0,0,0.3), 
                 fc = (0,1,0,0.1), 
                 linewidth = 0.0,
                 zorder = 1)
ax1.vlines(x = 1, 
           color = "green", 
           ymin = 0, 
           ymax = interpolation(0), 
           linestyle = "dashed",
           linewidth = 1.5, 
           zorder = 3)

# Adding text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_MW = ecdft(1)*100
formatted_probability = f"{probability_alone_MW:.0f}%"
text = "Probability to be alone \nin the Milky Way galaxy \n($N = 1$): " + formatted_probability
ax1.text(10**-32, 0.025, text, fontsize = 10)
ax1.text(10**-8, 0.009, f'{probability_alone_MW:.0f}%', fontsize = 12)
#---------
# Alone in the observable Universe marker filling
points = np.logspace(np.log10(min(N_cdf)), np.log10(5*10**-13), 1000)
ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch = '//', 
                 edgecolor = (0,0,0,0.3), 
                 fc = (0,1,0,0.1), 
                 linewidth = 0.0,
                 zorder = 2)
ax1.vlines(x = 5*10**-13, 
           color = "red", 
           ymin = 0, 
           ymax = interpolation(-12.3), 
           linestyle = "dashed",
           linewidth = 1.5, 
           zorder = 4)

# Adding text with info of probability of being alone in the observable Universe
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_OU = ecdft(5*10**-13)*100
formatted_probability = f"{probability_alone_OU:.0f}%"
text = "Probability to be alone \nin the observable Universe \n($N = 5 \\times 10^{{-13}}$): " + formatted_probability
ax1.text(10**-50, 0.004, text, fontsize = 10)
ax1.text(10**-20, 0.0025, f'{probability_alone_OU:.0f}%', fontsize = 12)

#----------------------
# N CDF
ax2.ecdf(N_cdf, 
         color = "red", 
         linewidth = 2,
         label = "Plot N ecdf", 
         zorder = 5)

# Design
ax2.set_xlabel("Number of civilizations in our Galaxy", fontsize = 10)
ax2.set_ylabel("Cumulative probability", fontsize = 10)
ax2.set_title("CDF of $N$", fontsize = 12, pad = 10)
#ax2.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax2.set_xscale("log")

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "inout", length = 7)
ax2.tick_params(which = "minor", direction = "in", length = 2)
ax2.tick_params(which = "both", bottom = True, top = True, left = True, right = True)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

# Color filling
#ecdft = ECDF(N_cdf)
ax2.fill_between(logbins, ecdft(logbins), color = "red", alpha = 0.1)

#---------
# Alone in the Galaxy marker filling
"""ax2.fill_between(logbins, 
                 ecdft(logbins), 
                 hatch = '\\\\', 
                 edgecolor = (0,0,0,0.3), 
                 fc = (0,1,0,0.1), 
                 linewidth = 0.0, 
                 zorder = 1)"""
ax2.vlines(x = 1, 
           color = "green", 
           ymin = 0, 
           ymax = ecdft(1), 
           linestyle = "dashed",
           linewidth = 1.5, 
           zorder = 3)

# Adding text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_MW = ecdft(1)*100
formatted_probability = f"{probability_alone_MW:.0f}%"
text = "Probability to be alone \nin the Milky Way galaxy: \n$N = 1$: " + formatted_probability
ax2.text(10**-27, 0.7, text, fontsize = 10)

#---------
# Alone in the observable Universe marker filling
"""ax2.fill_between(logbins, 
                 ecdft(logbins), 
                 hatch = '//', 
                 edgecolor = (0,0,0,0.3), 
                 fc = (0,1,0,0.1), 
                 linewidth = 0.0, 
                 zorder = 2)"""
ax2.vlines(x = 5*10**-13, 
           color = "red", 
           ymin = 0, 
           ymax = ecdft(5*10**-13), 
           linestyle = "dashed",
           linewidth = 1.5, 
           zorder = 4)

# Adding text with info of probability of being alone in the observable Universe
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_OU = ecdft(5*10**-13)*100
formatted_probability = f"{probability_alone_OU:.0f}%"
text = "Probability to be alone \nin the observable Universe: \n$N = 5 \\times 10^{{-13}}$: " + formatted_probability
ax2.text(10**-42, 0.15, text, fontsize = 10)

#----------------------
# Adjusting the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDF and CDF.png", bbox_inches = "tight")

#-------------------------------------------
print("All done!")

#-------------------------------------------
# Calling plt.show() to make graphics appear.
plt.show()