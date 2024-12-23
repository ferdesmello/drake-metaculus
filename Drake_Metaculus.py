"""
This script fetches data from the Metaculus API, processes it to obtain CDF and PDF data,
and provides functions to visualize the data using histograms and lineplots.

Modules:
    - numpy
    - pandas
    - matplotlib
    - scipy
    - statsmodels
    - requests
    - typing
"""
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
from scipy.integrate import simpson
from scipy import interpolate as inter
from scipy.stats import gaussian_kde
from scipy import interpolate
from statsmodels.distributions.empirical_distribution import ECDF
import requests
from typing import Dict, Union

#-------------------------------------------------------------------------------------------
print("Hello. Starting!")

# URLs of the API data
url_Rs = "https://www.metaculus.com/api2/questions/1337"
url_fp = "https://www.metaculus.com/api2/questions/1338"
url_ne = "https://www.metaculus.com/api2/questions/1339"
url_fl = "https://www.metaculus.com/api2/questions/1340"
url_fi = "https://www.metaculus.com/api2/questions/1341"
url_fc = "https://www.metaculus.com/api2/questions/1342"
url_L  = "https://www.metaculus.com/api2/questions/1343"

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def data_parser(url: str, 
                xmin: float, 
                xmax: float, 
                q: int = 1000) -> tuple[Dict[float, float], Dict[float, float]]:
     """
     Parses (read, format, reduce, and transform) data from a given Metaculus 
     API URL JSON file and processes it to obtain CDF and PDF data.

     Args:
          url (str): The URL of the Metaculus API question.
          xmin (float): The minimum value for the x-axis.
          xmax (float): The maximum value for the x-axis.
          q (int): The number of points sampled. Default is 1000.

     Returns:
          tuple of dictionaries:
          cdf_f (float, float): A dictionary containing the CDF data.
          df_pdf (float, float): A dictionary containing the PDF data.
     """
     #--------------------------------------------
     # Read the data
     # Make a GET request to the URL
     response = requests.get(url)

     # Check if the request was successful
     if response.status_code == 200:
          # Parse the JSON content
          data = response.json()

          # Extract the forecast_values array
          # For time weighted values
          forecast_values = data['question']['aggregations']['recency_weighted']['latest']['forecast_values']
          # For NOT time weighted values (but strange)
          #forecast_values = data['question']['aggregations']['metaculus_prediction']['latest']['forecast_values']

     else:
          print(f"Failed to retrieve data: Status code {response.status_code}")

     #--------------------------------------------
     # Get the CDF
     x_norm = np.linspace(0, 1, len(forecast_values))

     dictionary_cdf = {"x_norm": x_norm,
                       "CDF": forecast_values}
     df_cdf = pd.DataFrame(dictionary_cdf)
     
     df_cdf["x_norm"] = df_cdf["x_norm"].astype("float")
     df_cdf["CDF"] = df_cdf["CDF"].astype("float")

     #--------------------------------------------
     # Get the PDF
     # Create an interpolation function for the CDF
     cdf_interp = interpolate.interp1d(x_norm, forecast_values, kind='linear', bounds_error=False, fill_value=(0, 1))

     # Create a finer mesh for smoother plotting
     x_fine = np.linspace(0, 1, len(forecast_values))
     cdf_fine = cdf_interp(x_fine)

     # Calculate PDF by numerical differentiation
     pdf_values = np.gradient(cdf_fine, x_fine)

     dictionary_pdf = {"x_norm": x_fine,
                       "PDF": pdf_values}
     df_pdf = pd.DataFrame(dictionary_pdf)
     
     df_pdf["x_norm"] = df_pdf["x_norm"].astype("float")
     df_pdf["PDF"] = df_pdf["PDF"].astype("float")

     #--------------------------------------------
     # Adjust the CDF to start at 0
     df_cdf['CDF'] -= df_cdf['CDF'].min()
     # Normalize the CDF to end at 1
     df_cdf['CDF'] /= df_cdf['CDF'].max()

     # Random number generator
     uniform_samples = np.random.uniform(0, 1, q)

     # Transform x range from 0-1 to the real xmin-xmax
     xmin_log = np.log10(xmin)
     xmax_log = np.log10(xmax)

     df_cdf["x_log"] = pd.DataFrame(df_cdf["x_norm"]).apply(lambda x : (xmin_log + x * (xmax_log - xmin_log)))
     df_pdf["x_log"] = pd.DataFrame(df_pdf["x_norm"]).apply(lambda x : 10**(xmin_log + x * (xmax_log - xmin_log)))

     # Calculate the area under the curve
     total_area = simpson(y = df_pdf["PDF"], x = df_pdf["x_log"])
     # Normalize the PDF values so that the sum equals 1
     df_pdf['PDF'] = df_pdf["PDF"] / total_area
     #df_pdf['PDF'] = df_pdf["PDF"] / df_pdf["PDF"].sum()
     
     # Inverse interpolation
     inverse_cdf = inter.CubicSpline(df_cdf["CDF"], df_cdf["x_log"])

     # Sample
     sampled_normalized = inverse_cdf(uniform_samples)

     #---------------------------------------------
     # Fill in the cdf_f dataframe to be returned
     cdf_f = pd.DataFrame()
     cdf_f["x_log"] = pd.DataFrame(sampled_normalized).apply(lambda x : 10**x )
     cdf_f["cdf_values"] = pd.DataFrame(uniform_samples)

     return cdf_f.copy(), df_pdf.copy()

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Run the function
print("Reading and reducing the data...")

quantity = 10**6 # quantity of data in the simulation

df_Rs_cdf, df_Rs_pdf = data_parser(url = url_Rs, xmin = 0.01, xmax = 1000, q = quantity)
print("  Rs done.")
df_fp_cdf, df_fp_pdf = data_parser(url = url_fp, xmin = 0.01, xmax = 1, q = quantity)
print("  fp done.")
df_ne_cdf, df_ne_pdf = data_parser(url = url_ne, xmin = 10**-6, xmax = 100, q = quantity)
print("  ne done.")
df_fl_cdf, df_fl_pdf = data_parser(url = url_fl, xmin = 10**-31, xmax = 1, q = quantity)
print("  fl done.")
df_fi_cdf, df_fi_pdf = data_parser(url = url_fi, xmin = 10**-20, xmax = 1, q = quantity)
print("  fi done.")
df_fc_cdf, df_fc_pdf = data_parser(url = url_fc, xmin = 10**-5, xmax = 1, q = quantity)
print("  fc done.")
df_L_cdf, df_L_pdf = data_parser(url = url_L, xmin = 10, xmax = 10**10, q = quantity)
print("  L done.")

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Multiply the factors to find the N's
print("Multiplying the factors...")

Ns = (df_Rs_cdf["x_log"] * 
      df_fp_cdf["x_log"] * 
      df_ne_cdf["x_log"] * 
      df_fl_cdf["x_log"] * 
      df_fi_cdf["x_log"] * 
      df_fc_cdf["x_log"] * 
      df_L_cdf["x_log"])

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Make the figures
print("Making the figures...")
#-------------------------------------------------------------------------------------------
# HISTOGRAMS
print("  Making the histograms...")

# Number of bins for the histograms
nbins = 1000
# Custom dark gray color between dimgray and black
custom_dark_gray = (0.2, 0.2, 0.2)

#---------------------------------------
def histogram(ax: matplotlib.axes.Axes, 
              df_f: Union[np.ndarray, pd.Series], 
              color: str, 
              edgecolor: str, 
              label: str, 
              xlabel: str, 
              title: str, 
              q: int, 
              nbins: int) -> None:
     """
     Creates a Matplotlib histogram subfigure.

     Args:
          ax (matplotlib.axes.Axes): The axes on which to plot the histogram.
          df_f (array-like): Column in the pandas dataframe to be plotted in the histogram.
          color (str): The color of the histogram bars.
          edgecolor (str): The color of the edges of the histogram bars.
          label (str): The label for the histogram.
          xlabel (str): The label for the x-axis.
          title (str): The title of the histogram.
          q (int): The quantity of data in the simulation for the normalization weights.
          nbins (int): The number of bins for the histogram.
     """

     log_space_bins = np.logspace(np.log10(min(df_f)), np.log10(max(df_f)), nbins)
     # It is just the frequency per bin, it is not a normalization
     weights = np.zeros_like(df_f) + 1./q

     ax.hist(df_f,
             histtype = "step",
             fill = True,
             bins = log_space_bins,
             weights = weights,
             color = color,
             edgecolor = edgecolor, 
             linewidth = 1,
             label = label)

     # Design
     ax.set_xlabel(xlabel, fontsize = 12, color = custom_dark_gray)
     ax.set_ylabel("Frequency", fontsize = 10, color = custom_dark_gray)
     ax.set_title(title, fontsize = 10, pad = 10, color = custom_dark_gray)
     #ax.grid(True, linestyle = ":", linewidth = "1")

     # Axes
     ax.set_xscale("log")

     ax.minorticks_on()
     ax.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
     ax.tick_params(which = "minor", direction = "in", length = 0)
     ax.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
     ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
     ax.spines['right'].set_visible(False)
     ax.spines['top'].set_visible(False)
     ax.spines['left'].set_color(custom_dark_gray)
     ax.spines['bottom'].set_color(custom_dark_gray)
     ax.tick_params(axis = 'both', colors = custom_dark_gray)

#--------------------------------------------------------------------------------------------
# Create a figure object with size 14x8 inches and 9 subfigs
fig1, axes = plt.subplots(nrows = 3,
                          ncols = 3,
                          figsize = (14, 8))

# Axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

#---------------------------------------
# All this just to print a number pretty in the figure
# Converting number of simulations to float
number = float(quantity)
# Converting to scientific notation string to ensure exponent extraction
sci_notation = f"{number:.0e}"
# Extract the exponent part after 'e'
power = sci_notation.split('0')[1]
# Using LaTeX
LaTeX = f"$10^{{{power}}}$"
# All the text for the subtitle
text = "Frequencies given by Metaculus' CDFs, n = " + LaTeX
# A figure subtitle
fig1.suptitle(text, fontsize = 14, color = custom_dark_gray)
#-------------------------------------------------------------------------------------------
# Rs
histogram(ax = ax1, 
          df_f = df_Rs_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f1", 
          xlabel = "$R_{*}$ [stars/year]", 
          title = "Average SFR of suitable stars in our Galaxy", 
          q = quantity,
          nbins = nbins)
# fp
histogram(ax = ax2, 
          df_f = df_fp_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f2", 
          xlabel = "$f_{p}$", 
          title = "Fraction of stars that have planets", 
          q = quantity,
          nbins = nbins)
# ne
histogram(ax = ax3, 
          df_f = df_ne_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f3", 
          xlabel = "$n_{e}$", 
          title = "Average number of habitable planets per star", 
          q = quantity,
          nbins = nbins)
# fl
histogram(ax = ax4, 
          df_f = df_fl_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f4", 
          xlabel = "$f_{l}$", 
          title = "Fraction of habitable planets \nthat develops life", 
          q = quantity,
          nbins = nbins)
# fi
histogram(ax = ax5, 
          df_f = df_fi_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f5", 
          xlabel = "$f_{i}$", 
          title = "Fraction of habitable planets that \ndevelops intelligent life (civilizations)", 
          q = quantity,
          nbins = nbins)
# fc
histogram(ax = ax6, 
          df_f = df_fc_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f6", 
          xlabel = "$f_{c}$", 
          title = "Fraction of civilizations that \nreleases detectable signs", 
          q = quantity,
          nbins = nbins)
# L
histogram(ax = ax7, 
          df_f = df_L_cdf["x_log"],
          color = (0.0, 0.0, 1.0, 0.1),
          edgecolor = "blue",
          label = "Histo f7", 
          xlabel = "$L$ [Years]", 
          title = "Years a civilization releases detectable signs", 
          q = quantity,
          nbins = nbins)
# N
histogram(ax = ax8, 
          df_f = Ns,
          color = (1.0, 0.0, 0.0, 0.1),
          edgecolor = "red",
          label = "Histo N", 
          xlabel = "$N$", 
          title = "Number of civilizations in our Galaxy", 
          q = quantity,
          nbins = nbins)

#-------------------------------------------------------------------------------------------
# N cumulative fraction
log_space_bins = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), nbins)
weights = np.zeros_like(Ns) + 1./quantity

ax9.hist(Ns, 
         bins = log_space_bins, 
         histtype = "step",
         fill = True,
         weights = weights,
         cumulative = True, 
         color = (1.0, 0.0, 0.0, 0.1),
         edgecolor = "red",
         linewidth = 1, 
         label = "Histo N cumulative")

# Design
ax9.set_xlabel("$N$", fontsize = 12, color = custom_dark_gray)
ax9.set_ylabel("Cumulative frequency", fontsize = 10, color = custom_dark_gray)
ax9.set_title("CDF of the number of civilizations in our Galaxy", fontsize = 10, pad = 10, color = custom_dark_gray)
#ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
ax9.tick_params(which = "minor", direction = "in", length = 0)
ax9.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
ax9.spines['left'].set_color(custom_dark_gray)
ax9.spines['bottom'].set_color(custom_dark_gray)
ax9.tick_params(axis = 'both', colors = custom_dark_gray)

ecdft = ECDF(Ns)
ax9.fill_between(log_space_bins, ecdft(log_space_bins), color = "red", alpha = 0.1)
                 
#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus histos.png", bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# PLOTS
print("  Making the plots...")

#---------------------------------------
def plot(ax: matplotlib.axes.Axes, 
         df_f: Union[np.ndarray, pd.Series],
         color: str, 
         label: str, 
         xlabel: str, 
         title: str) -> None:
     """
     Creates a Matplotlib lineplot subfigure.

     Args:
          ax (matplotlib.axes.Axes): The axes on which to plot the lineplot.
          df_f (array-like): Column in the pandas dataframe to be plotted in the lineplot.
          color (str): The color of the lines.
          label (str): The label for the lineplot.
          xlabel (str): The label for the x-axis.
          title (str): The title of the lineplot.
     """

     ax.plot(df_f['x_log'],
             df_f['PDF'],
             color = color,
             linewidth = 2,
             label = label)

     # Design
     ax.set_xlabel(xlabel, fontsize = 12, color = custom_dark_gray)
     ax.set_ylabel("Probability", fontsize = 10, color = custom_dark_gray)
     ax.set_title(title, fontsize = 10, pad = 10, color = custom_dark_gray)
     #ax.grid(True, linestyle = ":", linewidth = "1")

     # Axes
     ax.set_xscale("log")

     ax.minorticks_on()
     ax.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
     ax.tick_params(which = "minor", direction = "in", length = 0)
     ax.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
     ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
     ax.spines['right'].set_visible(False)
     ax.spines['top'].set_visible(False)
     ax.spines['left'].set_color(custom_dark_gray)
     ax.spines['bottom'].set_color(custom_dark_gray)
     ax.tick_params(axis = 'both', colors = custom_dark_gray)

     ax.fill_between(df_f['x_log'], df_f['PDF'], color = "blue", alpha = 0.1)

#-------------------------------------------------------------------------------------------
# Create a figure object with size 14x8 inches and 9 subfigs
fig2, axes = plt.subplots(nrows = 3,
                          ncols = 3,
                          figsize = (14, 8))

# Axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

fig2.suptitle("Drake equation factors by Metaculus", fontsize = 14, color = custom_dark_gray)
#-------------------------------------------------------------------------------------------
# Rs
plot(ax = ax1, 
     df_f = df_Rs_pdf,
     color = "blue",
     label = "Plot f1",
     xlabel = "$R_{*}$ [stars/year]",
     title = "Average SFR of suitable stars in our Galaxy")
# fp
plot(ax = ax2,
     df_f = df_fp_pdf,
     color = "blue",
     label = "Plot f2", 
     xlabel = "$f_{p}$",
     title = "Fraction of stars that have planets")
# ne
plot(ax = ax3, 
     df_f = df_ne_pdf,
     color = "blue",
     label = "Plot f3",
     xlabel = "$n_{e}$", 
     title = "Average number of habitable planets per star")
# fl
plot(ax = ax4, 
     df_f = df_fl_pdf,
     color = "blue",
     label = "Plot f4", 
     xlabel = "$f_{l}$", 
     title = "Fraction of habitable planets \nthat develops life")
# fi
plot(ax = ax5, 
     df_f = df_fi_pdf,
     color = "blue",
     label = "Plot f5", 
     xlabel = "$f_{i}$", 
     title = "Fraction of habitable planets that \ndevelops intelligent life (civilizations)")
# fc
plot(ax = ax6, 
     df_f = df_fc_pdf,
     color = "blue",
     label = "Plot f6", 
     xlabel = "$f_{c}$", 
     title = "Fraction of civilizations that \nreleases detectable signs")
# L
plot(ax = ax7, 
     df_f = df_L_pdf,
     color = "blue",
     label = "Plot f7", 
     xlabel = "$L$ [Years]", 
     title = "Years a civilization releases detectable signs")
#-------------------------------------------------------------------------------------------
# N PDF
print('  Making the "smooth plot"...')

# Estimate the PDF in log space, because it is not working well in linear
log_data = np.log10(Ns)
N_density = gaussian_kde(log_data)
x = np.linspace(min(log_data), max(log_data), 100)

# This integral should be ~1
print("  The next integral should be ~1 \n    Integral =", simpson(y = N_density(x), x = x))

#---------------------------------------
ax8.plot(x, N_density(x), 
         color = "red", 
         linewidth = 2, 
         label = "Plot N")

# Design
ax8.set_xlabel("log($N$)", fontsize = 10, color = custom_dark_gray)
ax8.set_ylabel("Probability", fontsize = 10, color = custom_dark_gray)
ax8.set_title("Number of civilizations in our Galaxy", fontsize = 10, pad = 10, color = custom_dark_gray)
#ax8.grid(True, linestyle = ":", linewidth = "1")

# Axes
#ax8.set_xscale("log")
ax8.minorticks_on()
ax8.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
ax8.tick_params(which = "minor", direction = "in", length = 0)
ax8.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
ax8.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)
ax8.spines['left'].set_color(custom_dark_gray)
ax8.spines['bottom'].set_color(custom_dark_gray)
ax8.tick_params(axis = 'both', colors = custom_dark_gray)

ax8.fill_between(x, N_density(x), color = "red", alpha = 0.1)

#-------------------------------------------------------------------------------------------
# N CDF
print('  Making the CDF...')

log_space_bins = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 100)

ax9.ecdf(Ns, 
         color = "red", 
         linewidth = 2,
         label = "Plot N ecdf")

# Design
ax9.set_xlabel("$N$", fontsize = 12, color = custom_dark_gray)
ax9.set_ylabel("Cumulative probability", fontsize = 8)
ax9.set_title("CDF of the number of civilizations in our Galaxy", fontsize = 10, pad = 10, color = custom_dark_gray)
#ax9.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
ax9.tick_params(which = "minor", direction = "in", length = 0)
ax9.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
ax9.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
ax9.spines['left'].set_color(custom_dark_gray)
ax9.spines['bottom'].set_color(custom_dark_gray)
ax9.tick_params(axis = 'both', colors = custom_dark_gray)

ax9.fill_between(log_space_bins, ecdft(log_space_bins), color = "red", alpha = 0.1)

#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDFs.png", bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Double plot figure
print("  Making the double plot...")

# Create a figure object with size 6x8 inches and 2 subfigs
fig3, axes = plt.subplots(nrows = 2,
                          ncols = 1,
                          figsize = (6, 8))

# Axes is a 1D numpy array of AxesSubplot objects
ax1, ax2 = axes[0], axes[1]

#-------------------------------------------------------------------------------------------
# N frequency in a smooth curve
log_space_bins_edges = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 1001)

counts, _ = np.histogram(Ns, bins = log_space_bins_edges) # counts, bins

log_bin_widths = np.diff(log_space_bins_edges)
total_observations = counts.sum()
N_density = counts / total_observations

# Interpolate on the data from the histogram
# Use the middle of the bins, not the edges
log_space_bins_midpoints = (log_space_bins_edges[:-1] + log_space_bins_edges[1:]) / 2
log_of_bins_midpoints = np.log10(log_space_bins_midpoints)
interpolation = inter.UnivariateSpline(log_of_bins_midpoints, N_density, s = 1./(quantity))

# X and Y data for the smooth curve
bin_edges = np.log10(log_space_bins_edges[1:])
N_interpolated = interpolation(bin_edges)

#---------------------------------------
ax1.plot(log_space_bins_edges[1:], N_interpolated, 
         color = "red", 
         linewidth = 2, 
         label = "Plot N",
         zorder = 5)

# Design
ax1.set_xlabel("$N$", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Frequency of $N$", fontsize = 10, color = custom_dark_gray)
ax1.set_title("Number of civilizations in our Galaxy", fontsize = 12, pad = 10, color = custom_dark_gray)
#ax1.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax1.set_xscale("log")

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "in", length = 0)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

# Color filling
ax1.fill_between(log_space_bins_midpoints, N_interpolated, color = "red", alpha = 0.1)

#---------------------------------------
# Not alone in the Galaxy hatch filling
points = np.logspace(np.log10(1), np.log10(max(Ns)), 1000)

ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch = '||', 
                 edgecolor = (0,0,0,0.3), 
                 fc = (0,1,0,0.0), 
                 linewidth = 0.0,
                 zorder = 1)

# Text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_not_alone_MW = (ecdft(10**12) - ecdft(1))*100
formatted_probability = f"{probability_not_alone_MW:.0f}%"
text = "Probability of\n NOT being\n  alone in the\n   Milky Way\n    galaxy\n    ($N > 1$): " + formatted_probability
ax1.text(5*10**2, 0.0017, text, fontsize = 8, color = custom_dark_gray)
ax1.text(3*10**0, 0.00020, f'{probability_not_alone_MW:.0f}%', fontsize = 12, color = custom_dark_gray)

#---------------------------------------
# Alone in the Galaxy hatch filling and line
points = np.logspace(np.log10(min(Ns)), np.log10(1), 1000)

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

# Text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_MW = ecdft(1)*100
formatted_probability = f"{probability_alone_MW:.0f}%"
text = "Probability of being alone \nin the Milky Way galaxy \n($N < 1$): " + formatted_probability
ax1.text(5*10**-32, 0.0028, text, fontsize = 8, color = custom_dark_gray)
ax1.text(5*10**-9, 0.0013, f'{probability_alone_MW:.0f}%', fontsize = 12, color = custom_dark_gray)

#---------------------------------------
# Alone in the observable Universe hatch filling and line
points = np.logspace(np.log10(min(Ns)), np.log10(5*10**-13), 1000)

ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch = '/', 
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

# Text with info of probability of being alone in the observable Universe
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_OU = ecdft(5*10**-13)*100
formatted_probability = f"{probability_alone_OU:.0f}%"
text = "Probability of being alone \nin the observable Universe \n($N < 5 \\times 10^{{-13}}$): " + formatted_probability
ax1.text(10**-51, 0.0008, text, fontsize = 8, color = custom_dark_gray)
ax1.text(10**-24, 0.00028, f'{probability_alone_OU:.0f}%', fontsize = 12, color = custom_dark_gray)

#-------------------------------------------------------------------------------------------
# N CDF
log_space_bins = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 1000)

ax2.ecdf(Ns, 
         color = "red", 
         linewidth = 2,
         label = "Plot N ecdf", 
         zorder = 5)

# Design
ax2.set_xlabel("$N$", fontsize = 12, color = custom_dark_gray)
ax2.set_ylabel("Cumulative probability", fontsize = 10, color = custom_dark_gray)
ax2.set_title("CDF of the number of civilizations in our Galaxy", fontsize = 12, pad = 10, color = custom_dark_gray)
#ax2.grid(True, linestyle = ":", linewidth = "1")

# Axes
ax2.set_xscale("log")

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "out", length = 4, color = custom_dark_gray)
ax2.tick_params(which = "minor", direction = "in", length = 0)
ax2.tick_params(which = "both", bottom = True, top = False, left = True, right = False)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_color(custom_dark_gray)
ax2.spines['bottom'].set_color(custom_dark_gray)
ax2.tick_params(axis = 'both', colors = custom_dark_gray)

# Color filling
ax2.fill_between(log_space_bins, ecdft(log_space_bins), color = "red", alpha = 0.1)

#---------------------------------------
# Alone in the Galaxy line
ax2.vlines(x = 1, 
           color = "green", 
           ymin = 0, 
           ymax = ecdft(1), 
           linestyle = "dashed",
           linewidth = 1.5, 
           zorder = 3)

# Text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_MW = ecdft(1)*100
formatted_probability = f"{probability_alone_MW:.0f}%"
text = "Probability of being alone \nin the Milky Way galaxy: \n($N < 1$): " + formatted_probability
ax2.text(5*10**-25, 0.78, text, fontsize = 8, color = custom_dark_gray)

#---------------------------------------
# Alone in the observable Universe line
ax2.vlines(x = 5*10**-13, 
           color = "red", 
           ymin = 0, 
           ymax = ecdft(5*10**-13), 
           linestyle = "dashed",
           linewidth = 1.5, 
           zorder = 4)

# Text with info of probability of being alone in the observable Universe
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_OU = ecdft(5*10**-13)*100
formatted_probability = f"{probability_alone_OU:.0f}%"
text = "Probability of being alone \nin the observable Universe: \n($N < 5 \\times 10^{{-13}}$): " + formatted_probability
ax2.text(10**-40, 0.24, text, fontsize = 8, color = custom_dark_gray)

#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDF and CDF.png", bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
print("All done!")

#-------------------------------------------
# Call plt.show() to make the graphics appear.
plt.show()