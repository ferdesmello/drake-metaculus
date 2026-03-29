"""
This script fetches data from the Metaculus API, processes it to obtain CDF and 
PDF data for some questions regarding the parameters of the Drake equation, and 
plot the results in figures.

Modules:
     - numpy
     - pandas
     - matplotlib
     - scipy
     - statsmodels
     - requests
     - time
     - typing
     - dotenv
     - os
     - json
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
import time
from typing import Union
from dotenv import load_dotenv
import os
import json

#-------------------------------------------------------------------------------------------
print("Hello. Starting!")

# Set up the headers with the API key
load_dotenv()

load_dotenv("../KEYs/My_METACULUS_API_Key.env")
API_KEY = os.getenv("METACULUS_API_Key")

headers = {
"Authorization": f"Token {API_KEY}"
}

#--------------------------------------------
# URLs of the API data
url_Rs = "https://www.metaculus.com/api2/questions/1337"
url_fp = "https://www.metaculus.com/api2/questions/1338"
url_ne = "https://www.metaculus.com/api2/questions/1339"
url_fl = "https://www.metaculus.com/api2/questions/1340"
url_fi = "https://www.metaculus.com/api2/questions/1341"
url_fc = "https://www.metaculus.com/api2/questions/1342"
url_L  = "https://www.metaculus.com/api2/questions/1343"

# Local addresses for the forecasts (PDFs) 
f_path_Rs = "data/Drake's_Equation_First_Parameter_R_s/forecast_data.csv"
f_path_fp = "data/Drake's_Equation_Second_Parameter_f_p/forecast_data.csv"
f_path_ne = "data/Drake's_Equation_Third_Parameter_n_e/forecast_data.csv"
f_path_fl = "data/Drake's_Equation_Fourth_Parameter_f_l/forecast_data.csv"
f_path_fi = "data/Drake's_Equation_Fifth_Parameter_f_i/forecast_data.csv"
f_path_fc = "data/Drake's_Equation_Sixth_Parameter_f_c/forecast_data.csv"
f_path_L  = "data/Drake's_Equation_Seventh_Parameter_L/forecast_data.csv"

# Local addresses for the questions data
q_path_Rs = "data/Drake's_Equation_First_Parameter_R_s/question_data.csv"
q_path_fp = "data/Drake's_Equation_Second_Parameter_f_p/question_data.csv"
q_path_ne = "data/Drake's_Equation_Third_Parameter_n_e/question_data.csv"
q_path_fl = "data/Drake's_Equation_Fourth_Parameter_f_l/question_data.csv"
q_path_fi = "data/Drake's_Equation_Fifth_Parameter_f_i/question_data.csv"
q_path_fc = "data/Drake's_Equation_Sixth_Parameter_f_c/question_data.csv"
q_path_L  = "data/Drake's_Equation_Seventh_Parameter_L/question_data.csv"

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def data_parser(url: str = None, 
                d_type: int = 0,
                f_path: str = None, 
                q_path: str = None,
                q: int = 1000,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
     """
     Parses (read, format, reduce, and transform) data from a given Metaculus 
     local CSV question data or from an URL via the Metaculus API and then
     processes it to obtain CDF and PDF data.

     Args:
          url (str): The URL of the Metaculus API question.
          d_type (int): The type of data to process. 0 means user, 1 means community.
          f_path (str): The file path to the local CSV containing the forecast data.
          q_path (str): The file path to the local CSV containing the question data.
          q (int): The number of points sampled. Default is 1000.
          

     Returns:
          tuple of dictionaries:
          cdf_f (float, float): A pandas dataframe containing the CDF data.
          df_pdf (float, float): A pandas dataframe containing the PDF data.
     """
    
    # --- BRANCH 1: LOAD FROM LOCAL CSV ---
     if f_path and q_path and d_type == 0:
          # 1. Get Range from question_data.csv
          df_q = pd.read_csv(q_path)
          xmin = float(df_q["Lower Bound"].iloc[-1])
          xmax = float(df_q["Upper Bound"].iloc[-1])

          # 2. Get CDF values from forecast_data.csv
          df_f = pd.read_csv(f_path)
          # Use .iloc[:,-1] to get the last row, and the specific column name
          cdf_string = df_f["Continuous CDF"].iloc[-1]
          
          # The CSV stores the array as a string like "[0.1, 0.2...]". 
          # We need to convert it to a list of floats.
          forecast_values = json.loads(cdf_string)

     # --- BRANCH 2: LOAD FROM API ---
     elif url and d_type != 0:
          response = requests.get(url, headers=headers)
          if response.status_code == 200:
               data = response.json()
          else:
               raise Exception(f"Failed API request: {response.status_code}")
          
          # Extract scaling
          scaling = data['question'].get('scaling', {})
          xmin = scaling.get('range_min')
          xmax = scaling.get('range_max')

          # Extract values for user or community forecasts
          if d_type == 1:
               my_forecast = data['question'].get('my_forecasts', {}).get('latest')
               forecast_values = my_forecast['forecast_values'] if my_forecast else None
          elif d_type == 2:
               comm_forecast = data['question'].get('aggregations', {}).get('recency_weighted', {}).get('latest')
               forecast_values = comm_forecast['forecast_values'] if comm_forecast else None
               
          time.sleep(2) # Or else they get angry at us for making too many requests in a short time
    
     else:
          raise ValueError("You must provide either a URL or both CSV paths.")

     if forecast_values is None:
          raise ValueError("No forecast data found.")

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

# Do you want to use the local data (0), user forecast via API (1), or community forecast via API (2)?
# API data may not be available for some questions, so you may want to use the local data for those.
data_type = 0

# Quantity of data in the simulation.
quantity = 10**6

print("  Doing Rs...")
df_Rs_cdf, df_Rs_pdf = data_parser(url=url_Rs, d_type=data_type, f_path=f_path_Rs, q_path=q_path_Rs, q=quantity)
print("  Doing fp...")
df_fp_cdf, df_fp_pdf = data_parser(url=url_fp, d_type=data_type, f_path=f_path_fp, q_path=q_path_fp, q=quantity)
print("  Doing ne...")
df_ne_cdf, df_ne_pdf = data_parser(url=url_ne, d_type=data_type, f_path=f_path_ne, q_path=q_path_ne, q=quantity)
print("  Doing fl...")
df_fl_cdf, df_fl_pdf = data_parser(url=url_fl, d_type=data_type, f_path=f_path_fl, q_path=q_path_fl, q=quantity)
print("  Doing fi...")
df_fi_cdf, df_fi_pdf = data_parser(url=url_fi, d_type=data_type, f_path=f_path_fi, q_path=q_path_fi, q=quantity)
print("  Doing fc...")
df_fc_cdf, df_fc_pdf = data_parser(url=url_fc, d_type=data_type, f_path=f_path_fc, q_path=q_path_fc, q=quantity)
print("  Doing L...")
df_L_cdf, df_L_pdf   = data_parser(url=url_L,  d_type=data_type, f_path=f_path_L,  q_path=q_path_L,  q=quantity)

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
             histtype="step",
             fill=True,
             bins=log_space_bins,
             weights=weights,
             color=color,
             edgecolor=edgecolor, 
             linewidth=1,
             label=label)

     # Design
     ax.set_xlabel(xlabel, fontsize=12, color=custom_dark_gray)
     ax.set_ylabel("Frequency", fontsize=10, color=custom_dark_gray)
     ax.set_title(title, fontsize=10, pad=10, color=custom_dark_gray)
     #ax.grid(True, linestyle=":", linewidth="1")

     # Axes
     ax.set_xscale("log")

     ax.minorticks_on()
     ax.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
     ax.tick_params(which="minor", direction="in", length=0)
     ax.tick_params(which="both", bottom=True, top=False, left=True, right=False)
     ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
     ax.spines['right'].set_visible(False)
     ax.spines['top'].set_visible(False)
     ax.spines['left'].set_color(custom_dark_gray)
     ax.spines['bottom'].set_color(custom_dark_gray)
     ax.tick_params(axis='both', colors=custom_dark_gray)

#--------------------------------------------------------------------------------------------
# Create a figure object with size 14x8 inches and 9 subfigs
fig1, axes = plt.subplots(nrows=3,
                          ncols=3,
                          figsize=(14, 8))

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
fig1.suptitle(text, fontsize=14, color=custom_dark_gray)
#-------------------------------------------------------------------------------------------
# Rs
histogram(ax=ax1, 
          df_f=df_Rs_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f1", 
          xlabel="$R_{*}$ [stars/year]", 
          title="Average SFR of suitable stars in our Galaxy", 
          q=quantity,
          nbins=nbins)
# fp
histogram(ax=ax2, 
          df_f=df_fp_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f2", 
          xlabel="$f_{p}$", 
          title="Fraction of stars that have planets", 
          q=quantity,
          nbins=nbins)
# ne
histogram(ax=ax3, 
          df_f=df_ne_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f3", 
          xlabel="$n_{e}$", 
          title="Average number of habitable planets per star", 
          q=quantity,
          nbins=nbins)
# fl
histogram(ax=ax4, 
          df_f=df_fl_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f4", 
          xlabel="$f_{l}$", 
          title="Fraction of habitable planets \nthat develops life", 
          q=quantity,
          nbins=nbins)
# fi
histogram(ax=ax5, 
          df_f=df_fi_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f5", 
          xlabel="$f_{i}$", 
          title="Fraction of habitable planets that \ndevelops intelligent life (civilizations)", 
          q=quantity,
          nbins=nbins)
# fc
histogram(ax=ax6, 
          df_f=df_fc_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f6", 
          xlabel="$f_{c}$", 
          title="Fraction of civilizations that \nreleases detectable signs", 
          q=quantity,
          nbins=nbins)
# L
histogram(ax=ax7, 
          df_f=df_L_cdf["x_log"],
          color=(0.0, 0.0, 1.0, 0.1),
          edgecolor="blue",
          label="Histo f7", 
          xlabel="$L$ [Years]", 
          title="Years a civilization releases detectable signs", 
          q=quantity,
          nbins=nbins)
# N
histogram(ax=ax8, 
          df_f=Ns,
          color=(1.0, 0.0, 0.0, 0.1),
          edgecolor="red",
          label="Histo N", 
          xlabel="$N$", 
          title="Number of civilizations in our Galaxy", 
          q=quantity,
          nbins=nbins)

#-------------------------------------------------------------------------------------------
# N cumulative fraction
log_space_bins = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), nbins)
weights = np.zeros_like(Ns) + 1./quantity

ax9.hist(Ns, 
         bins=log_space_bins, 
         histtype="step",
         fill=True,
         weights=weights,
         cumulative=True, 
         color=(1.0, 0.0, 0.0, 0.1),
         edgecolor="red",
         linewidth=1, 
         label="Histo N cumulative")

# Design
ax9.set_xlabel("$N$", fontsize=12, color=custom_dark_gray)
ax9.set_ylabel("Cumulative frequency", fontsize=10, color=custom_dark_gray)
ax9.set_title("CDF of the number of civilizations in our Galaxy", fontsize=10, pad=10, color=custom_dark_gray)
#ax9.grid(True, linestyle=":", linewidth="1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
ax9.tick_params(which="minor", direction="in", length=0)
ax9.tick_params(which="both", bottom=True, top=False, left=True, right=False)
ax9.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
ax9.spines['left'].set_color(custom_dark_gray)
ax9.spines['bottom'].set_color(custom_dark_gray)
ax9.tick_params(axis='both', colors=custom_dark_gray)

ecdft = ECDF(Ns)
ax9.fill_between(log_space_bins, ecdft(log_space_bins), color="red", alpha=0.1)
                 
#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus histos.png", bbox_inches="tight")

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
             color=color,
             linewidth=2,
             label=label)

     # Design
     ax.set_xlabel(xlabel, fontsize=12, color=custom_dark_gray)
     ax.set_ylabel("Probability", fontsize=10, color=custom_dark_gray)
     ax.set_title(title, fontsize=10, pad=10, color=custom_dark_gray)
     #ax.grid(True, linestyle=":", linewidth="1")

     # Axes
     ax.set_xscale("log")

     ax.minorticks_on()
     ax.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
     ax.tick_params(which="minor", direction="in", length=0)
     ax.tick_params(which="both", bottom=True, top=False, left=True, right=False)
     ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
     ax.spines['right'].set_visible(False)
     ax.spines['top'].set_visible(False)
     ax.spines['left'].set_color(custom_dark_gray)
     ax.spines['bottom'].set_color(custom_dark_gray)
     ax.tick_params(axis='both', colors=custom_dark_gray)

     ax.fill_between(df_f['x_log'], df_f['PDF'], color="blue", alpha=0.1)

#-------------------------------------------------------------------------------------------
# Create a figure object with size 14x8 inches and 9 subfigs
fig2, axes = plt.subplots(nrows=3,
                          ncols=3,
                          figsize=(14, 8))

# Axes is a 2D numpy array of AxesSubplot objects
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
ax7, ax8, ax9 = axes[2, 0], axes[2, 1], axes[2, 2]

fig2.suptitle("Drake equation factors by Metaculus", fontsize=14, color=custom_dark_gray)
#-------------------------------------------------------------------------------------------
# Rs
plot(ax=ax1, 
     df_f=df_Rs_pdf,
     color="blue",
     label="Plot f1",
     xlabel="$R_{*}$ [stars/year]",
     title="Average SFR of suitable stars in our Galaxy")
# fp
plot(ax=ax2,
     df_f=df_fp_pdf,
     color="blue",
     label="Plot f2", 
     xlabel="$f_{p}$",
     title="Fraction of stars that have planets")
# ne
plot(ax=ax3, 
     df_f=df_ne_pdf,
     color="blue",
     label="Plot f3",
     xlabel="$n_{e}$", 
     title="Average number of habitable planets per star")
# fl
plot(ax=ax4, 
     df_f=df_fl_pdf,
     color="blue",
     label="Plot f4", 
     xlabel="$f_{l}$", 
     title="Fraction of habitable planets \nthat develops life")
# fi
plot(ax=ax5, 
     df_f=df_fi_pdf,
     color="blue",
     label="Plot f5", 
     xlabel="$f_{i}$", 
     title="Fraction of habitable planets that \ndevelops intelligent life (civilizations)")
# fc
plot(ax=ax6, 
     df_f=df_fc_pdf,
     color="blue",
     label="Plot f6", 
     xlabel="$f_{c}$", 
     title="Fraction of civilizations that \nreleases detectable signs")
# L
plot(ax=ax7, 
     df_f=df_L_pdf,
     color="blue",
     label="Plot f7", 
     xlabel="$L$ [Years]", 
     title="Years a civilization releases detectable signs")
#-------------------------------------------------------------------------------------------
# N PDF
print('  Making the "smooth plot"...')

# Estimate the PDF in log space, because it is not working well in linear
log_data = np.log10(Ns)
N_density = gaussian_kde(log_data)
x = np.linspace(min(log_data), max(log_data), 100)

# This integral should be ~1
print("  The next integral should be ~1 \n    Integral =", simpson(y=N_density(x), x=x))

#---------------------------------------
ax8.plot(x, N_density(x), 
         color="red", 
         linewidth=2, 
         label="Plot N")

# Design
ax8.set_xlabel("log($N$)", fontsize=10, color=custom_dark_gray)
ax8.set_ylabel("Probability", fontsize=10, color=custom_dark_gray)
ax8.set_title("Number of civilizations in our Galaxy", fontsize=10, pad=10, color=custom_dark_gray)
#ax8.grid(True, linestyle=":", linewidth="1")

# Axes
#ax8.set_xscale("log")
ax8.minorticks_on()
ax8.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
ax8.tick_params(which="minor", direction="in", length=0)
ax8.tick_params(which="both", bottom=True, top=False, left=True, right=False)
ax8.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)
ax8.spines['left'].set_color(custom_dark_gray)
ax8.spines['bottom'].set_color(custom_dark_gray)
ax8.tick_params(axis='both', colors=custom_dark_gray)

ax8.fill_between(x, N_density(x), color="red", alpha=0.1)

#-------------------------------------------------------------------------------------------
# N CDF
print('  Making the CDF...')

log_space_bins = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 100)

ax9.ecdf(Ns, 
         color="red", 
         linewidth=2,
         label="Plot N ecdf")

# Design
ax9.set_xlabel("$N$", fontsize=12, color=custom_dark_gray)
ax9.set_ylabel("Cumulative probability", fontsize=8)
ax9.set_title("CDF of the number of civilizations in our Galaxy", fontsize=10, pad=10, color=custom_dark_gray)
#ax9.grid(True, linestyle=":", linewidth="1")

# Axes
ax9.set_xscale("log")

ax9.minorticks_on()
ax9.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
ax9.tick_params(which="minor", direction="in", length=0)
ax9.tick_params(which="both", bottom=True, top=False, left=True, right=False)
ax9.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
ax9.spines['left'].set_color(custom_dark_gray)
ax9.spines['bottom'].set_color(custom_dark_gray)
ax9.tick_params(axis='both', colors=custom_dark_gray)

ax9.fill_between(log_space_bins, ecdft(log_space_bins), color="red", alpha=0.1)

#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDFs.png", bbox_inches="tight")

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Double plot figure
print("  Making the double plot...")

# Create a figure object with size 6x8 inches and 2 subfigs
fig3, axes = plt.subplots(nrows=2,
                          ncols=1,
                          figsize=(6, 8))

# Axes is a 1D numpy array of AxesSubplot objects
ax1, ax2 = axes[0], axes[1]

#-------------------------------------------------------------------------------------------
# N frequency in a smooth curve
log_space_bins_edges = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 1001)

counts, _ = np.histogram(Ns, bins=log_space_bins_edges) # counts, bins

log_bin_widths = np.diff(log_space_bins_edges)
total_observations = counts.sum()
N_density = counts / total_observations

# Interpolate on the data from the histogram
# Use the middle of the bins, not the edges
log_space_bins_midpoints = (log_space_bins_edges[:-1] + log_space_bins_edges[1:]) / 2
log_of_bins_midpoints = np.log10(log_space_bins_midpoints)
interpolation = inter.UnivariateSpline(log_of_bins_midpoints, N_density, s=1./(quantity))

# X and Y data for the smooth curve
bin_edges = np.log10(log_space_bins_edges[1:])
N_interpolated = interpolation(bin_edges)

#---------------------------------------
ax1.plot(log_space_bins_edges[1:], N_interpolated, 
         color="red", 
         linewidth=2, 
         label="Plot N",
         zorder=5)

# Design
ax1.set_xlabel("$N$", fontsize=12, color=custom_dark_gray)
ax1.set_ylabel("Frequency of $N$", fontsize=10, color=custom_dark_gray)
ax1.set_title("Number of civilizations in our Galaxy", fontsize=12, pad=10, color=custom_dark_gray)
#ax1.grid(True, linestyle=":", linewidth="1")

# Axes
ax1.set_xscale("log")

ax1.minorticks_on()
ax1.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
ax1.tick_params(which="minor", direction="in", length=0)
ax1.tick_params(which="both", bottom=True, top=False, left=True, right=False)
ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)
ax1.tick_params(axis='both', colors=custom_dark_gray)

# Color filling
ax1.fill_between(log_space_bins_midpoints, N_interpolated, color="red", alpha=0.1)

#---------------------------------------
# Not alone in the Galaxy hatch filling
points = np.logspace(np.log10(1), np.log10(max(Ns)), 1000)

ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch='||', 
                 edgecolor=(0,0,0,0.3), 
                 fc=(0,1,0,0.0), 
                 linewidth=0.0,
                 zorder=1)

# Limits of the plot, to know where to put the text
xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()

# Text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_not_alone_MW = (ecdft(10**12) - ecdft(1))*100
formatted_probability = f"{probability_not_alone_MW:.0f}%"
text = "Probability of\nNOT being\nalone in the\nMilky Way\ngalaxy\n($N > 1$): " + formatted_probability

# Text
x = 10**((interpolation(0) + np.log10(xmax)) * 0.5)
y = max((ymax * 0.3), (interpolation(0) * 0.5))
# 'xy' is the point on the graph (Data coords)
# 'xytext' is the offset (Points/Pixels)
ax1.annotate(text, 
             xy=(x, y),
             xytext=(10, 0), 
             textcoords='offset points',
             fontsize=8,
             color=custom_dark_gray, 
             ha='center',
             va='center',
             zorder=10)

"""ax1.text(0.90, 0.10, text, 
         transform=ax1.transAxes, 
         fontsize=8, 
         #verticalalignment='top',
         color=custom_dark_gray,
         zorder=10)"""

# Percentage text
x = 10**(2)
y = (ymax - ymin) * 0.05
ax1.text(x, 
         y, 
         f'{probability_not_alone_MW:.0f}%', 
         fontsize=12, 
         color=custom_dark_gray, 
         ha='left',
         va='bottom',
         zorder=10)

#---------------------------------------
# Alone in the Galaxy hatch filling and line
points = np.logspace(np.log10(min(Ns)), np.log10(1), 1000)

ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch='\\\\', 
                 edgecolor=(0,0,0,0.3), 
                 fc=(0,1,0,0.1), 
                 linewidth=0.0,
                 zorder=1)
ax1.vlines(x=1, 
           color="green", 
           ymin=0, 
           ymax=interpolation(0), 
           linestyle="dashed",
           linewidth=1.5, 
           zorder=3)

# Text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_MW = ecdft(1)*100
formatted_probability = f"{probability_alone_MW:.0f}%"
text = "Probability of being alone\nin the Milky Way galaxy\n($N < 1$): " + formatted_probability

# Text
x = 10**((np.log10(xmin) + (-12.3)) * 0.35)
y = max((ymax * 0.2), min((interpolation(-12.3) * 1.1), (ymax * 0.8)))
ax1.annotate(text, 
             xy=(x, y),
             xytext=(0, 0), 
             textcoords='offset points',
             fontsize=8,
             color=custom_dark_gray, 
             ha='center',
             va='center',
             zorder=10)

# Percentage text
x = 10**(-12.3)
y = (ymax - ymin) * 0.05
ax1.text(x, 
         y, 
         f'{probability_alone_MW:.0f}%', 
         fontsize=12, 
         color=custom_dark_gray, 
         ha='center',
         va='bottom',
         zorder=10)

#---------------------------------------
# Alone in the observable Universe hatch filling and line
points = np.logspace(np.log10(min(Ns)), np.log10(5*10**-13), 1000)

ax1.fill_between(points, 
                 interpolation(np.log10(points)), 
                 hatch='/', 
                 edgecolor=(0,0,0,0.3), 
                 fc=(0,1,0,0.1), 
                 linewidth=0.0,
                 zorder=2)
ax1.vlines(x=5*10**-13, 
           color="red", 
           ymin=0, 
           ymax=interpolation(-12.3), 
           linestyle="dashed",
           linewidth=1.5, 
           zorder=4)

# Text with info of probability of being alone in the observable Universe
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_OU = ecdft(5*10**-13)*100
formatted_probability = f"{probability_alone_OU:.0f}%"
text = "Probability of being alone\nin the observable Universe\n($N < 5 \\times 10^{{-13}}$): " + formatted_probability

# Text
x = 10**((np.log10(xmin) + (-12.3)) * 0.6)
y = max((ymax * 0.2), min((interpolation(-12.3) * 0.5), (ymax * 0.8)))
ax1.annotate(text, 
             xy=(x, y),
             xytext=(0, 0), 
             textcoords='offset points',
             fontsize=8,
             color=custom_dark_gray, 
             ha='center',
             va='center',
             zorder=10)

# Percentage text
x = 10**((-12.3 + np.log10(xmin)) * 0.4)
y = (ymax - ymin) * 0.05
ax1.text(x, 
         y, 
         f'{probability_alone_OU:.0f}%', 
         fontsize=12, 
         color=custom_dark_gray,
         ha='left',
         va='bottom',
         zorder=10)

#-------------------------------------------------------------------------------------------
# N CDF
log_space_bins = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 1000)

ax2.ecdf(Ns, 
         color="red", 
         linewidth=2,
         label="Plot N ecdf", 
         zorder=7)

# Limits of the plot, to know where to put the text
#xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()

# Design
ax2.set_xlabel("$N$", fontsize=12, color=custom_dark_gray)
ax2.set_ylabel("Cumulative probability", fontsize=10, color=custom_dark_gray)
ax2.set_title("CDF of the number of civilizations in our Galaxy", fontsize=12, pad=10, color=custom_dark_gray)
#ax2.grid(True, linestyle=":", linewidth="1")

# Axes
ax2.set_xscale("log")

ax2.minorticks_on()
ax2.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
ax2.tick_params(which="minor", direction="in", length=0)
ax2.tick_params(which="both", bottom=True, top=False, left=True, right=False)
ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_color(custom_dark_gray)
ax2.spines['bottom'].set_color(custom_dark_gray)
ax2.tick_params(axis='both', colors=custom_dark_gray)

# Color filling
ax2.fill_between(log_space_bins, ecdft(log_space_bins), color="red", alpha=0.1)

#---------------------------------------
# Alone in the Galaxy line
ax2.vlines(x=1, 
           color="green", 
           ymin=0, 
           ymax=ecdft(1), 
           linestyle="dashed",
           linewidth=1.5, 
           zorder=5)

# Text with info of probability of being alone in the galaxy
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_MW = ecdft(1)*100
formatted_probability = f"{probability_alone_MW:.0f}%"
text = "Probability of being alone\nin the Milky Way galaxy:\n($N < 1$): " + formatted_probability

x = 1
y = min(max(ecdft(1), (ymax * 0.2)), (ymax * 0.9))
ax2.annotate(text, 
             xy=(x, y),
             xytext=(-30, 0), 
             textcoords='offset points',
             fontsize=8,
             color=custom_dark_gray, 
             ha='right',
             va='top',
             zorder=10,
             arrowprops=dict(arrowstyle="-", color=custom_dark_gray, linewidth=0.5))

#---------------------------------------
# Alone in the observable Universe line
ax2.vlines(x=5*10**-13, 
           color="red", 
           ymin=0, 
           ymax=ecdft(5*10**-13), 
           linestyle="dashed",
           linewidth=1.5, 
           zorder=5)

# Text with info of probability of being alone in the observable Universe
# {:.0f} formats the number to have 0 places after the decimal, effectively making it an integer
probability_alone_OU = ecdft(5*10**-13)*100
formatted_probability = f"{probability_alone_OU:.0f}%"
text = "Probability of being alone in\nthe observable Universe:\n($N < 5 \\times 10^{{-13}}$): " + formatted_probability

x = 10**(-12.3)
#y = max((ymax * 0.2), min((ecdft(-12.3) * 0.5), (ymax * 0.8)))
y = min(max(ecdft(-12.3), (ymax * 0.2)), (ymax * 0.9))
ax2.annotate(text, 
             xy=(x, y),
             xytext=(-30, 0),
             textcoords='offset points',
             fontsize=8,
             color=custom_dark_gray, 
             ha='right',
             va='bottom',
             zorder=10,
             arrowprops=dict(arrowstyle="-", color=custom_dark_gray, linewidth=0.5))

#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

plt.savefig("Drake by Metaculus PDF and CDF.png", bbox_inches="tight")

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
print("All done!")

#-------------------------------------------
# Call plt.show() to make the graphics appear.
plt.show()