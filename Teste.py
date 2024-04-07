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

#--------------------------------
#P1

df_p1 = pd.read_csv(url_p1)

# reducing data to just the PDF distribution

df_p1 = df_p1.iloc[-2, 11:211].reset_index(drop = False)
df_p1.columns = ['x_norm', 'PDF']
df_p1["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p1["x_norm"] = df_p1["x_norm"].astype("float")
df_p1["PDF"] = df_p1["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(0.01)
p1_xmax = np.log10(1000)
#print(p1_xmin, p1_xmax)

df_p1["x_log"] = df_p1["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the Jacobian for each x-value
#jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p1["x_log"]

# Adjust the PDF by the Jacobian
df_p1["PDF_adjusted"] = df_p1["PDF"]# * jacobian

# Calculate the area under the curve
#total_area = simps(df_p1["PDF_adjusted"], df_p1["x_log"])
#total_area = np.trapz(df_p1["PDF_adjusted"], df_p1["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p1['pdf_values'] = df_p1["PDF_adjusted"] / total_area

df_p1['pdf_values'] = df_p1["PDF"] / df_p1["PDF"].sum()
print(df_p1["PDF"].sum())




p1_kde = gaussian_kde(df_p1['pdf_values'])
p1_values = np.linspace(min(df_p1['pdf_values']), max(df_p1['pdf_values']), 10000)
p1_pdf = p1_kde(p1_values)

# Normalize the PDF values so that the sum equals 1
total_area = simps(p1_pdf, p1_values)
p1_pdf = p1_pdf / total_area





print("p1 done!")

#--------------------------------
#P2

df_p2 = pd.read_csv(url_p2)

# reducing data to just the PDF distribution

df_p2 = df_p2.iloc[-2, 11:211].reset_index(drop = False)
df_p2.columns = ['x_norm', 'PDF']
df_p2["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p2["x_norm"] = df_p2["x_norm"].astype("float")
df_p2["PDF"] = df_p2["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(0.01)
p1_xmax = np.log10(1)
#print(p1_xmin, p1_xmax)

df_p2["x_log"] = df_p2["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the area under the curve
#total_area = simps(df_p2["PDF"], df_p2["x_log"])
#total_area = np.trapz(df_p2["PDF"], df_p2["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p2['pdf_values'] = df_p2["PDF"] / total_area

df_p2['pdf_values'] = df_p2["PDF"] / df_p2["PDF"].sum()
print(df_p2["PDF"].sum())
print("p2 done!")

#--------------------------------
#P3

df_p3 = pd.read_csv(url_p3)

# reducing data to just the PDF distribution

df_p3 = df_p3.iloc[-2, 11:211].reset_index(drop = False)
df_p3.columns = ['x_norm', 'PDF']
df_p3["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p3["x_norm"] = df_p3["x_norm"].astype("float")
df_p3["PDF"] = df_p3["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(10**-6)
p1_xmax = np.log10(100)
#print(p1_xmin, p1_xmax)

df_p3["x_log"] = df_p3["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the area under the curve
#total_area = simps(df_p3["PDF"], df_p3["x_log"])
#total_area = np.trapz(df_p3["PDF"], df_p3["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p3['pdf_values'] = df_p3["PDF"] / total_area

df_p3['pdf_values'] = df_p3["PDF"] / df_p3["PDF"].sum()
print(df_p3["PDF"].sum())
print("p3 done!")


# Step 2: Compute the CDF
cdf = pd.Series(np.cumsum(df_p3['pdf_values']))
print(cdf)
#cdf /= max(cdf)  # Normalize

# Step 3: Interpolate the CDF
cdf_interp = interp1d(cdf, df_p3["x_log"], fill_value = 'extrapolate')

# Generate uniform samples for our CDF range and find corresponding x values
num_samples = 10000
uniform_samples = np.random.uniform(0, 1, num_samples)
sampled_x = cdf_interp(uniform_samples)








#--------------------------------
#P4

df_p4 = pd.read_csv(url_p4)

# reducing data to just the PDF distribution

df_p4 = df_p4.iloc[-2, 11:211].reset_index(drop = False)
df_p4.columns = ['x_norm', 'PDF']
df_p4["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p4["x_norm"] = df_p4["x_norm"].astype("float")
df_p4["PDF"] = df_p4["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(10**-31)
p1_xmax = np.log10(1)
#print(p1_xmin, p1_xmax)

df_p4["x_log"] = df_p4["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the area under the curve
#total_area = simps(df_p4["PDF"], df_p4["x_log"])
#total_area = np.trapz(df_p4["PDF"], df_p4["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p4['pdf_values'] = df_p4["PDF"] / total_area

df_p4['pdf_values'] = df_p4["PDF"] / df_p4["PDF"].sum()
print(df_p4["PDF"].sum())
print("p4 done!")

#--------------------------------
#P5

df_p5 = pd.read_csv(url_p5)

# reducing data to just the PDF distribution

df_p5 = df_p5.iloc[-2, 11:211].reset_index(drop = False)
df_p5.columns = ['x_norm', 'PDF']
df_p5["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p5["x_norm"] = df_p5["x_norm"].astype("float")
df_p5["PDF"] = df_p5["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(10**-20)
p1_xmax = np.log10(1)
#print(p1_xmin, p1_xmax)

df_p5["x_log"] = df_p5["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the area under the curve
#total_area = simps(df_p5["PDF"], df_p5["x_log"])
#total_area = np.trapz(df_p5["PDF"], df_p5["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p5['pdf_values'] = df_p5["PDF"] / total_area

df_p5['pdf_values'] = df_p5["PDF"] / df_p5["PDF"].sum()
print(df_p5["PDF"].sum())
print("p5 done!")

#--------------------------------
#P6

df_p6 = pd.read_csv(url_p6)

# reducing data to just the PDF distribution

df_p6 = df_p6.iloc[-2, 11:211].reset_index(drop = False)
df_p6.columns = ['x_norm', 'PDF']
df_p6["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p6["x_norm"] = df_p6["x_norm"].astype("float")
df_p6["PDF"] = df_p6["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(10**-5)
p1_xmax = np.log10(1)
#print(p1_xmin, p1_xmax)

df_p6["x_log"] = df_p6["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the area under the curve
#total_area = simps(df_p6["PDF"], df_p6["x_log"]) + 0.0000001
#total_area = np.trapz(df_p6["PDF"], df_p6["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p6['pdf_values'] = df_p6["PDF"] / total_area

df_p6['pdf_values'] = df_p6["PDF"] / df_p6["PDF"].sum()
print(simps(df_p6['pdf_values'], df_p6["x_log"]))
print("p6 done!")

#--------------------------------
#P7

df_p7 = pd.read_csv(url_p7)

# reducing data to just the PDF distribution

df_p7 = df_p7.iloc[-2, 11:211].reset_index(drop = False)
df_p7.columns = ['x_norm', 'PDF']
df_p7["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)
df_p7["x_norm"] = df_p7["x_norm"].astype("float")
df_p7["PDF"] = df_p7["PDF"].astype("float")

# transforming scales
p1_xmin = np.log10(10)
p1_xmax = np.log10(10**10)
#print(p1_xmin, p1_xmax)

df_p7["x_log"] = df_p7["x_norm"].apply(lambda x : 10**(p1_xmin + x * (p1_xmax - p1_xmin)))

# Calculate the area under the curve
#total_area = simps(df_p7["PDF"], df_p7["x_log"])
#total_area = np.trapz(df_p7["PDF"], df_p7["x_log"])

# Normalize the PDF values so that the sum equals 1
#df_p7['pdf_values'] = df_p7["PDF"] / total_area

df_p7['pdf_values'] = df_p7["PDF"] / df_p7["PDF"].sum()
print(df_p7["PDF"].sum())
print("p7 done!")

#-------------------------------------------
# Multiplying the factors

print("Sampling")

"""p1_sampled = pd.Series(choices(df_p1["x_log"], df_p1['pdf_values'], k = 1000))
p2_sampled = pd.Series(choices(df_p2["x_log"], df_p2['pdf_values'], k = 1000))
p3_sampled = pd.Series(choices(df_p3["x_log"], df_p3['pdf_values'], k = 1000))
p4_sampled = pd.Series(choices(df_p4["x_log"], df_p4['pdf_values'], k = 1000))
p5_sampled = pd.Series(choices(df_p5["x_log"], df_p5['pdf_values'], k = 1000))
p6_sampled = pd.Series(choices(df_p6["x_log"], df_p6['pdf_values'], k = 1000))
p7_sampled = pd.Series(choices(df_p7["x_log"], df_p7['pdf_values'], k = 1000))"""

p1_sampled = pd.Series(np.random.choice(df_p1["x_log"], size = 1000, p = df_p1['pdf_values']))
p2_sampled = pd.Series(np.random.choice(df_p2["x_log"], size = 1000, p = df_p2['pdf_values']))
p3_sampled = pd.Series(np.random.choice(df_p3["x_log"], size = 1000, p = df_p3['pdf_values']))
p4_sampled = pd.Series(np.random.choice(df_p4["x_log"], size = 1000, p = df_p4['pdf_values']))
p5_sampled = pd.Series(np.random.choice(df_p5["x_log"], size = 1000, p = df_p5['pdf_values']))
p6_sampled = pd.Series(np.random.choice(df_p6["x_log"], size = 1000, p = df_p6['pdf_values']))
p7_sampled = pd.Series(np.random.choice(df_p7["x_log"], size = 1000, p = df_p7['pdf_values']))

"""p1_sampled = df_p1['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)
p2_sampled = df_p2['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)
p3_sampled = df_p3['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)
p4_sampled = df_p4['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)
p5_sampled = df_p5['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)
p6_sampled = df_p6['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)
p7_sampled = df_p7['pdf_values'].sample(n = 500, replace = True).reset_index(drop = True)"""

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

"""print(p_samples)
print(len(p_samples))
print(p_samples.hasnans)

p_samples = p_samples.dropna()"""

print("Monte Carling.")

# Estimate the PDF of x using KDE
p_kde = gaussian_kde(p_samples)
p_values = np.linspace(min(p_samples), max(p_samples), 10000)
p_pdf = p_kde(p_values)

# Normalize the PDF values so that the sum equals 1
total_area = simps(p_pdf, p_values)
p_pdf = p_pdf / total_area

print("Together done!")

#-------------------------------------------

print("Making Figures.")

# Figures

fig1, ax1 = plt.subplots(figsize = (10, 6))

ax1.plot(df_p1["x_norm"], df_p1["PDF"],
         color = "red",
         linewidth = 2,
         label = "p1")

ax1.plot(df_p2["x_norm"], df_p2["PDF"],
         color = "blue",
         linewidth = 2,
         label = "p2")

ax1.plot(df_p3["x_norm"], df_p3["PDF"],
         color = "green",
         linewidth = 2,
         label = "p3")

ax1.plot(df_p4["x_norm"], df_p4["PDF"],
         color = "magenta",
         linewidth = 2,
         label = "p4")

ax1.plot(df_p5["x_norm"], df_p5["PDF"],
         color = "cyan",
         linewidth = 2,
         label = "p5")

ax1.plot(df_p6["x_norm"], df_p6["PDF"],
         color = "orange",
         linewidth = 2,
         label = "p6")

ax1.plot(df_p7["x_norm"], df_p7["PDF"],
         color = "brown",
         linewidth = 2,
         label = "p7")


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

plt.legend()

# Design
ax1.set_xlabel("X", fontsize = 12)
ax1.set_ylabel("probability", fontsize = 12)
ax1.set_title("PDF's factors Metaculus", fontsize = 16, pad = 20)

ax1.grid(True, linestyle = ":", linewidth = "1")

# Axes
#ax1.set_xlim(45, 200)
#ax1.set_ylim(0, 300)
#plt.xscale("log")

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "inout", length = 7)
ax1.tick_params(which = "minor", direction = "in", length = 2)
ax1.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

plt.savefig("pdf Metaculus.png", bbox_inches = "tight")

#-------------------------------------------
fig2, ax2 = plt.subplots(figsize = (10, 6))

ax2.plot(p_values, p_pdf,
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

plt.savefig("Fermi Metaculus.png", bbox_inches = "tight")

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
ax1.plot(df_p1['x_log'], df_p1['pdf_values'],
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
ax2.plot(df_p2['x_log'], df_p2['pdf_values'],
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
ax3.plot(df_p3['x_log'], df_p3['pdf_values'],
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
ax4.plot(df_p4['x_log'], df_p4['pdf_values'],
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
ax5.plot(df_p5['x_log'], df_p5['pdf_values'],
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
ax6.plot(df_p6['x_log'], df_p6['pdf_values'],
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
ax7.plot(df_p7['x_log'], df_p7['pdf_values'],
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
print(logbins)

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
ax8.set_xlabel("average rate of formation of suitable stars (stars/year)", fontsize = 8)
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
#Histo P1

logbins = np.logspace(np.log10(min(p1_sampled)), np.log10(max(p1_sampled)), 50)
print(logbins)
#p1_pdf, p1_values
ax9.hist(sampled_x,#p1_pdf,#p1_sampled, 
         #range = (45, 200), 
         histtype = "step", 
         fill = True, 
         density = True, 
         #alpha = 0.2, 
         bins = logbins,
         color = (0.0, 0.0, 1.0, 0.1), 
         edgecolor = "blue", 
         linewidth = 2,
         label = "Histo P1")

# Design
ax9.set_xlabel("average rate of formation of suitable stars (stars/year)", fontsize = 8)
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

plt.savefig("Fermi Metaculus Parameters.png", bbox_inches = "tight")

print("All done!")

#-------------------------------------------
# Calling plt.show() to make graphics appear.
plt.show()