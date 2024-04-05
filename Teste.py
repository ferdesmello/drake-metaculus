import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import gaussian_kde
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.signal import fftconvolve

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
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p1["x_log"]

# Adjust the PDF by the Jacobian
df_p1["PDF_adjusted"] = df_p1["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p1["PDF_adjusted"], df_p1["x_log"])
#total_area = np.trapz(df_p1["PDF_adjusted"], df_p1["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p1['pdf_values'] = df_p1["PDF_adjusted"] / total_area

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

# Calculate the Jacobian for each x-value
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p2["x_log"]

# Adjust the PDF by the Jacobian
df_p2["PDF_adjusted"] = df_p2["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p2["PDF_adjusted"], df_p2["x_log"])
#total_area = np.trapz(df_p2["PDF_adjusted"], df_p2["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p2['pdf_values'] = df_p2["PDF_adjusted"] / total_area

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

# Calculate the Jacobian for each x-value
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p3["x_log"]

# Adjust the PDF by the Jacobian
df_p3["PDF_adjusted"] = df_p3["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p3["PDF_adjusted"], df_p3["x_log"])
#total_area = np.trapz(df_p3["PDF_adjusted"], df_p3["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p3['pdf_values'] = df_p3["PDF_adjusted"] / total_area

print("p3 done!")

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

# Calculate the Jacobian for each x-value
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p4["x_log"]

# Adjust the PDF by the Jacobian
df_p4["PDF_adjusted"] = df_p4["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p4["PDF_adjusted"], df_p4["x_log"])
#total_area = np.trapz(df_p4["PDF_adjusted"], df_p4["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p4['pdf_values'] = df_p4["PDF_adjusted"] / total_area

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

# Calculate the Jacobian for each x-value
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p5["x_log"]

# Adjust the PDF by the Jacobian
df_p5["PDF_adjusted"] = df_p5["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p5["PDF_adjusted"], df_p5["x_log"])
#total_area = np.trapz(df_p5["PDF_adjusted"], df_p5["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p5['pdf_values'] = df_p5["PDF_adjusted"] / total_area

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

# Calculate the Jacobian for each x-value
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p6["x_log"]

# Adjust the PDF by the Jacobian
df_p6["PDF_adjusted"] = df_p6["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p6["PDF_adjusted"], df_p6["x_log"])
#total_area = np.trapz(df_p6["PDF_adjusted"], df_p6["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p6['pdf_values'] = df_p6["PDF_adjusted"] / total_area

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

# Calculate the Jacobian for each x-value
jacobian = np.log(10) * (p1_xmax - p1_xmin) * df_p7["x_log"]

# Adjust the PDF by the Jacobian
df_p7["PDF_adjusted"] = df_p7["PDF"]# * jacobian

# Calculate the area under the curve
total_area = simps(df_p7["PDF_adjusted"], df_p7["x_log"])
#total_area = np.trapz(df_p7["PDF_adjusted"], df_p7["x_log"])

# Normalize the PDF values so that the sum equals 1
df_p7['pdf_values'] = df_p7["PDF_adjusted"] / total_area

print("p7 done!")

#-------------------------------------------
# Getting together

# Multiplying the factors

# ------------------------
# Calculate the distance PDF using convolution
pdf_12 = fftconvolve(df_p1['pdf_values'], df_p2['pdf_values'], mode='full')
# The distance range is a result of the convolution process
x_12 = np.linspace(df_p1['x_log'][0] * df_p2['x_log'][0], df_p1['x_log'].iloc[-1] * df_p2['x_log'].iloc[-1], len(pdf_12))
# Normalize the distance PDF
total_area = simps(pdf_12, x_12)
# Normalize the PDF values so that the sum equals 1
pdf_12 = pdf_12 / total_area

# ------------------------

# Calculate the distance PDF using convolution
pdf_123 = fftconvolve(pdf_12, df_p3['pdf_values'], mode='full')
# The distance range is a result of the convolution process
x_123 = np.linspace(x_12[0] * df_p3['x_log'][0], x_12[-1] * df_p3['x_log'].iloc[-1], len(pdf_123))
# Normalize the distance PDF
total_area = simps(pdf_123, x_123)
# Normalize the PDF values so that the sum equals 1
pdf_123 = pdf_123 / total_area

# ------------------------

# Calculate the distance PDF using convolution
pdf_1234 = fftconvolve(pdf_123, df_p4['pdf_values'], mode='full')
# The distance range is a result of the convolution process
x_1234 = np.linspace(x_123[0] * df_p4['x_log'][0], x_123[-1] * df_p4['x_log'].iloc[-1], len(pdf_1234))
# Normalize the distance PDF
total_area = simps(pdf_1234, x_1234)
# Normalize the PDF values so that the sum equals 1
pdf_1234 = pdf_1234 / total_area

# ------------------------

# Calculate the distance PDF using convolution
pdf_12345 = fftconvolve(pdf_1234, df_p5['pdf_values'], mode='full')
# The distance range is a result of the convolution process
x_12345 = np.linspace(x_1234[0] * df_p5['x_log'][0], x_1234[-1] * df_p5['x_log'].iloc[-1], len(pdf_12345))
# Normalize the distance PDF
total_area = simps(pdf_12345, x_12345)
# Normalize the PDF values so that the sum equals 1
pdf_12345 = pdf_12345 / total_area

# ------------------------

# Calculate the distance PDF using convolution
pdf_123456 = fftconvolve(pdf_12345, df_p6['pdf_values'], mode='full')
# The distance range is a result of the convolution process
x_123456 = np.linspace(x_12345[0] * df_p6['x_log'][0], x_12345[-1] * df_p6['x_log'].iloc[-1], len(pdf_123456))
# Normalize the distance PDF
total_area = simps(pdf_123456, x_123456)
# Normalize the PDF values so that the sum equals 1
pdf_123456 = pdf_123456 / total_area

# ------------------------

# Calculate the distance PDF using convolution
pdf_1234567 = fftconvolve(pdf_12345, df_p7['pdf_values'], mode='full')
# The distance range is a result of the convolution process
x_1234567 = np.linspace(x_123456[0] * df_p7['x_log'][0], x_123456[-1] * df_p7['x_log'].iloc[-1], len(pdf_1234567))
# Normalize the distance PDF
total_area = simps(pdf_1234567, x_1234567)
# Normalize the PDF values so that the sum equals 1
pdf_1234567 = pdf_1234567 / total_area









# ------------------------

print("Sampling")

p1_sampled = df_p1['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)
p2_sampled = df_p2['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)
p3_sampled = df_p3['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)
p4_sampled = df_p4['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)
p5_sampled = df_p5['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)
p6_sampled = df_p6['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)
p7_sampled = df_p7['pdf_values'].sample(n = 200, replace = True).reset_index(drop=True)

print("Multiplying.")

p_samples = (p1_sampled * 
             p2_sampled * 
             p3_sampled * 
             p4_sampled * 
             p5_sampled * 
             p6_sampled * 
             p7_sampled)



print(p_samples)
print(len(p_samples))
print(p_samples.hasnans)

p_samples = p_samples.dropna()

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
ax1.set_title("PDF Metaculus", fontsize = 16, pad = 20)

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
#plt.xscale("log")

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "inout", length = 7)
ax2.tick_params(which = "minor", direction = "in", length = 2)
ax2.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

plt.savefig("Fermi Metaculus.png", bbox_inches = "tight")

#-------------------------------------------
fig3, ax3 = plt.subplots(figsize = (10, 6))

ax3.plot(df_p7['x_log'], df_p7['pdf_values'],
         color = "black",
         linewidth = 2,
         label = "teste")

# Design
ax3.set_xlabel("X", fontsize = 12)
ax3.set_ylabel("probability", fontsize = 12)
ax3.set_title("Teste Metaculus", fontsize = 16, pad = 20)

ax3.grid(True, linestyle = ":", linewidth = "1")

# Axes
#ax3.set_xlim(45, 200)
#ax3.set_ylim(0, 300)
plt.xscale("log")

ax3.minorticks_on()
ax3.tick_params(which = "major", direction = "inout", length = 7)
ax3.tick_params(which = "minor", direction = "in", length = 2)
ax3.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax3.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

plt.savefig("Fermi Metaculus2.png", bbox_inches = "tight")

#-------------------------------------------
fig4, ax4 = plt.subplots(figsize = (10, 6))

ax4.plot(x_12, pdf_12,
         color = "black",
         linewidth = 2,
         label = "teste")

# Design
ax4.set_xlabel("N", fontsize = 12)
ax4.set_ylabel("probability", fontsize = 12)
ax4.set_title("Teste Metaculus Convolution", fontsize = 16, pad = 20)

ax4.grid(True, linestyle = ":", linewidth = "1")

# Axes
#ax4.set_xlim(10**11, 10**12)
#ax4.set_ylim(0, 300)
plt.xscale("log")

ax4.minorticks_on()
ax4.tick_params(which = "major", direction = "inout", length = 7)
ax4.tick_params(which = "minor", direction = "in", length = 2)
ax4.tick_params(which = "both", bottom = True, top = True, left=True, right = True)
ax4.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False)

plt.savefig("Fermi Metaculus Convolution.png", bbox_inches = "tight")


print("All done!")

#-------------------------------------------
# Calling plt.show() to make graphics appear.
plt.show()