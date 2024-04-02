import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# reading csv data from the API

url_p1 = "https://www.metaculus.com/api2/questions/1337/download_csv/"
url_p2 = "https://www.metaculus.com/api2/questions/1338/download_csv/"
url_p3 = "https://www.metaculus.com/api2/questions/1339/download_csv/"
url_p4 = "https://www.metaculus.com/api2/questions/1340/download_csv/"
url_p5 = "https://www.metaculus.com/api2/questions/1341/download_csv/"
url_p6 = "https://www.metaculus.com/api2/questions/1342/download_csv/"
url_p7 = "https://www.metaculus.com/api2/questions/1343/download_csv/"

df_p1 = pd.read_csv(url_p3)

# reducing data to just the PDF distribution

df_p1 = df_p1.iloc[-2, 11:211].reset_index(drop = False)

df_p1.columns = ['x_norm', 'PDF']

df_p1["x_norm"].replace(to_replace = "pdf_at_", value = "", regex = True, inplace = True)

df_p1["x_norm"] = df_p1["x_norm"].astype("float")
df_p1["PDF"] = df_p1["PDF"].astype("float")
df_p1["x"] = df_p1["x_norm"].apply(lambda x : 10**-6 + x*(100 - 10**-6))



print(df_p1["PDF"].max())


print(df_p1.info())
print(df_p1.shape)
print(df_p1.head())
print(df_p1.tail())



#-------------------------------------------
# Figures

fig1, ax1 = plt.subplots(figsize = (10, 6))

ax1.plot(df_p1["x"], df_p1["PDF"],
         color = "red",
         linewidth = 2)

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
ax1.set_xlabel("average number of habitable planets per star", fontsize = 12)
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
# Calling plt.show() to make graphics appear.
plt.show()