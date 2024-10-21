# Drake-Metaculus

## Overview

This folder contains code that calculates the [Drake equation](https://en.wikipedia.org/wiki/Drake_equation) using distributions of estimates (not one-point estimates) for the equation parameters, obtaining a distribution (not a point estimate) of the possible number of detectable alien civilizations in our Galaxy. That is done using forecast data from the [Metaculus](https://www.metaculus.com/home/) website.

If you want to know more about all of this and about the process of doing it all, take a look at [this post](https://fdesmello.wordpress.com/2024/10/21/the-drake-equation-metaculus-and-alien-civilizations/).

## Running instructions

Run the Drake_Metaculus.py file (it uses many common libraries: [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [scipy](https://scipy.org/), and [statsmodels](https://www.statsmodels.org/stable/index.html)). It will download data from Metaculus's questions about the 7 parameters of the Drake equation. Using the API, the most recent forecast data - the PDFs and CDFs for each parameter - will be parsed and used to calculate $N$, the number of detectable alien civilizations in our Galaxy. The result will be presented in some figures. In my machine, it took ~1 minute to finish.

Metaculus has changed the format of the API output twice since I finished most of the code, which broke the parsing and I had to change the code twice in response. This may happen again, so I can't guarantee that the code will always work. But I will try to keep it working.

## The end estimates
At the end, you will have three figures, the last and most important one is shown below.

<img src="Drake by Metaculus PDF and CDF.png" alt="Distribution of the estimates of the possible number of detectable alien civilizations in our Galaxy by Metaculus." style="height: 789px; width:589px;"/>

## The Drake equation

$N = R_{*}\cdot f_{p}\cdot n_{e}\cdot f_{l}\cdot f_{i}\cdot f_{c}\cdot L$

where

$N$ = the number of civilizations in the Milky Way galaxy with which communication might be possible (i.e. which are on the current past light cone);

and

$R_{∗}$ = the average rate of star formation in our Galaxy.

$f_{p}$ = the fraction of those stars that have planets.

$n_{e}$ = the average number of planets that can potentially support life per star that has planets.

$f_{l}$ = the fraction of planets that could support life that actually develop life at some point.

$f_{i}$ = the fraction of planets with life that go on to develop intelligent life (civilizations).

$f_{c}$ = the fraction of civilizations that develop a technology that releases detectable signs of their existence into space.

$L$ = the length of time for which such civilizations release detectable signals into space.

The Drake equation is usually computed using point estimates for every parameter. However, we are not certain of the value of many parameters, with estimates varying many orders of magnitude. Following Sandberg, Drexler, and Toby Ord's [Dissolving the Fermi Paradox](https://arxiv.org/abs/1806.02404), it seems better to use probability density functions and get a distribution for the possible values of $N$.

## The questions on Metaculus

See the links below:

[Fermi Paradox Series on Metaculus](https://www.metaculus.com/project/2994/)

[1st Parameter R∗](https://www.metaculus.com/questions/1337/drakes-equation-1st-parameter-r/)

[2nd parameter f_p](https://www.metaculus.com/questions/1338/drakes-equation-2nd-parameter-f_p/)

[3rd parameter n_e](https://www.metaculus.com/questions/1339/drakes-equation-3rd-parameter-n_e/)

[4th parameter f_l](https://www.metaculus.com/questions/1340/drakes-equation-4th-parameter-f_l/)

[5th Parameter f_i](https://www.metaculus.com/questions/1341/drakes-equation-fifth-parameter-f_i/)

[6th parameter f_c](https://www.metaculus.com/questions/1342/drakes-equation-6th-parameter-f_c/)

[7th parameter L](https://www.metaculus.com/questions/1343/drakes-equation-7th-parameter-l/)
