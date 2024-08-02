# Drake-Metaculus

This folder contains code that calculates the [Drake equation](https://en.wikipedia.org/wiki/Drake_equation) using probability functions (not point estimates) for the equation factors, obtaining a distribution (not a point estimate) of the possible number of detectable alien civilizations in our Galaxy. This is done using forecasts for the value of the factors on the [Metaculus](https://www.metaculus.com/home/) website.

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

The Drake equation is usually computed using point estimates for every factor. However, we are not certain of the value of many factors, with estimates varying many orders of magnitude. Following Sandberg, Drexler, and Toby Ord's [Dissolving the Fermi Paradox](https://arxiv.org/abs/1806.02404), it seems better to use probability density functions and get a distribution for possible values of $N$.

## The code

Run the .py file (it uses many common libraries: numpy, pandas, matplotlib, scipy, statsmodels). It will download API data from Metaculus's questions about the 7 factors of the Drake equation. The most recent forecast data - the PDFs and CDFs for every factor - will be parsed and used to calculate $N$, the number of detectable alien civilizations in our Galaxy. The result will be presented in some figures. In my machine, it took ~1 minute to finish.

## The End Estimates
<img src="Drake by Metaculus PDF and CDF.png" alt="Distribution of the estimates of the possible number of detectable alien civilizations in our Galaxy by Metaculus." style="height: 789px; width:589px;"/>

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
