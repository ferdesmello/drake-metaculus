# Drake-Metaculus
This code calculate the [Drake equation](https://en.wikipedia.org/wiki/Drake_equation) using probability functions, not point estimates. this is done using prediction data from [Metaculus](https://www.metaculus.com/home/).

## The Drake equation:

{\displaystyle N=R_{*}\cdot f_{\mathrm {p} }\cdot n_{\mathrm {e} }\cdot f_{\mathrm {l} }\cdot f_{\mathrm {i} }\cdot f_{\mathrm {c} }\cdot L}
where

N = the number of civilizations in the Milky Way galaxy with which communication might be possible (i.e. which are on the current past light cone);

and

R∗ = the average rate of star formation in our Galaxy.
fp = the fraction of those stars that have planets.
ne = the average number of planets that can potentially support life per star that has planets.
fl = the fraction of planets that could support life that actually develop life at some point.
fi = the fraction of planets with life that go on to develop intelligent life (civilizations).
fc = the fraction of civilizations that develop a technology that releases detectable signs of their existence into space.
L = the length of time for which such civilizations release detectable signals into space.

## The code

Run the .py file. It will download API data from Metaculus's questions about the 7 factors of the Drake equation. The most recent prediction data - the PDFs and CDFs for every factor - will be parsed and used to calculate N, the PDF of the number of detectable alien civilizations in our Galaxy. And the result will be presented in some figures.

## The questions

See the links bellow:

[Fermi Paradox Series on Metaculus](https://www.metaculus.com/project/2994/)

[1st Parameter R∗](https://www.metaculus.com/questions/1337/drakes-equation-1st-parameter-r/)

[2nd parameter f_p](https://www.metaculus.com/questions/1338/drakes-equation-2nd-parameter-f_p/)

[3rd parameter n_e](https://www.metaculus.com/questions/1339/drakes-equation-3rd-parameter-n_e/)

[4th parameter f_l](https://www.metaculus.com/questions/1340/drakes-equation-4th-parameter-f_l/)

[5th Parameter f_i](https://www.metaculus.com/questions/1341/drakes-equation-fifth-parameter-f_i/)

[6th parameter f_c](https://www.metaculus.com/questions/1342/drakes-equation-6th-parameter-f_c/)

[7th parameter L](https://www.metaculus.com/questions/1343/drakes-equation-7th-parameter-l/)