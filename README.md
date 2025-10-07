# Hidden flux partitioning in the global water cycle

## Introduction

This repository contains code used for the analyses and for making the figures in the corresponding publication.
The final figures in the paper were created using image processing software.

## Overview

* *plot\_fluxes\_vs\_aridity\_observations.py* creates Figure 2
* *plot\_fluxes\_vs\_aridity\_toy\_model.py* creastes Figure 3
* *resample\_rasters.py* is used to resample the raster files
* the folder *functions* contains a range of helper functions

## Data sources

Groundwater recharge observations from MacDonald et al. (2021) are available from https://www2.bgs.ac.uk/nationalgeosciencedatacentre/citedData/catalogue/45d2b71c-d413-44d4-8b4b-6190527912ff.html.

Groundwater recharge data from Moeck et al. (2020) are available from https://opendata.eawag.ch/dataset/globalscale\_groundwater\_moeck.

Groundwater recharge data from Lee et al. (2024) are available from https://www.hydroshare.org/resource/5e7b8bfcc1514680902f8ff43cc254b8/

CHELSA data are available from https://chelsa-climate.org/downloads/ (Brun et al., 2022; Karger et al., 2021).

Baseflow data are taken from Gnann et al. (2019) and can be found in the repository here.

## References

Moeck et al. 2020. A Global-Scale Dataset of Direct Natural Groundwater Recharge Rates: A Review of Variables, Processes and Relationships. Science of The Total Environment 717 (May):137042. https://doi.org/10.1016/j.scitotenv.2020.137042

MacDonald et al. 2021. Mapping Groundwater Recharge in Africa from Ground Observations and Implications for Water Security. Environmental Research Letters 16 (3): 034012. https://doi.org/10.1088/1748-9326/abd661

Lee et al. 2024. A high-resolution map of diffuse groundwater recharge rates for Australia." Hydrol. Earth Syst. Sci., 28, 1771–1790. https://doi.org/10.5194/hess-28-1771-2024

Brun et al. (2022b). Global climate-related predictors at kilometer resolution for the past and future. Earth System Science Data, 14(12), 5573–5603. https://doi.org/10.5194/essd-14-5573-2022

Karger et al. (2017). Climatologies at high resolution for the Earth’s land surface areas. Scientific Data, 4(1), 170122. https://doi.org/10.1038/sdata.2017.122

Gnann et al. 2019. Is there a baseflow Budyko curve?. Water Resources Research, 55, 2838–2855. https://doi.org/10.1029/2018WR024464

