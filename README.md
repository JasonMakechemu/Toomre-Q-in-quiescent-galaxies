# ToomreQ

This repository contains the code I used to obtain results for my paper (in prep) titled 'WISDOM Project - XXVII. Evaluating the Role of the Toomre 𝑄
Parameter in the Stability of the Cold Gaseous Discs of Early-type Galaxies'.

Many elliptical galaxies are rich in cold molecular gas, yet quiescent, quiescent to the extent that they are referred to as 'red and dead' galaxies . This is an intriguing phenomenon because galaxies rich in cold molecular gas are in priciple the ideal environment for star formation to occur. 

We present a study of the large-scale cold molecular gas disc dynamics and stability of seven early-type galaxies (ETGs)
via the Toomre 𝑄 (𝑄_T) parameter. We use high angular resolution CO line data obtained with the Atacama Large Millime-
ter/submillimeter Array (ALMA) as part of the mm-Wave Interferometric Survey of Dark Object Masses (WISDOM) project.
Using low-𝐽 CO transitions, and the modelling of the stellar mass distributions with multi-Gaussian expansion models, we
measure 𝑄_T across the inner ∼ 102 − 103 pc in radius of these galaxies at the spatial resolutions of the data (20 – 70 pc). We
thus characterise 𝑄_T across many thousands of individual sight lines (≈ 50, 000 at our highest physical resolution). The Toomre
𝑄 parameter is ≈ 10 times larger in the gaseous disc of each of our ETGs than in previously measured late-type galaxies.
Additionally, when averaged over the whole discs, 𝑄_T is most strongly correlated with the velocity dispersion, quantified here by
the effective width of each line, thus indicating as expected that gaseous discs with higher velocity dispersions also have higher
𝑄_T. This implies that gas turbulence plays the largest role in maintaining overall stability, by increasing the effective support
against disc self-gravity. Our results are consistent with highly stable cold molecular gas discs. The large-scale dynamics of the
discs is thus key to regulate the star formation (or lack thereof) of our systems. These findings provide an intuitive explanation
for the high molecular gas mass surface densities, yet low star-formation rates, of ETGs


The Python files within this repository are used for astronomical data analysis and visualization related to the galaxy NGC 0524, but apply to each of the seven NGC galaxies analysed in the paper. The code performs calculations to determine the stability of the galaxy's gas and stellar disks using the Toomre Q parameter.


**Functions_For_Q_NGC0524.py** - In this file, key parameters are calculated and plotted for determining the stability of the galaxy's gas and stellar disks, specifically using the Toomre Q parameter. The script contains functions for opening and processing FITS astronomical data files, generating masks to filter out noise, and creating spectral cubes. A major part of the code is dedicated to creating and plotting various moment maps (moment 0, 1, and 2) to visualize the integrated intensity, velocity, and velocity dispersion of the CO gas. Additionally, it includes functions for converting coordinates, calculating the deprojected galactocentric radius, and determining the circular velocity using a Multi-Gaussian Expansion (MGE) model, which we then use to derive the epicyclic frequency (kappa), a critical component for the disk stability analysis. Additionally we model the stellar surface density and stellar velocity dispersion. With this, we can calculate all the parameters required to calculate a spaxel-by-spaxel view of Toomre Q in the gaseous disks in our sample of galaxies.


**Call_Functions_for_Q_NGC0524.py** -  This is the main script that uses the functions from Functions_For_Q_NGC0524.py. It sets up the necessary variables and calls the functions to perform the analysis. It serves as an example of how to run the code.

The file contains variables for:

File names: Specifies the FITS files for the galaxy data and moment maps.

Galaxy properties: Defines the celestial coordinates, distance, and inclination of the galaxy NGC 0524, which are used as inputs for the functions.

This script imports all the necessary functions from Functions_For_Q_NGC0524.py and then defines the parameters for the analysis. It is the control script for executing the code.
