#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:54:41 2023

@author: husmak
"""



import numpy as np
import matplotlib.pyplot as plt
from jampy.jam_axi_proj import jam_axi_proj

from astropy.io import fits
from astropy import units as u
from radio_beam import Beam
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
from uncertainties import umath

import uncertainties

'''
import necessary function for moment maps
'''
from Functions_For_Q_NGC0524 import open_fits_file, generate_cubes, galaxy_coordinates, moment_0_image, moment_1_image, moment_2_image, linewidth_image, error_for_moment_0, error_for_moment_1, error_for_moment_2


'''
import necessary functions for arcsec to rad conversion
'''

from Functions_For_Q_NGC0524 import arcsec_to_rad


'''
import necessary functions for galactocentric radius
 
'''
from Functions_For_Q_NGC0524 import meshgrid_galactocentric_radius, values_for_galactocentric_radius, get_gcradius


'''
import necessary functions for beta from circular velocity and necessary derivatives.
'''
from Functions_For_Q_NGC0524 import mge_vcirc_example

'''
import necessary functions for kappa map generation
'''

from Functions_For_Q_NGC0524 import get_kappa_values, plot_kappa

'''
import necessary functions for Q_gas map generation
'''

from Functions_For_Q_NGC0524 import plot_Q

'''
import necessary functions for stellar surface density
'''

from Functions_For_Q_NGC0524 import plot_stellar_surface_density

'''
import necessary functions for stellar velocity dispersion
'''

from Functions_For_Q_NGC0524 import values_needed_for_vrms_plot, plot_for_vrms


'''
import necessary functions for stellar velocity dispersion map
'''

from Functions_For_Q_NGC0524 import get_vrms_values, plot_vrms


'''
import necessary functions for Q_star parameter
'''

from Functions_For_Q_NGC0524 import plot_Q_star


'''
import necessary functions for Q_total parameter
'''
from Functions_For_Q_NGC0524 import plot_Q_total

from Functions_For_Q_NGC0524 import Q_total_error_map

'''
import necessary functions for Q_total parameter
'''
from Functions_For_Q_NGC0524 import generate_q_and_r_arrays, q_as_func_of_r

from Functions_For_Q_NGC0524 import variance_from_median


'''
import funtion to plot each Q parameter individually
'''

from Functions_For_Q_NGC0524 import components_of_both_Qs


from Functions_For_Q_NGC0524 import plotderivedproducts


'''
Getting errors for ktau measurements
'''
from Functions_For_Q_NGC0524 import perturbedktau

'''
key variables
'''


'''
https://www.canfar.net/storage/vault/list/AstroDataCitationDOI/CISTI.CANFAR/23.0016/data
where I take data cubes from
'''

filename = 'ngc0524_12m+7m_co21_2p5kms.fits' #filename of galaxy

mom_0_filename = 'ngc0524_12m+7m_co21_2p5kms_strict_mom0.fits'
mom_1_filename = 'ngc0524_12m+7m_co21_2p5kms_strict_mom1.fits'
mom_2_filename = 'ngc0524_12m+7m_co21_2p5kms_strict_mom2.fits'
ew_filename = 'ngc0524_12m+7m_co21_2p5kms_strict_ew.fits'






center_ra = '01h24m47.74s' #center ra of galaxy
center_dec = '+9d32m20.12s' #center dec of galaxy
size_of_cutout = (181, 181) #size of image cutout (i.e. how much image is zoomed in)


'''
WISDOM project – IV. A molecular gas dynamical measurement of the supermassive black hole mass in NGC 524
Mark D. Smith et al. 2019
'''


distance_to_galaxy = 23.3 #distance to galaxy in Mpc
inclination_of_galaxy = 23.7 #inclination of galaxy in degrees (actual value of 20.6 is too low for code to work "raise ValueError('Inclination too low for deprojection')" on jampy), 25 is the upper bound they have, but chose 23.7 since it'll reach the upper bound given the error i incorporated for it
arcsec_to_radian = (1/3600) * (np.pi / 180) #1 arcsec to rad

cos_inclination = np.cos(np.radians(inclination_of_galaxy)) #cos of the inclination of the galaxy


position_angle = np.radians(39.6) #position angle of galaxy in radians
gravitational_constant = 0.004301  # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
galaxy_black_hole_mass = 4.0e8 
radius_range = np.geomspace(0.65, 21, 100) # Radii in arscec where Vcirc has to be computed. 0.05 arcsec to 3 arcsec in 40 steps


pi = 3.1415


'''
WISDOM project – IV. A molecular gas dynamical measurement of the supermassive black hole mass in NGC 524
Mark D. Smith et al. 2019
'''
CO_conversion_factor = (4.35/0.7)

black_hole_soi = 73 #black hole sphere of influence in pc

co_disk_radius = 94.6 #radius of co disk in pc
#%%


'''
WISDOM project – IV. A molecular gas dynamical measurement of the supermassive black hole mass in NGC 524
Mark D. Smith et al. 2019
'''

#mlinner = unumpy.uarray([2.78], [0.21])
#mlouter = unumpy.uarray([2.36], [0.12])
#mltotal = ((mlinner + mlouter)/2) #mean of th inner and outer mass to light ratios

#print(unumpy.nominal_values(np.array(mltotal))) #value for ml ratio
#print(unumpy.std_devs(np.array(mltotal))) #error for ml ratio


mass_to_light_ratio = 5.7 # ml ratio

surf, sigma, qobs = np.loadtxt('values_0524.txt').T

mass_to_light_ratio_error = 0.285 #estimated 1-sigma error (10%)
galaxy_inclination_error = 1.3 # 3-sigma error is 6.5  but deprojection issue so i use estimated 1-sigma error (5%)
black_hole_mass_error = 0.4e8 # estimated 1-sigma error (10%)


'''
Surface luminosity, dispersion, and axial ratio are taken from MGE fitting
for the repsective galaxy.
'''

surface_luminosity = [21677.04, 6412.10, 26977.40, 8203.52, 4345.10, 3388.44, 1183.04, 281.84, 67.92, 19.95]
dispersion = [0.017, 0.0632, 0.2986, 0.9183, 2.1232, 4.2560, 12.0781, 29.8538, 51.0505, 135.5190]
axial_ratio = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]









'''
initialises data and runs moment maps 
'''
hdu, data, header = open_fits_file(filename) #open fits file

cube, moment_0, moment_1, moment_2 = generate_cubes(filename) #generates cubes and moment_0 

position, x_cen, y_cen = galaxy_coordinates(moment_0, center_ra, center_dec) # gets galaxy coordinates

#%%

moment_0_data = moment_0_image(position, size_of_cutout, moment_0) #makes moment 0 map


#%%


moment_1_data, mean_velocity = moment_1_image(position, size_of_cutout, cube) # makes moment 1 map and removes systemic velocity

moment_2_data = moment_2_image(position, size_of_cutout, cube) # makes moment 2 map, i.e velocity dispersion


#%%
sigma_map_data = linewidth_image(position, size_of_cutout, cube) # makes moment 2 map in km/s, i.e velocity dispersion in km/s, i.e. linewidth

#%%



'''
Gets velocities along los of data cube? and moment map errors 
'''

error = error_for_moment_0() #erroy for integrated intensity of the gas
total_error = error_for_moment_1() #error for intensity weighted velocity
moment_2_plot = error_for_moment_2() #error for effective width


'''
gets errors for my moment maps
'''


#error for moment 0
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

im1 = plt.imshow(error, cmap='inferno', vmin=0, vmax=12, origin='lower')
cbar = plt.colorbar(im1,fraction=0.046, pad=0.15)



#error for moment 1
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

im1 = plt.imshow(total_error, cmap='inferno', vmin=0, vmax=3, origin='lower')
cbar = plt.colorbar(im1,fraction=0.046, pad=0.15)



#error for moment 2
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

im1 = plt.imshow(moment_2_plot, cmap='inferno', vmin=0, vmax=10, origin='lower')
cbar = plt.colorbar(im1,fraction=0.046, pad=0.15)



#%%


'''
run arcsec to rad program
'''
actual_size = arcsec_to_rad(distance_to_galaxy, arcsec_to_radian, inclination_of_galaxy, cos_inclination)


black_hole_soi_arcsec = 73 / (actual_size/0.146)



'''
initialises data and runs moment maps 
'''


my_header = fits.getheader(filename)  
my_beam = Beam.from_fits_header(my_header)  
BMAJ_projected = my_beam.beam_projected_area((distance_to_galaxy*10**3)*u.kpc).to(u.pc**2)
BMAJ_projected = np.sqrt(BMAJ_projected)/2 #divided by 2 because bmaj is diameter and we want radius (gc_radius)
BMAJ_projected = BMAJ_projected.value

bin_size = int(BMAJ_projected*2)  # Convert beam radius to index units

print(my_header)
print(my_beam)
print(BMAJ_projected)
print(bin_size)
print(type(bin_size))


from astropy.wcs import WCS


# Extract WCS information
wcs = WCS(my_header)

# Get pixel scale (in degrees) and convert to arcseconds
pixel_scale_x = abs(wcs.wcs.cdelt[0]) * 3600 # arcsec/pixel
pixel_scale_y = abs(wcs.wcs.cdelt[1]) * 3600 # arcsec/pixel
#0.146 arcsec per pixel



print(f"Pixel Scale: {pixel_scale_x:.3f} x {pixel_scale_y:.3f} arcsec/pixel")

#%%


'''
run galactocentric radius map program
'''
xi, yi = meshgrid_galactocentric_radius(cube, x_cen, y_cen, moment_2)
cos_a, sin_a, x_proj, y_proj = values_for_galactocentric_radius(xi, yi, position_angle, inclination_of_galaxy, cos_inclination)
x_squared, y_squared, galactocentric_radius, gal_cutout, gcr_mask, selected_pixels, gal_cutout_selected_pixels, bin_width, max_radius, num_bins, bin_edges = get_gcradius(x_proj, y_proj, moment_2, position, size_of_cutout, BMAJ_projected, actual_size)



#%%

'''
run beta program (and finds kappa)
'''


plt.clf()
vcirc, gradient_of_logs, y, errors_for_y, x = mge_vcirc_example(arcsec_to_radian, inclination_of_galaxy,
                                                                distance_to_galaxy, actual_size, 
                                                                gravitational_constant, 
                                                                galaxy_black_hole_mass, radius_range,
                                                                mass_to_light_ratio, mass_to_light_ratio_error, 
                                                                galaxy_inclination_error, black_hole_mass_error,
                                                                black_hole_soi)
plt.pause(1)
    


#%%

'''
run kappa map program
'''

kappa_2d, surface_density, small_sigma, big_sigma, kappa, kappa_2d_error, max_kappa_error, min_kappa_error, kappa_2d_with_error, ml_ratio_with_error, unique_radii, sorted_radii = get_kappa_values(pi, CO_conversion_factor,
                                                                                                                                                    radius_range, vcirc, gradient_of_logs, 
                                                                                                                                                    actual_size, galactocentric_radius, moment_0_data, 
                                                                                                                                                    sigma_map_data, gravitational_constant,
                                                                                                                                                    inclination_of_galaxy, mass_to_light_ratio,
                                                                                                                                                    distance_to_galaxy, y, errors_for_y, x, error,
                                                                                                                                                    moment_2_plot, mass_to_light_ratio_error, 
                                                                                                                                                    galaxy_inclination_error, black_hole_mass_error,
                                                                                                                                                    )

plot_kappa(kappa_2d, moment_2, position, size_of_cutout)













#%%

'''
run Q_gas map program
'''


Q, Q_to_plot, small_sigma_and_error, big_sigma_and_error = plot_Q(small_sigma, kappa_2d, pi, gravitational_constant, big_sigma, moment_2_data, 
                                                                  position, size_of_cutout, kappa_2d_error, kappa_2d_with_error, moment_2_plot, error,
                                                                  surface_density)



#%%


'''
run stellar surface density map program
'''

stellar_density_values, stellar_density_values_and_error, stellar_density_error_values = plot_stellar_surface_density(surface_luminosity, moment_2, position, size_of_cutout, 
                                                                                        dispersion, axial_ratio, x_squared, y_squared,
                                                                                        ml_ratio_with_error, actual_size)



#%%


'''
run stellar velocity dispersion program
'''
surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot, xbin, ybin, mbh_vrms = values_needed_for_vrms_plot(surface_luminosity, dispersion, axial_ratio, 
                                                                                                                 galaxy_black_hole_mass, mass_to_light_ratio,
                                                                                                                 radius_range, actual_size)   



vrms, vrms_with_error = plot_for_vrms(surf_lum, sigma_lum, qobs_lum, surf_pot,
                                      sigma_pot, qobs_pot, mbh_vrms, xbin, ybin,
                                      inclination_of_galaxy, distance_to_galaxy,
                                      mass_to_light_ratio, actual_size, mass_to_light_ratio_error,
                                      black_hole_mass_error, co_disk_radius)



#%%

'''
run stellar velocity dispersion map program
'''
vrms_2d, vrms_2d_with_error = get_vrms_values(unique_radii, sorted_radii, xbin, vrms, 
                                              galactocentric_radius, vrms_with_error)

plot_vrms(vrms_2d, moment_2, position, size_of_cutout, vrms_2d_with_error)





#%%


'''
run Q_star map program
'''
Q_star, Q_star_to_plot = plot_Q_star(vrms_2d, kappa_2d, pi, gravitational_constant,
                     stellar_density_values, moment_2, position, 
                     size_of_cutout, vrms_2d_with_error, kappa_2d_with_error,
                     stellar_density_values_and_error)

#%%

'''
run Q_total map program
'''
Q_total, Q_total_to_plot, Q_total_cutout, selected_pixels_Q_total = plot_Q_total(Q, Q_star, moment_2, position, size_of_cutout, gal_cutout_selected_pixels)


Q_total_error_to_plot, Q_error_cutout, selected_pixels_Q_total_error = Q_total_error_map(moment_2, position, size_of_cutout, Q_total, gal_cutout_selected_pixels)



#%%




qrfig, flat_q, flat_gc, ind, flat_q_error, axs, masked_flat_q, masked_flat_gc, masked_flat_q_error = generate_q_and_r_arrays(Q, Q_total, galactocentric_radius,
                                                                                                                             Q_total_cutout, Q_error_cutout,
                                                                                                                             gal_cutout, selected_pixels_Q_total,
                                                                                                                             selected_pixels_Q_total_error, 
                                                                                                                             gal_cutout_selected_pixels)



#%%
q_gas_flat = Q_to_plot.flatten()
q_gas_flat = q_gas_flat[~np.isnan(q_gas_flat)]

q_star_flat = Q_star_to_plot.flatten()
q_star_ind = np.argsort(flat_gc)
q_star_flat = q_star_flat[q_star_ind]
q_star_not_nan_idx = np.where(~np.isnan(q_star_flat))

    
q_star_flat = q_star_flat[q_star_not_nan_idx]



print(np.nanmedian(q_gas_flat), np.nanmedian(q_star_flat))
print(np.nanmedian(kappa_2d), np.nanmedian(kappa_2d_error))


#%%
variance = variance_from_median(masked_flat_q)

#%%

'''
run Q as a function of R program
'''


data_window, medians, lowpercentile, highpercentile, mean_bin_data, median_bin_data, binned_x = q_as_func_of_r(flat_gc, flat_q, qrfig, flat_q_error,
                                                                                                               Q_total_error_to_plot, axs, actual_size,
                                                                                                               Q_total_cutout, Q_error_cutout, black_hole_soi,
                                                                                                               co_disk_radius, inclination_of_galaxy,
                                                                                                                masked_flat_gc, masked_flat_q, masked_flat_q_error,
                                                                                                               selected_pixels_Q_total_error, selected_pixels_Q_total,
                                                                                                               bin_width, max_radius, num_bins, bin_edges, moment_2,
                                                                                                               bin_size, selected_pixels)



#%%
two_dplot_gaseous_velocity_dispersion, two_dplot_gaseous_surface_density, two_dplot_stellar_velocity_dispersion, two_dplot_stellar_surface_density, two_dplot_epicyclic_frequency, flat_gvd, flat_gsd, flat_svd, flat_ssd, flat_ef, flat_gcr, ktau_Q_gvd, ktau_Q_svd, ktau_Q_gsd, flat_gvd_selected_pixels, flat_gsd_selected_pixels = components_of_both_Qs(small_sigma_and_error, big_sigma_and_error, vrms_2d_with_error,
                                                                                                                                                                                                                                                                                                                                                             stellar_density_values_and_error, kappa_2d_with_error,
                                                                                                                                                                                                                                                                                                                                                             galactocentric_radius, gal_cutout, actual_size, moment_2,
                                                                                                                                                                                                                                                                                                                                                             position, size_of_cutout, medians, flat_gc, flat_q, lowpercentile,
                                                                                                                                                                                                                                                                                                                                                             highpercentile, inclination_of_galaxy, black_hole_soi, co_disk_radius, 
                                                                                                                                                                                                                                                                                                                                                             gcr_mask, masked_flat_q, gal_cutout_selected_pixels,
                                                                                                                                                                                                                                                                                                                                                             bin_size)


#%%
plotderivedproducts(position, size_of_cutout, cube, sigma_map_data, moment_1_data, moment_0_data, moment_0, mean_velocity)


#%%


perturbed_gvd_first_median, perturbed_gvd_second_median, perturbed_gvd_full_median, perturbed_gsd_first_median, perturbed_gsd_second_median, perturbed_gsd_full_median, perturbed_gvd_first_error, perturbed_gvd_second_error, perturbed_gvd_full_error, perturbed_gsd_first_error, perturbed_gsd_second_error, perturbed_gsd_full_error, perturbed_pval_gvd_first_median, perturbed_pval_gvd_second_median, perturbed_pval_gvd_full_median, perturbed_pval_gsd_first_median, perturbed_pval_gsd_second_median, perturbed_pval_gsd_full_median = perturbedktau(flat_gvd_selected_pixels, flat_gsd_selected_pixels, masked_flat_q,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               masked_flat_q_error, flat_q, flat_q_error)














