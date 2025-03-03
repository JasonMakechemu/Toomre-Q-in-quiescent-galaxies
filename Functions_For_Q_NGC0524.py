#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:52:07 2023

@author: husmak
"""


import math
import numpy
import datetime
import numpy as np
import uncertainties #python package for error propagation
import photutils.utils #detects and performs photometry of astronomical sources
import jampy as jampy_package
import matplotlib.pyplot as plt
from astropy.nddata import CCDData


from scipy import stats
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import Cutout2D

#allows for dealing with coordinate systems in Python
import astropy.units as u #imports astronomical units
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import Angle, Latitude, Longitude  # Angles

from statistics import median
from scipy.stats import gaussian_kde

from jampy.mge_vcirc import mge_vcirc
from jampy.jam_axi_proj import jam_axi_proj

from spectral_cube import SpectralCube


from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
from uncertainties import unumpy as unp
from uncertainties import umath

import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


#%%


'''
A data cube is a multi-dimensional ("n-D") array of values. In this case we have a
3D array of values with z, y, x representing the velocity, and the y- and x-coordinates
respectively. This function opens the fits file and gets the data, header, and info from it.
'''


def open_fits_file(filename):
    #open the .fits file of the elliptical early tpe galaxy NGC 0383
    hdu = fits.open(filename)
    
    #gives us information about the data and header of the .fits file, and general file information
    data = hdu[0].data
    header = hdu[0].header
    hdu.info()
    
    return hdu, data, header



'''
This code generates a mask for the moment maps, and error maps.

We have a Gaussian noise distribution withing the data, and we are selecting all pixels
a certain std deviation away from the peak of the noise distribution. This reduces our
chance of selecting noisy pixels.


However it must be stated that this also removes some of the useful data, so a balance
must be struck between removing noise, and keeping useful data.

Higher order moments introduce more noise, than lower order moments, but the mask must be
the same for all  moments.
'''

def create_masks(data):
    std = np.nanstd(data[:3, :, :])
    mask = data > 3.1 * std #mask for moment 0 and moment 1, and moment 0 error map

    return std, mask






'''
SpectralCube allows us to interact and manipulate data cubes with two positional 
dimensions and one spectral dimension (the spectral dimension being velocity).
By default, moments are computed along the spectral dimension, the default is 
the standard in this code.

We assign the .fits file of NGC 0383 to be a spectral cube, then apply a mask to each 
of the generated cubes.

moment 0 is the integrated intensity over the spectral line, the 0th moment, we are 
observing the CO spectrum over ? wavelength. This is because CO is a proxy for cold H2 
which is associated with star forming regions, it doesn't track the densest regions of
stars though (why?).

CO is the most abundant molecule in a galaxy after H2, but CO is much easier to observe. 
ALMA looks at the J 2-1 (second) rotational transition, because ALMA is most sensitive to 
that. Higher order transitions are harder for ALMA to detect.

The CO is 12CO and is in a number of dense molecular clouds associated with H II regions.
'''

def generate_cubes(filename):
    
    cube = SpectralCube.read(filename)  # Initiate a SpectralCube for moment 0
    
    moment_0 = cube.with_spectral_unit(u.km/u.s).moment(order=0)  # Zero-th moment 
    moment_1 = cube.with_spectral_unit(u.km/u.s).moment(order=1)  # Zero-th moment 
    moment_2 = cube.with_spectral_unit(u.km/u.s).moment(order=2)  # Zero-th moment 


    return cube, moment_0, moment_1, moment_2



'''
This function gets the coordinates of the galaxy center in pixels, after inputting the coordinates of the
galaxy center in ra and declination.
'''

def galaxy_coordinates(moment_0, center_ra, center_dec):
    
    '''
    This function gets the center pixels of the CO disk
    from measured ra and dec coordinates.
    '''
    
    '''
    gets the world coordinate system of the moment 0;
    same for all moments because they're in the same place
    '''
        
    mom0header = moment_0.header
    wcs = WCS(mom0header)    
    
    
    '''
    measured central ra and dec 
    '''
    ra = center_ra 
    dec = center_dec 
    
    
    '''
    ra and dec coordinates in the ICRS (International Celestial Reference System)
    '''
    coord = SkyCoord(ra, dec)
    
    
    '''
    world coordinates to pixel coordinates
    '''
    pixelx, pixely = wcs.world_to_pixel(coord) 
    
    
    '''
    The headers corresponding to the central x, and central y pixel of the image.
    '''
    
    x_cen = pixelx
    y_cen = pixely
    
    
    '''
    pixel location of galaxy center
    '''
    position = (x_cen, y_cen) 
    
    
    return position, x_cen, y_cen






'''
This is a function to generate a map of the integrated intensity of CO in the galaxy, 
moment 0.
'''

def moment_0_image(position, size_of_cutout, moment_0): 
    
    
    
    '''
    Initiate a figure and axis object with WCS projection information. The World Coordinate
    System (WCS) describes the geometric transformations between one set of coordinates and
    another.
    
    Also displays the moment map image, specifically the 0th moment.

    '''


    moment_0_data = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_mom0.fits')


    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection=moment_0_data.wcs)
    ax.set_facecolor("black")

    ax.set_xlabel('RA (J2000)')   # Set bottom x-axis label
    ax.set_ylabel('Dec (J2000)')  # Set left y-axis label

    # Access RA and Dec coordinates
    ra = ax.coords['ra']
    dec = ax.coords['dec']

    # Set the number of ticks for RA and Dec
    ra.set_ticks(number=2, direction='in', color='white')   # Set 5 ticks for RA
    dec.set_ticks(number=3, direction='in', color='white')  # Set 5 ticks for Dec

    textstr = 'NGC0524'
    
    


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    im = plt.imshow(moment_0_data, cmap='inferno', vmin=0, vmax=50)
    # Set the axis limits (zoom into the desired region)
    plt.xlim(0, 181)  # Adjust x limits to zoom
    plt.ylim(0, 181)  # Adjust y limits to zoom



    
    
    ''' 
    Add a colorbar and label to the plot.
    '''
    
    cbar = plt.colorbar(im,fraction=0.046, pad=0)
    cbar.set_label('Integrated intensity ($\mathrm{ K km\,s^{-1}}$)', size=12, labelpad=20)
    
    
    fig.savefig("Integrated intensity of CO in NGC 0524 (strict mask).svg", format='svg', dpi=600) 
    fig.savefig("Integrated intensity of CO in NGC 0524 (strict mask).png", format='png', dpi=300, bbox_inches='tight')

    return moment_0_data


















'''
This function plots the intensity weighted velocity of the CO spectral line in the galaxy,
moment 1.

The systemic velocity has been removed.

We want the intensity weighted velocity (because each intensity has it's own velocity)
'''

def moment_1_image(position, size_of_cutout, cube):

    moment_1_data = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_mom1.fits')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=moment_1_data.wcs)
    
    # mean velocity of moment 1
    hdu = moment_1_data.to_hdu()
    
    mean_velocity = numpy.nanmean(hdu[0].data)  
    

    '''
    Display the 1st moment map image, minus the mean to get the non systemic velocity
    and adds a colorbar
    '''
    im2 = plt.imshow((moment_1_data-mean_velocity), cmap='RdBu_r', vmin=-180, vmax=180)

    cbar = plt.colorbar(im2, pad=.07)
    cbar.set_label('Intensity weighted velocity (km/s)', size=16)
    
    
    '''
    This code generates a cutout of the large plot of the first moment 
    '''
    
    #hdu = fits.ImageHDU(data=moment_1.value, header=moment_1.hdu.header)
    
    cutout = Cutout2D((hdu[0].data-mean_velocity), position, size_of_cutout, wcs=moment_1_data.wcs)

    
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(projection=cutout.wcs)
    
    the_image = plt.imshow(cutout.data, cmap='RdBu_r', vmin=-180, vmax=180, origin='lower')
    
    cbar = plt.colorbar(the_image, fraction=0.046, pad=0.25)
    cbar.set_label('Intensity weighted velocity (km/s)', size=12, labelpad=10)
    
    
    '''
    Overlay set of RA/Dec Axes and saves the images
    '''
    overlay2 = ax2.get_coords_overlay('fk5')
    overlay2[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay2[1].set_axislabel('Declination (J2000)', fontsize=16)
    
    fig2.savefig("Intensity weighted velocity of CO in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    fig2.savefig("Intensity weighted velocity of CO in NGC 0524 (strict mask).png", format='png', dpi=600)
    
    
    return moment_1_data, mean_velocity

    
        
'''
This function plots the velocity dispersion of the CO gas, moment 2.
'''

def moment_2_image(position, size_of_cutout, cube):  
            
    
    moment_2_data = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_mom2.fits')
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=moment_2_data.wcs)
    
    im3 = plt.imshow(moment_2_data, cmap='inferno', vmin=0, vmax=400)
    

    # Add a colorbar
    cbar = plt.colorbar(im3, pad=.07)
    cbar.set_label('Velocity (km/s)', size=16)
    
    
    '''
    This code generates a cutout of the large plot of the second moment 
    '''
    
    
    #hdu = fits.ImageHDU(data=np.sqrt(moment_2.value), header=moment_2.hdu.header)
    
    hdu = moment_2_data.to_hdu()

    cutout = Cutout2D(hdu[0].data, position, size_of_cutout, wcs=moment_2_data.wcs)

    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(projection=cutout.wcs)
    
    the_image = plt.imshow(cutout.data, cmap='inferno', vmin=0, vmax=20, origin='lower')

    cbar = plt.colorbar(the_image, fraction=0.046, pad=0.25)
    cbar.set_label('Velocity Dispersion (km^2/s^2)', size=12, labelpad=10)
    
    
    '''
    Overlay set of RA/Dec Axes and saves the images
    '''
    overlay3 = ax3.get_coords_overlay('fk5')
    overlay3[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay3[1].set_axislabel('Declination (J2000)', fontsize=16)
    
    
    fig3.savefig("Effective Width of CO in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    fig3.savefig("Effective Width of CO in NGC 052 (strict mask).png", format='png', dpi=600)

    return moment_2_data



'''
This function plots the line-width of the CO in my galaxy, which is the square root 
of the second moment.

The linewidth is AKA the velocity dispersion in units of km/s
'''

def linewidth_image(position, size_of_cutout, cube):
        
    sigma_map_data = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_ew.fits')


    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection=sigma_map_data.wcs)
    ax.set_xlabel('RA (J2000)')   # Set bottom x-axis label
    ax.set_ylabel('Dec (J2000)')  # Set left y-axis label

    # Access RA and Dec coordinates
    ra = ax.coords['ra']
    dec = ax.coords['dec']

    # Set the number of ticks for RA and Dec
    ra.set_ticks(number=1)   # Set 5 ticks for RA
    dec.set_ticks(number=3)  # Set 5 ticks for Dec

    textstr = 'NGC0524'

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    im4 = plt.imshow(sigma_map_data, cmap='inferno', vmin=0, vmax=20)
    # Set the axis limits (zoom into the desired region)
    plt.xlim(0, 181)  # Adjust x limits to zoom
    plt.ylim(0, 181)  # Adjust y limits to zoom

    # Add a colorbar
    cbar = plt.colorbar(im4, pad=0)
    cbar.set_label('CO Effective Width ($\mathrm{km\,s^{-1}}$)', size=14, labelpad=10)
    
    
    fig.savefig("Effective Width of CO in NGC 0524 (strict mask).svg", format='svg', dpi=300)
    fig.savefig("Effective Width of CO in NGC 0524 (strict mask).png", format='png', dpi=300, bbox_inches='tight')
    
    return sigma_map_data





#%%


'''
ERRORS IN MOMENT MAPS.

Tom has provided the error map cubes, so previous manual calculations are unnecessary.
'''


def error_for_moment_0():
    
    mom0_error = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_emom0.fits')
    
    error = mom0_error
    
    return error





def error_for_moment_1():
    
    mom1_error = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_emom1.fits')

    total_error = mom1_error 
    
    return total_error




def error_for_moment_2():

    #actually effective width error (which is what we want.)
    mom2_error = CCDData.read('ngc0524_12m+7m_co21_2p5kms_strict_eew.fits')

    moment_2_plot = mom2_error #moment 2 error
    
    return moment_2_plot



#%%







'''
Function to convert one arcsec to parcec.

Why does inclination not matter?

does 'actual size'' need perturbing by the distance?

'''

def arcsec_to_rad(distance_to_galaxy, arcsec_to_radian, inclination_of_galaxy, cos_inclination):
        
    distance = distance_to_galaxy #in Mpc
    arcsec_to_rad_conversion_factor = arcsec_to_radian # 1 arcsec in rad   
        
    actual_size = (2 * distance * math.tan(arcsec_to_rad_conversion_factor/2)) * 10**6 #1 arcsec in pc
    
    return actual_size







####################################################################################################################







'''
meshgrid returns coordinate matrices from coordinate vectors.

Want xi and yi to be the center of the galaxy.
'''

def meshgrid_galactocentric_radius(cube, x_cen, y_cen, moment_2_data):
    
    moment_2 = moment_2_data  # Velociy Dispersion - Second moment

    xi, yi = np.meshgrid((np.arange(moment_2.shape[1]) - x_cen),
                         (np.arange(moment_2.shape[0]) - y_cen))
    
    return xi, yi





'''
Function to find required values for the galactocentric radius.
'''

def values_for_galactocentric_radius(xi, yi, position_angle, inclination_of_galaxy, cos_inclination):
    
    '''
    position angle, the angle along where the longer axis of the CO disk is,
    in this case, 142 degrees anti-clockwise
    '''
    angle = position_angle
    
    
                           
    '''
    Getting the following:
    
    The cos and sin of my galaxy position angle.
    
    Galaxy inclination in degrees
    
    Cos of the inclination in radians.
    
    Deprojected distances projected as...
    
    '''
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    inc = inclination_of_galaxy # galaxy inclination in degrees
    
    cos_inc = cos_inclination #cos of the inclination in radians
    
    x_proj = (xi * cos_a + yi * sin_a)  # projection of the ???
    x_proj /= cos_inc

    y_proj = (- xi * sin_a + yi * cos_a) # projection of the ???
    

    return cos_a, sin_a, x_proj, y_proj




'''
Galactocentric radius is r = sqrt(x**2 + y**2), so here I am simply calculating x**2, y**2,
then taking the square root and defining that as equal to the galactocentric radius.

It's then plotted onto the screen. The colorbar refers to the radius in arcsec from 
the CO disk center.

Galactocentric radius is deprojected.

0.146 arcsec per pixel.

BMIN BMAX header is ALMA resolution. (~50 pc)

'''

def get_gcradius(x_proj, y_proj, moment_2, position, size_of_cutout, BMAJ_projected, actual_size):
    
    scale_factor = actual_size * 0.146

    x_squared = np.square(x_proj * scale_factor) #parsec squared
    y_squared = np.square(y_proj * scale_factor) #parsec squared
    galactocentric_radius = np.sqrt(x_squared + y_squared) #in units of parsec
    
          
    '''
    Define the lower and upper bounds of the galactocentric radius range, the region
    where the SMBH SOI ends, and the extent of the gaseous disk ends
    '''
    radius_min = 73
    radius_max = 2500
    
    
    
 
    '''
    making the gc radius map from the black hole soi as measured for this galaxy,
    to the edge of the co disk. This will give us values of Q only between this range,
    the range that we're interested in.
    '''
    bin_width = BMAJ_projected # major axis resolutionin parsec
    max_radius = 2500 # co disk radius in parsec
    
    num_bins = int(np.ceil(max_radius/bin_width)) # number of bins
    bin_edges = np.arange(0, (num_bins + 1)*bin_width, bin_width) # upper and lower bin limits in steps of bin width
    

    
    '''
    Create a boolean mask for the galactocentric radius within the specified range, true means the
    coordinate is assigned a value, False means it's not.
    '''
    gcr_mask = (galactocentric_radius >= radius_min) & (galactocentric_radius <= radius_max)

    
    '''
    Selected pixels is now a map where we have galactocentric radius values only where the points in
    gcr_mask are true. So only within or specified radius range.
    '''
    selected_pixels = np.where(gcr_mask, galactocentric_radius, np.nan)

    gal_cutout_selected_pixels = Cutout2D(selected_pixels.data, position, size_of_cutout, wcs=moment_2.wcs)

    '''
    Quick test to show that our map works as intended
    '''
    plt.imshow(selected_pixels.data, vmin=0, vmax=95.3, origin='lower')
    plt.show()
    

    
    '''
    Makes a HDU, the highest level component of the FITS file structure, consisting of
    a header and (typically) a data array or table. 
    
    Then makes a cutout of it.
    '''
    hdu = fits.ImageHDU(data=galactocentric_radius, header=moment_2.hdu.header)
    gal_cutout = Cutout2D(hdu.data, position, size_of_cutout, wcs=moment_2.wcs)


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=gal_cutout.wcs) #to get coordinates in ra and dec

    im = plt.imshow(gal_cutout.data, vmin=0, vmax=3, origin='lower')
    
     
    cbar = plt.colorbar(im,fraction=0.046, pad=0.25)
    cbar.set_label('Radius in arcsec from galaxy center', size=12, labelpad=10)



    '''
    To get cordinates in ra and dec and save figure
    '''
    
    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
     
    fig.savefig("Galactocentric Radius Map of CO in NGC 0524 s(trict mask).svg", format='svg', dpi=600)
    fig.savefig("Galactocentric Radius Map of CO in NGC 0524 (strict mask).png", format='png', dpi=600)

    plt.show()
    
    return x_squared, y_squared, galactocentric_radius, gal_cutout, gcr_mask, selected_pixels, gal_cutout_selected_pixels, bin_width, max_radius, num_bins, bin_edges

 





'''
This functions computes a projected MGE from the intrinsic one in my text table with values
from North 2019 paper. Code from "mge_vcirc_example" from mge_fit package written by Michele
Cappellari.
'''


def mge_vcirc_example(arcsec_to_radian, inclination_of_galaxy, distance_to_galaxy,
                      actual_size, gravitational_constant, galaxy_black_hole_mass,
                      radius_range, mass_to_light_ratio, mass_to_light_ratio_error,
                      galaxy_inclination_error, black_hole_mass_error, black_hole_soi):
    
    '''
    Radii in arscec where Vcirc has to be computed. 0.05 arcsec to 3 arcsec in 40
    steps in my CO images, the galaxy scale is 0.05 arcsec per pixel or 16pc per pixel.
    '''
    rad = radius_range*actual_size #rad is units of pc
    
    '''
    surface luminosity in luminosity pc^-2,
    standard deviation width of gaussian component of mge for stellar surface luminosity, 
    axial ratio. From Table 1, North et. al. 2019
    
    These values are contained in a local text file.
    
    mass to light ratio is added to the surface luminosity.
    '''
    
    surf, sigma, qobs = np.loadtxt('values_0524.txt').T
    surf_ML = surf*mass_to_light_ratio
    
    G = gravitational_constant      # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    mbh = galaxy_black_hole_mass    #solar masses
    vcirc = mge_vcirc(surf_ML, sigma, qobs, inclination_of_galaxy, mbh, distance_to_galaxy,
                      rad/actual_size)          #find circular velocity of galaxy.
    
    vcirc = vcirc #now vcirc is in km/s instead of m/s
    
    
    '''
    MCMC for adding error to circular velocity.
    
    To find the errors in the circular velocity, all the contributing parameters in the 
    circular velocity are perturbed by their errors. Then the circular velocity is perturbed
    with these perturbed parameters.
    
    They are perturbed in such a way that the values form a gaussian distribution centered
    around 0. meaning that 1 sigma, 2 sigma, etc are easily quantifiable.
    
    The perturbed circular velocities are added to a list, and the "true" circular velocity 
    is the median circular velocity. The one at the center of the normal distribution 
    centered at zero that all the parameters are beind perturbed around.
    
    The error is then taken the be the 84th percentile circular velocity value minus
    the median.
    
    why not perturb the distance? Because it's a systemic error that won't meaningfully affect Q?
    '''
    
    perturbed_vcirc = []
        
    for i in range(1000):
        ml_perturb = mass_to_light_ratio + np.random.normal(loc=0, scale=mass_to_light_ratio_error)
        surf_perturb = surf * ml_perturb
        
        inclination_perturb = inclination_of_galaxy + np.random.normal(loc=0, scale=galaxy_inclination_error)
        perturbed_inclination = inclination_of_galaxy * (inclination_perturb/inclination_of_galaxy)
        
        mbh_perturb = mbh + np.random.normal(loc=0, scale=black_hole_mass_error)
        perturbed_mbh = mbh * (mbh_perturb/mbh)
        
        vcirc_perturb = mge_vcirc(surf_perturb, sigma, qobs, perturbed_inclination, perturbed_mbh, 
                                  distance_to_galaxy, rad/actual_size) #circular velocity of galaxy
        vcirc_perturb = vcirc_perturb
        
        perturbed_vcirc.append(vcirc_perturb)
        

    vcirc_median = np.nanmedian(perturbed_vcirc, axis=0)
    sixteenth_per_perturbed_vcirc = numpy.nanpercentile(perturbed_vcirc, 16, axis=0)
    eighty_fourth_per_perturbed_vcirc = numpy.nanpercentile(perturbed_vcirc, 84, axis=0)
    
   
    vcirc_with_error = unumpy.uarray(vcirc_perturb, (eighty_fourth_per_perturbed_vcirc - vcirc_median))
    

    '''
    This simply plots the circular velocity of the gas as a function of radius (in arcsec).
    '''
    fig = plt.figure(figsize=(8, 8))



    '''
    Plots the following
    
    Vertical line at 320 pc. (~ the black hole sphere of influence)
    Plots the circular velocity points (joined by a curve) (maybe plot vcirc at all radii?)
    Plots the error bars for each point
    '''
    
    plt.axvline(x=black_hole_soi, color='purple', ls='--', lw=2,) 
    plt.plot(rad, vcirc, color='blue')
    plt.errorbar(rad, vcirc_median*np.sin(np.radians(inclination_of_galaxy)),
                 unumpy.std_devs(vcirc_with_error*np.radians(inclination_of_galaxy)), 0,
                 label="Circular Velocity", fmt="o")


    plt.ylim([0, 400]) 
    ax = plt.gca() #gets current axes

    plt.xlabel('Parsec (pc)')
    plt.ylabel('$V_{circ}$ (km/s)')
    plt.legend(loc=4)
    
    fig.savefig('MGE approximation of Circular Velocity of NGC 0524 (strict mask).svg', format='svg',
                dpi=600)
    fig.savefig('MGE approximation of Circular Velocity of NGC 0524 (strict mask).png', format='png',
                dpi=600)
 
    
 
    
 
    
    '''
    This plots the natural log of the circular velocity against the natural
    log of the radius in arcseconds, along with the associated error.
    '''
    fig, ax = plt.subplots()
    x = np.log(rad/actual_size) #natural log of radii
    y = np.log(vcirc*np.sin(np.radians(inclination_of_galaxy))) #natural log of circular velocity
    
    
    '''
    takes the log of the error from v circ
    '''

    errors_for_y = unumpy.log(vcirc_with_error * np.sin(np.radians(inclination_of_galaxy)))
    
    plt.plot(x, unumpy.nominal_values(np.array(errors_for_y)), color='blue')
    plt.errorbar(x, unumpy.nominal_values(np.array(errors_for_y)), unumpy.std_devs(errors_for_y), 0, label='log plot', fmt='o')
    plt.legend()


    plt.xlabel('ln R (parsec)')
    plt.ylabel('ln $V_{circ}$ (km/s)')
    
    fig.savefig('MGE approximation of Circular Velocity of NGC 0524 (in ln format strict mask).svg', format='svg', dpi=600)
    fig.savefig('MGE approximation of Circular Velocity of NGC 0524 (in ln format strict mask).png', format='png', dpi=600)







    
    '''
    the gradient of the natural log is defined as dlnR, and to get the 
    (numerical derivative) of natural log y against natural log x, we us numpy.gradient
    for y and x which are the np.log(vcirc) and np.log(rad) respectively. Giving us a 
    derivative of the lnvcirc/lnR.
    
    This is the beta parameter needed for the epicyclic frequency.
    '''
    
    '''
    log of radius at index 1 vs at index 0? Is it the same as everywhere else?
    where x is defined as np.log(rad)
    '''
    
    dlnR = x[1] - x[0]
    

    gradient_of_logs = np.gradient(y, x) # where y is ln(V_circ), and x is lnr_gal
    fig1, ax1 = plt.subplots()
    
   
  
    '''
    perturbs y (ln vcirc) going into the gradient equation by the error in the logarithm,
    x could be perturbed by the error in the distance which goes into the arcsec to pc
    calculation, but the error wouldn't meaningfully affect Q...? So x is not perturbed.
    '''
    
    perturbed_gradient_list = []
        
    for i in range(1000):
        
        y_perturb = y + np.random.normal(loc=0, scale=unumpy.std_devs(errors_for_y))
             
        gradient_perturb = np.gradient(y_perturb, x)
        
        perturbed_gradient_list.append(gradient_perturb)

    gradient_median = np.nanmedian(perturbed_gradient_list, axis=0)  
    
    
    '''
    error is taken to be the 84th percentile minus the median.
    '''
    sixteenth_per_perturbed_grad = numpy.nanpercentile(perturbed_gradient_list, 16, axis=0)
    eighty_fourth_per_perturbed_grad = numpy.nanpercentile(perturbed_gradient_list, 84,
                                                           axis=0)   
    gradient_with_error = unumpy.uarray(gradient_median, 
                                        ((eighty_fourth_per_perturbed_grad-gradient_median)))
    
    
    plt.plot(x, gradient_median, color='blue')
    plt.errorbar(x, gradient_median, (eighty_fourth_per_perturbed_grad-gradient_median), 
                 0, label='gradient', fmt='o')
    plt.legend(loc=4)


    plt.xlabel('ln R')
    plt.ylabel('d lnV_circ / d lnR (i.e beta)')
    
    fig1.savefig('Derivative of ln V_circ over ln_R against ln_R 0524 (strict mask).svg', format='svg', dpi=600)
    fig1.savefig('Derivative of ln V_circ over ln_R against ln_R 0524 (strict mask).png', format='png', dpi=600)


    
    return vcirc, gradient_of_logs, y, errors_for_y, x






 










'''
This function generates kappa (epicyclic frequency), required for the Toomre Q parameter 
calculations for both gas and stars.

Kappa is defined in equation ... from Tom's paper titled "title"

As is Q, but Q has a general definition as referenced here ...

'''


def get_kappa_values(pi, CO_conversion_factor, radius_range, vcirc, gradient_of_logs, actual_size,
                     galactocentric_radius, moment_0_data, sigma_map_data, gravitational_constant,
                     inclination_of_galaxy, mass_to_light_ratio, distance_to_galaxy, y,
                     errors_for_y, x, error, moment_2_plot, mass_to_light_ratio_error, 
                     galaxy_inclination_error, black_hole_mass_error):

    
    
    
    '''
    surface density of CO in the disk is moment 0; this is multiplied by a CO conversion
    factor as found in Tom's paper (title) to get an estimation of the amount of H2, and 
    other molecular gas contributions within the disk.
    '''
    moment_0_data = moment_0_data.to_hdu()

    surface_density = moment_0_data[0].data * CO_conversion_factor
    
    surface_density[surface_density == 0] = np.nan

    
    '''
    Gravitational Constant (Units of pc * (km/s)^2 * M_sun^-1). These units of G can be used 
    when all other units are ... (can be found on wikipedia)
    '''
    
    G = gravitational_constant  
    
    
    '''
    small sigma is the linewidth/velocity dispersion of the CO in the disk.
    Pretty much the same for HII because the CO is coupled to the HII
    '''
    sigma_map_data = sigma_map_data.to_hdu()
    small_sigma = sigma_map_data[0] #km/s
    
    
    '''
    big_sigma is the surface desity of the gas in disk.
    '''
    big_sigma = surface_density #M_sun / pc^2
    big_sigma[big_sigma == 0] = np.nan

    
    #delicious
    pi = pi
    
    
    
    '''
    
    Reminder of variables in kappa equation:
    
    vcirc - circural velocity in km/s upto 3 arcsec (non systemic)
    rad - radius of CO region in arcsec
    actual_size - 1 arcsec in pc, so converts arcsec to parsec given the distance to my
    galaxy as found in my function titled "arcsec_to_rad".
    gradient_of_logs: np.gradient(y, x) the numerical derivative of log y / log x,
    dimensionless.
        
    
    '''
    
    kappa = np.sqrt(2) * (vcirc/(radius_range*actual_size)) * np.sqrt(1 + gradient_of_logs) 
    kappa = kappa #to get in km/s
    
    rad = radius_range*actual_size

    surf, sigma, qobs = np.loadtxt('values_0524.txt').T
    mbh = 4.0e8 

    perturbed_vcirc = []
    perturbed_gradient_list = []
    perturbed_kappa_list = [] 
        

    '''
    perturbing neccesary variables to get the perturbed kappa, aka epicyclic frequency.
    '''

    for i in range(1000):
        
        ml_perturb = mass_to_light_ratio + np.random.normal(loc=0, scale=mass_to_light_ratio_error)
        surf_perturb = surf * ml_perturb
        
        inclination_perturb = inclination_of_galaxy + np.random.normal(loc=0, scale=galaxy_inclination_error)
        perturbed_inclination = inclination_of_galaxy * (inclination_perturb/inclination_of_galaxy)
        
        mbh_perturb = mbh + np.random.normal(loc=0, scale=black_hole_mass_error)
        perturbed_mbh = mbh * (mbh_perturb/mbh)
        
        vcirc_perturb = mge_vcirc(surf_perturb, sigma, qobs, perturbed_inclination, 
                                  perturbed_mbh, distance_to_galaxy, rad/actual_size) #circular velocity of galaxy
        vcirc_perturb = vcirc_perturb
        
        perturbed_vcirc.append(vcirc_perturb)
                  
        y_perturb = y + np.random.normal(loc=0, scale=unumpy.std_devs(errors_for_y))           
        gradient_perturb = np.gradient(y_perturb, x)       
        perturbed_gradient_list.append(gradient_perturb)
        
        perturbed_kappa = 1.41 * (vcirc_perturb/(radius_range*actual_size)) * np.sqrt(1 + gradient_perturb) 
        perturbed_kappa_list.append(perturbed_kappa)

    
    #error from mass to light ratio
    ml_ratio_with_error = unumpy.uarray(mass_to_light_ratio, mass_to_light_ratio_error)


    kappa_median = np.nanmedian(perturbed_kappa_list, axis=0)          
    sixteenth_per_perturbed_kappa= numpy.nanpercentile(perturbed_kappa_list, 16, axis=0)
    eighty_fourth_per_perturbed_kappa = numpy.nanpercentile(perturbed_kappa_list, 84, axis=0)  
    kappa_with_error = unumpy.uarray(kappa_median,
                                     (eighty_fourth_per_perturbed_kappa-kappa_median))
      
    
    
    

    from scipy.interpolate import interp1d

    # Define grid size and max radius
    max_radius = np.max(galactocentric_radius)  # Maximum galactocentric radius

    # Compute the galactocentric radius in a radially uniform manner
    galactocentric_radius = galactocentric_radius



    # Define kappa with quadratic decrease
    max_kappa = np.max(kappa)
    min_kappa = np.min(kappa)
    steps = 100000  # Number of steps



    # Calculate the decay constant (lambda)
    lambda_value = -np.log(min_kappa / max_kappa) / np.max(rad)

    # Calculate kappa at each step using exponential decay
    dec_kappa = max_kappa * np.exp(-lambda_value * rad)
    print(dec_kappa)

    # Create interpolation function
    interp_func = interp1d(rad, dec_kappa, kind='cubic', fill_value="extrapolate")
    #interp_func = interp1d(radius_range, dec_kappa, kind='linear', fill_value="extrapolate")


    # Interpolate kappa values over the 2D radial grid
    kappa_2d = interp_func(galactocentric_radius)


    # Plot the kappa map
    plt.figure(figsize=(8, 6))
    plt.imshow(kappa_2d, origin='lower', cmap='plasma', extent=[-max_radius, max_radius, -max_radius, max_radius])
    plt.colorbar(label='Interpolated Kappa (0 to $10^8$)')
    plt.xlabel('X-axis (arbitrary units)')
    plt.ylabel('Y-axis (arbitrary units)')
    plt.title('2D Interpolated Kappa Map (Radius Scaled by 80)')
    plt.show()



    # Plot kappa vs galactocentric radius
    plt.figure(figsize=(8, 5))
    plt.plot(radius_range, dec_kappa, color='b', linestyle='-', linewidth=2, label=r'$kappa = (1 - (r / r_{max})^2) \times 10^8$')

    # Labels and title
    plt.xlabel("Galactocentric Radius")
    plt.ylabel("Kappa")
    plt.title("Quadratic Decrease of Kappa with Radius")
    plt.legend()
    plt.grid()








    # Define kappa with quadratic decrease
    max_kappa_error = np.max(unumpy.std_devs(kappa_with_error))
    min_kappa_error = np.min(unumpy.std_devs(kappa_with_error))



    # Calculate the decay constant (lambda)
    lambda_value_error = -np.log(min_kappa_error / max_kappa_error) / np.max(rad)

    # Calculate kappa at each step using exponential decay
    dec_kappa_error = max_kappa_error * np.exp(-lambda_value_error * rad)


    # Create interpolation function
    interp_func_error = interp1d(rad, dec_kappa_error, kind='linear', fill_value="extrapolate")
    #interp_func = interp1d(radius_range, dec_kappa, kind='linear', fill_value="extrapolate")


    # Interpolate kappa values over the 2D radial grid
    kappa_2d_error = interp_func_error(galactocentric_radius)



    kappa_2d_with_error = unumpy.uarray(kappa_2d, kappa_2d_error)

    unique_radii = numpy.unique(galactocentric_radius)
    sorted_radii = numpy.sort(unique_radii)

    
    '''
    Explanation of the following three variables...
    
    unique_radii: finds unique values of my galactocentric radius, and it is in a 1D array
    sorted_radii: sorts my unique radii
    unique_kappa: this uses interpolation to find a kappa at each point of the unique radii
                  that I have.
                  
    
    interpolates for kappa and kappa error
    '''
    
    '''
    from scipy.interpolate import interp1d


    unique_radii = numpy.unique(galactocentric_radius)
    sorted_radii = numpy.sort(unique_radii)
    
    unique_kappa = numpy.interp(sorted_radii, rad, kappa)
                                        
    unique_kappa_error = numpy.interp(sorted_radii, (radius_range*actual_size), unumpy.std_devs(kappa_with_error))
    '''
    
    
    '''
    this code makes an empty array for the 2D values of kappa that I need, it is of the 
    shape of the galactocentric radius seing as for every radial value, there is a value for kappa.
    
    The loop looks through my unique radii list, and finds where the galactocentric radius
    matches the unique radii that I have, this is assigned to idx. Galactocentric radius is
    a 2D array. So now, with kappa_2d[idx], it is made to be equal to unique_kappa[i]. 

    Which thanks to interpolation... is 2D also
    
    '''
    '''
    kappa_2d = np.zeros_like(galactocentric_radius) 
    
    kappa_2d_error = np.zeros_like(galactocentric_radius)
    '''

    
    '''
    this code takes a long time to run because of the amount of values in unique_radii, 
    howver, one could plot in steps of 10 or 100 if they wished for 
    i in range(len(unique_radii[::100])): for example.
    '''
    
    '''
    for i in range(len(unique_radii)):
        idx = np.where(galactocentric_radius == unique_radii[i])
        error_idx = idx
        #print(idx)
        kappa_2d[idx] = unique_kappa[i]
        #print(unique_kappa[i])
        kappa_2d_error[error_idx] = unique_kappa_error[i]
    
    kappa_2d_with_error = unumpy.uarray(kappa_2d, kappa_2d_error)
    
    
    
    print("Unique Kappa", np.max(unique_kappa))                            
    print("Kappa", np.max(kappa))                            
    print("Kappa 2D", np.max(kappa_2d))  
    '''


    return (kappa_2d, surface_density, small_sigma, big_sigma, kappa, kappa_2d_error,
            max_kappa_error, min_kappa_error, kappa_2d_with_error, ml_ratio_with_error,
            unique_radii, sorted_radii)


    
'''
Here I am plotting the map for kappa_2D.
'''  

def plot_kappa(kappa_2d, moment_2, position, size_of_cutout):
    
    
    hdu = fits.ImageHDU(data=kappa_2d, header=moment_2.hdu.header)
    cutout = Cutout2D(hdu.data, position, size_of_cutout, wcs=moment_2.wcs)

    
    kappa_fig = plt.figure(figsize=(8, 8))
    ax = kappa_fig.add_subplot(projection=cutout.wcs) #to get coordinates in ra and dec

    image_of_kappa = plt.imshow(cutout.data, vmin=2, vmax=10, origin='lower') #colourbar between 5% and 95% of kappa values
    

    cbar = plt.colorbar(image_of_kappa,fraction=0.046, pad=0.25)
    cbar.set_label('Epicyclic Frequency', size=12, labelpad=10)
      
    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
    
    
  
    kappa_fig.savefig("Map of Kappa parameter in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    kappa_fig.savefig("Map of Kappa parameter in NGC 0524 (strict mask).png", format='png', dpi=600)
    
    plt.show()






'''
Here I am defining Q_gas and plotting the map for Q_gas. This is according to equation (1)
in Williams et al, yet to be published.

No perturbation is neccessary for Q since all the values in Q have the associated errors 
already. Once the multiplication and division is done, the error has already propagated
thanks to the 'uncertainties' package.
'''


def plot_Q(small_sigma, kappa_2d, pi, gravitational_constant,
           big_sigma, moment_2_data, position, size_of_cutout,
           kappa_2d_error, kappa_2d_with_error,
           moment_2_plot, error, surface_density):      
    
    small_sigma_and_error = unumpy.uarray((small_sigma.data), moment_2_plot.data) #linewidth
    big_sigma_and_error = unumpy.uarray(surface_density.data, error.data) #moment_0 (?)
    
    print("Median ew, and median gsd")
    print(np.nanmedian(unumpy.std_devs(small_sigma_and_error)), np.nanmedian(unumpy.std_devs(big_sigma_and_error)))

      
    Q = (small_sigma_and_error*kappa_2d_with_error) / (pi*gravitational_constant*big_sigma_and_error)
    
    Q_to_plot = unumpy.nominal_values(Q)

 
    hdu = moment_2_data.to_hdu()

    #hdu = fits.ImageHDU(data=Q_to_plot, header=moment_2.hdu.header)
    cutout = Cutout2D(Q_to_plot, position, size_of_cutout, wcs=moment_2_data.wcs)


    Q_fig = plt.figure(figsize=(8, 8))
    ax = Q_fig.add_subplot(projection=cutout.wcs) #to get coordinates in ra and dec
    

    image_of_Q = plt.imshow(cutout.data, cmap='inferno', vmin=0, vmax=70, origin='lower') 

    cbar = plt.colorbar(image_of_Q,fraction=0.046, pad=0.25)
    cbar.set_label(r'$Q_{gas}$', size=12, labelpad=10)
    
    
    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
    
        
    Q_fig.savefig("Map of Q_gas parameter in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    Q_fig.savefig("Map of Q_gas parameter in NGC 0524 (strict mask).png", format='png', dpi=600)

    return Q, Q_to_plot, small_sigma_and_error, big_sigma_and_error

    

    




'''
Code to calculate the stellar surface density in the region of my CO disk.

Stellar density equation is from Cappellari 2002, equation 1.

The for loop is creating an image of stellar surface density for each value of
surface_luminosity, dispersion, and axial ratio respectively. So 6 images. The stellar 
surface density is acquired when each of the 6 images are overlayed on top of each other.
That is what the last plot represents.

Stellar surface density needs no perturbation also; same reason as for Q.

'''


def plot_stellar_surface_density(surface_luminosity, moment_2_data, position, size_of_cutout, 
                                 dispersion, axial_ratio, x_squared, y_squared,
                                 ml_ratio_with_error, actual_size):

    stellar_density_values = np.zeros_like(moment_2_data.data)
    stellar_density_error_values = np.zeros_like(moment_2_data.data)

    for i in range(len(surface_luminosity)):
                 
        
        
        #hdu = fits.ImageHDU(data=stellar_density_values, header=moment_2.hdu.header)
        cutout = stellar_density_values.data
        
        sd_fig = plt.figure(figsize=(5, 4)) # stellar density fig
        ax = sd_fig.add_subplot(111, projection=moment_2_data.wcs) #to get coordinates in ra and dec
        ax.set_xlabel('RA (J2000)')   # Set bottom x-axis label
        ax.set_ylabel('Dec (J2000)')  # Set left y-axis label

        # Access RA and Dec coordinates
        ra = ax.coords['ra']
        dec = ax.coords['dec']

        # Set the number of ticks for RA and Dec
        ra.set_ticks(number=2, direction='in', color='white')   # Set 5 ticks for RA
        dec.set_ticks(number=3, direction='in', color='white')  # Set 5 ticks for Dec

        textstr = 'NGC0524'

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        
    
        stellar_density = ((surface_luminosity[i]*ml_ratio_with_error)/(2*np.pi*(dispersion[i]**2)*axial_ratio[i])) * np.exp((-1/(2*(dispersion[i]**2))) * ((((x_squared/actual_size) * (0.146**2)) + ((y_squared/actual_size) * (0.146**2))/axial_ratio[i]**2))) 
        stellar_density_values += unumpy.nominal_values(stellar_density)
        stellar_density_error_values += unumpy.std_devs(stellar_density)

        stellar_map = plt.imshow(np.log10(cutout),
                                 cmap='inferno', vmin=np.log(1), vmax=np.log(100),
                                 origin='lower')
        
        plt.xlim(0, 181)  # Adjust x limits to zoom
        plt.ylim(0, 181)  # Adjust y limits to zoom
        
        cbar = plt.colorbar(stellar_map, pad=0)
        cbar.set_label(r'$\rm \log_{10}(\Sigma_\ast[M_\odot pc^{-2}])$', size=12, labelpad=12)
        

        sd_fig.savefig("Stellar Surface Density in CO region in NGC 0524 (strict mask).svg", format='svg',
                       dpi=300)
        sd_fig.savefig("Stellar Surface Density in CO region in NGC 0524 (strict mask).png", format='png',
                       dpi=300, bbox_inches='tight')
       
        
    stellar_density_values_and_error = unumpy.uarray(stellar_density_values,
                                                     stellar_density_error_values)
        
    return (stellar_density_values, stellar_density_values_and_error,
            stellar_density_error_values)
    
 




'''
Using jampy, the stellar velocity dispersion can be determined from the surface_luminosity,
dispersion, and axial_ratio values from MGE fitting (North et al. for NGC 383). The surface
luminosity must be multiplied by the M/L ratio because the model predictions are computed
assuming surf_pot gives the total mass, not only visible mass. 

Is this also the case for circular velocity??

The black hole mass (mbh) must also be multiplied by the M/L ratio.

Follow up with Michele... what to expect of mbh output? 

pass M/L ratio to plot_for_vrms

All parameters from North et al.

'''


def values_needed_for_vrms_plot(surface_luminosity, dispersion, axial_ratio, galaxy_black_hole_mass,
                                mass_to_light_ratio, radius_range, actual_size):    
    
    surf_lum = np.array(surface_luminosity)
    sigma_lum = np.array(dispersion)
    qobs_lum = np.array(axial_ratio)
    
    surf_pot = surf_lum 
    sigma_pot = sigma_lum
    qobs_pot = qobs_lum 
    
    mbh_vrms = galaxy_black_hole_mass/mass_to_light_ratio # Black hole mass in kg
    xbin = radius_range*actual_size
    ybin = radius_range*actual_size # region in arcsec for my galaxy dispersion to be calculated
    
    return surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot, xbin, ybin, mbh_vrms
           



def plot_for_vrms(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot, mbh_vrms,
                  xbin, ybin, inc, distance, mass_to_light_ratio, actual_size,
                  mass_to_light_ratio_error, black_hole_mass_error, co_disk_radius):
    
    jam = jam_axi_proj(
             surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
             inc, mbh_vrms, distance, xbin/actual_size, ybin/actual_size, align='cyl', analytic_los=True,
             beta=None, data=None, epsrel=1e-2, errors=None, flux_obs=None,
             gamma=None, goodbins=None, interp=True, kappa=None,
             logistic=False, ml=mass_to_light_ratio, moment='zz', nang=10, nlos=1500,
             nodots=False, normpsf=1., nrad=30, pixang=0., pixsize=0.,
             plot=True, quiet=False, rbh=0.01, sigmapsf=0., step=0)
    
    
    vrms = jam.model  # with moment='zz' the output is the LOS Vrms
    
    
    
    '''
    perturbing velocity dispersion because of the parameters that it depends on.
    '''
    
    perturbed_vrms = []
    
    for i in range(1000):
        ml_perturb = mass_to_light_ratio + np.random.normal(loc=0, scale=mass_to_light_ratio_error)
        perturbed_ml = mass_to_light_ratio * (ml_perturb/mass_to_light_ratio)
        
        mbh_perturb = mbh_vrms + np.random.normal(loc=0, scale=black_hole_mass_error)
        perturbed_mbh = mbh_vrms * (mbh_perturb/mbh_vrms)

        jam_perturbed = jam_axi_proj(
                 surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                 inc, perturbed_mbh, distance, xbin/actual_size, ybin/actual_size, align='cyl', analytic_los=True,
                 beta=None, data=None, epsrel=1e-2, errors=None, flux_obs=None,
                 gamma=None, goodbins=None, interp=True, kappa=None,
                 logistic=False, ml=perturbed_ml, moment='zz', nang=10, nlos=1500,
                 nodots=False, normpsf=1., nrad=30, pixang=0., pixsize=0.,
                 plot=True, quiet=False, rbh=0.01, sigmapsf=0., step=0)
        
        perturbed_vrms.append(jam_perturbed.model)        
        
        
    vrms_median = np.nanmedian(perturbed_vrms, axis=0)
    
    sixteenth_per_perturbed_vrms = numpy.nanpercentile(perturbed_vrms, 16, axis=0)
    eighty_fourth_per_perturbed_vrms = numpy.nanpercentile(perturbed_vrms, 84, axis=0)

    vrms_with_error = unumpy.uarray(vrms_median, (eighty_fourth_per_perturbed_vrms - vrms_median))
    
    
    svd_fig = plt.figure(figsize=(8, 8)) #stellar velocity dispersion fig
    
    
    plt.plot(xbin*actual_size, unumpy.nominal_values(vrms_with_error), color='blue')    
    plt.errorbar(xbin*actual_size, unumpy.nominal_values(vrms_with_error), unumpy.std_devs(vrms_with_error),
                 0, label="Stellar Velocity Dispersion", fmt="o")
    
    plt.xlabel("Radius in Parsec (pc)")
    plt.ylabel("Velocity (km/s)")
    
    plt.legend()
    
    
    plt.xlim([0, co_disk_radius])
    
    plt.xticks(np.arange(0, co_disk_radius, 100))
    
     
    plt.show()
    

    svd_fig.savefig("Stellar Velocity Dispersion in CO region in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    svd_fig.savefig("Stellar Velocity Dispersion in CO region in NGC 0524 (strict mask).png", format='png', dpi=600)

    return vrms, vrms_with_error







'''
function to find the stellar velocity dispersion as a function of radius.
'''

def get_vrms_values(unique_radii, sorted_radii,
                    xbin, vrms, galactocentric_radius, vrms_with_error):
    
    '''
    Explanation of the following variable...
    
    unique_vrms: this uses interpolation to find a V_rms (root mean squared velocity), AKA
                  velocity dispersion at each point of the unique radii that I have.
                  
    '''
    

    unique_vrms = numpy.interp(unique_radii, xbin, vrms)
    
    unique_vrms_error = numpy.interp(unique_radii, xbin, unumpy.std_devs(vrms_with_error))
    
    '''
    this code makes an empty array for the 2D values of V_rms that I need, it is of the 
    shape of the galactocentric radius seeing as for every radial value, there is a value
    for V_rms.
    
    The loop looks through my unique radii list, and finds where the galactocentric radius 
    matches the unique radii that I have, this is assigned to idx. Galactocentric radius is
    a 2D array. So now, with vrms_2d[idx], it is made to be equal to unique_vrms[i]. 
    
    Which thanks to interpolation, is 2d also.    
    '''
    
    vrms_2d = np.zeros_like(galactocentric_radius) 
    
    vrms_2d_error = np.zeros_like(galactocentric_radius) 
    
    
    
    '''
    this code takes a long time to run because of the amount of values in unique_radii, 
    however, one could plot in steps of 10 or 100 if they wished; 
    for i in range(len(unique_radii[::100])): for example.
    '''
    
    for i in range(len(unique_radii)):
        idx = np.where(galactocentric_radius == unique_radii[i])
        vrms_error_idx = idx
        
        vrms_2d[idx] = unique_vrms[i]
        vrms_2d_error[vrms_error_idx] = unique_vrms_error[i]
    
    vrms_2d_with_error = unumpy.uarray(vrms_2d, vrms_2d_error)
        
    return vrms_2d, vrms_2d_with_error
     
   


'''
Here I am plotting the map for vrms_2d.
'''  

def plot_vrms(vrms_2d, moment_2, position, size_of_cutout, vrms_2d_with_error):
    
    
    hdu = fits.ImageHDU(data=unumpy.nominal_values(vrms_2d_with_error), header=moment_2.hdu.header)
    cutout = Cutout2D(hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    

    
    vrms_fig = plt.figure(figsize=(8, 8))
    ax = vrms_fig.add_subplot(projection=cutout.wcs) #to get coordinates in ra and dec
    
    image_of_vrms = plt.imshow(cutout.data, vmin=250,
                               vmax=290, origin='lower') #colourbar between 5% and 95% of kappa values
    


    cbar = plt.colorbar(image_of_vrms, fraction=0.046, pad=0.25)
    cbar.set_label(r'$V_{rms}\ (km/s)$ ', size=12, labelpad=10)

    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)

    
    vrms_fig.savefig("Map of Stellar Velocity Dispersion in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    vrms_fig.savefig("Map of Stellar Velocity Dispersion in NGC 0524 (strict mask).png", format='png', dpi=600)
    
    plt.show()







'''
Here I am defining Q_star and plotting the map for Q_star. This is according to equation (1)
in Williams et al, yet to be published. However, now including V_rms and 
stellar_density_values instead of small_ and big_sigma as seen in Q (which is for Q_gas)

No need for perturbation because components in Q star already have the errors that will be
propagated through.
'''



def plot_Q_star(vrms_2d, kappa_2d, pi, gravitational_constant, 
                stellar_density_values, moment_2, position, 
                size_of_cutout, vrms_2d_with_error, kappa_2d_with_error,
                stellar_density_values_and_error):  


    
    Q_star_with_error = (vrms_2d_with_error*kappa_2d_with_error) / (pi*gravitational_constant*(stellar_density_values_and_error))
    Q_star_only = unumpy.nominal_values(Q_star_with_error)
    Q_star_error = unumpy.std_devs(Q_star_with_error)
    Q_star = unumpy.uarray(Q_star_only, Q_star_error)
    
    Q_star_to_plot = unumpy.nominal_values(Q_star)
    
    
    hdu = fits.ImageHDU(data=Q_star_to_plot, header=moment_2.hdu.header)
    cutout = Cutout2D(hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    
  

    plt.figure(figsize=(8, 8))
    
    Q_star_fig = plt.figure(figsize=(8, 8))
    ax = Q_star_fig.add_subplot(projection=cutout.wcs) #to get coordinates in ra and dec

    '''
    Takes Q values between 150:380 on the x- and y-axes to eliminate as much
    noise as reasonably possible.
    
    '''
    image_of_Q_star = plt.imshow(np.log10(cutout.data),
                                 cmap='inferno', vmin=np.log10(10), vmax=np.log10(1000000),
                                 origin='lower') 


    cbar = plt.colorbar(image_of_Q_star,fraction=0.046, pad=0.25)
    cbar.set_label(r'$\log_{10}(Q_{star})$', size=12, labelpad=12)
    
    #to get cordinates in ra and dec
    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
  
    Q_star_fig.savefig("Map of Q_star parameter in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    Q_star_fig.savefig("Map of Q_star parameter in NGC 0524 (strict mask).png", format='png', dpi=600)

    plt.show()
    
    return Q_star, Q_star_to_plot
    






'''
For Q total, 1/Q_total = 1/Q_gas + 1/Q_star as shown in ...

No perturbation as errors will be propagated through by parameters with the error already.
'''


def plot_Q_total(Q, Q_star, moment_2, position, size_of_cutout, gal_cutout_selected_pixels):  
    


    '''
    Getting Q and Q_star into compatible formats.
    '''
    Q_val, Q_err = np.array(unumpy.nominal_values(Q)), np.array(unumpy.std_devs(Q))
    idx = np.where(np.logical_or(Q_val == 0, Q_err == 0))
    Q_val[idx] = np.nan
    Q_err[idx] = np.nan    
    Q = unp.uarray(Q_val, Q_err)
    
    
    Q_star_val, Q_star_err = np.array(unumpy.nominal_values(Q_star)), np.array(unumpy.std_devs(Q_star))
    idx = np.where(np.logical_or(Q_star_val == 0, Q_star_err == 0))
    Q_star_val[idx] = np.nan
    Q_star_err[idx] = np.nan    
    Q_star = unp.uarray(Q_star_val, Q_star_err)


    Q_total = ((1/Q) + (1/Q_star)) ** -1

    Q_total_to_plot = unumpy.nominal_values(Q_total)    
    
    
    hdu = fits.ImageHDU(data=Q_total_to_plot, header=moment_2.hdu.header)
    Q_total_cutout = Cutout2D(hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    
    
    
    
    
    
    plt.figure(figsize=(5, 4))

    Q_total_fig = plt.figure(figsize=(5, 4))
    ax = Q_total_fig.add_subplot(111, projection=moment_2.wcs) #to get coordinates in ra and dec


    ax.set_xlabel('RA (J2000)')   # Set bottom x-axis label
    ax.set_ylabel('Dec (J2000)')  # Set left y-axis label

    # Access RA and Dec coordinates
    ra = ax.coords['ra']
    dec = ax.coords['dec']

    # Set the number of ticks for RA and Dec
    ra.set_ticks(number=1)   # Set 5 ticks for RA
    dec.set_ticks(number=3)  # Set 5 ticks for Dec

    textstr = 'NGC0524'

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    
    

    '''
    Takes Q values between 150:380 on the x- and y-axes to eliminate as much noise as 
    reasonably possible.
    
    '''
    

    image_of_Q_total = plt.imshow(Q_total_cutout.data, cmap='inferno', vmin=0, vmax=50, origin='lower') 

    plt.xlim(0, 181)  # Adjust x limits to zoom
    plt.ylim(0, 181)  # Adjust y limits to zoom

    cbar = plt.colorbar(image_of_Q_total, pad=0)
    cbar.set_label(r'$Q_{total}$', size=14, labelpad=10)
    
    

    
    #to get cordinates in ra and dec
    '''
    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
    '''
        
    Q_total_fig.savefig("Map of Q_total parameter in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    Q_total_fig.savefig("Map of Q_total parameter in NGC 0524 (strict mask).png", format='png', dpi=600)

    plt.show()
    
    
    
    plt.figure(figsize=(8, 8))    
    selected_pixels_Q_total = np.where(gal_cutout_selected_pixels.data, Q_total_cutout.data, np.nan)
    plt.imshow(selected_pixels_Q_total, cmap='inferno', vmin=0, vmax=5, origin='lower')
         
    
    
    
    
    
    return Q_total, Q_total_to_plot, Q_total_cutout, selected_pixels_Q_total
    




'''
Plots error map for Q_total, for the sake of comparing it with Q_total
'''

def Q_total_error_map(moment_2, position, size_of_cutout, Q_total, gal_cutout_selected_pixels):
    
    
    

    '''
    simply takes the error from Q after all errors are propagated, and plots it.
    '''
    
    Q_total_error_to_plot = unumpy.std_devs(Q_total)



    hdu = fits.ImageHDU(data=Q_total_error_to_plot, header=moment_2.hdu.header)
    Q_error_cutout = Cutout2D(hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
        
   
    Q_error_fig = plt.figure(figsize=(8, 8))
    ax = Q_error_fig.add_subplot(projection=Q_error_cutout.wcs) #to get coordinates in ra and dec


    image_of_Q_total = plt.imshow(Q_error_cutout.data, cmap='inferno',
                                  vmin=0, vmax=3, origin='lower') 
    
    cbar = plt.colorbar(image_of_Q_total,fraction=0.046, pad=0.25)
    

    cbar.set_label(r'Error for $Q_{total}$', size=12, labelpad=10)

    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
        
    Q_error_fig.savefig("Map of Q_total Error in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    Q_error_fig.savefig("Map of Q_total Error in NGC 0524 (strict mask).png", format='png', dpi=600)

    plt.show()
    
    
    plt.figure(figsize=(8, 8))    
    selected_pixels_Q_total_error = np.where(gal_cutout_selected_pixels.data, Q_error_cutout.data, np.nan)
    plt.imshow(selected_pixels_Q_total_error, cmap='inferno', vmin=0, vmax=5, origin='lower')
      
        
    
    
    
    return Q_total_error_to_plot, Q_error_cutout, selected_pixels_Q_total_error


















'''
These functions plot Q total as a function of R

Along with a line going through the median of each data point, and a line fill going to the
16th and 84th percentile above and below the median. i.e. -1 and +1 sigma.
'''

'''
For Q as a function of R, return cutout (with a different variable name)
flatten the new cutout.data
    
'''

def generate_q_and_r_arrays(Q, Q_total, galactocentric_radius,
                            Q_total_cutout, Q_error_cutout,
                            gal_cutout, selected_pixels_Q_total,
                            selected_pixels_Q_total_error, 
                            gal_cutout_selected_pixels):
    
    qrfig, axs = plt.subplots(figsize=(8, 8))
    
    
    '''
    Gets Q vs R as a numpy array
    '''
    Q_total_val, Q_total_err = np.array(Q_total_cutout.data), np.array(Q_error_cutout.data)
    idx = np.where(np.logical_or(Q_total_val == 0, Q_total_err == 0))
    Q_total_val[idx] = np.nan
    Q_total_err[idx] = np.nan    
    QvR_total = unp.uarray(Q_total_val, Q_total_err) # Qvr - Q vs R - Q against R
    
    
    
    '''
    Gets Q vs R as a numpy array (masked)
    '''
    masked_Q_total_val, masked_Q_total_err = np.array(selected_pixels_Q_total), np.array(selected_pixels_Q_total_error)
    masked_idx = np.where(np.logical_or(masked_Q_total_val == 0, masked_Q_total_err == 0))
    masked_Q_total_val[masked_idx] = np.nan
    masked_Q_total_err[masked_idx] = np.nan    
    masked_QvR_total = unp.uarray(masked_Q_total_val, masked_Q_total_err) # Qvr - Q vs R - Q against R
     

    


    
    '''
    Q total and Q total error
    '''
    Q_to_plot = unumpy.nominal_values(QvR_total)  
    Q_error_to_plot = unumpy.std_devs(QvR_total)
    
    '''
    Q total and Q total error (masked)
    '''
    masked_Q_to_plot = unumpy.nominal_values(masked_QvR_total)  
    masked_Q_error_to_plot = unumpy.std_devs(masked_QvR_total)
    




    
    '''
    flattens to 1d so they can be plotted, sorts with galactocentric radius.
    '''
    flat_q = Q_to_plot.flatten()
    flat_gc = (gal_cutout.data).flatten()  

    flat_q_error = (Q_error_cutout.data).flatten()
    
    ind = np.argsort(flat_gc)
    
    flat_q = flat_q[ind]
    flat_gc = flat_gc[ind]
    flat_q_error = flat_q_error[ind]
    
    not_nan_idx = np.where(~np.isnan(flat_q))
    not_nan_idx_error = np.where(~np.isnan(flat_q_error))

    
    flat_q = flat_q[not_nan_idx]
    flat_gc = flat_gc[not_nan_idx]
    
    
    flat_q_error = flat_q_error[not_nan_idx_error]
    
    
    
    '''
    flattens to 1d so they can be plotted, sorts with galactocentric radius masked
    '''
    masked_flat_q = masked_Q_to_plot.flatten()
    masked_flat_gc = (gal_cutout_selected_pixels.data).flatten()  
    masked_flat_q_error = (selected_pixels_Q_total_error).flatten()
    
    masked_ind = np.argsort(masked_flat_gc)
    
    masked_flat_q = masked_flat_q[masked_ind]
    masked_flat_gc = masked_flat_gc[masked_ind]
    masked_flat_q_error = masked_flat_q_error[masked_ind]
    
    masked_not_nan_idx = np.where(~np.isnan(masked_flat_q))
    masked_not_nan_idx_error = np.where(~np.isnan(masked_flat_q_error))

    
    masked_flat_q = masked_flat_q[masked_not_nan_idx]
    masked_flat_gc = masked_flat_gc[masked_not_nan_idx]
    
    
    masked_flat_q_error = masked_flat_q_error[masked_not_nan_idx_error]
    

    print("Masked median Q", np.nanmedian(masked_flat_q))
    print("median Q", np.nanmedian(flat_q))
    print("Masked median Q error", np.nanmedian(masked_flat_q_error))

    return qrfig, flat_q, flat_gc, ind, flat_q_error, axs, masked_flat_q, masked_flat_gc, masked_flat_q_error





def q_as_func_of_r(flat_gc, flat_q, qrfig, flat_q_error,
                   Q_total_error_to_plot, axs, actual_size,
                   Q_total_cutout, Q_error_cutout, black_hole_soi,
                   co_disk_radius, inclination_of_galaxy,
                   masked_flat_gc, masked_flat_q, masked_flat_q_error,
                   selected_pixels_Q_total_error, selected_pixels_Q_total,
                   bin_width, max_radius, num_bins, bin_edges, moment_2,
                   bin_size, selected_pixels):
    
    
    medians = []
    lowpercentile = []
    highpercentile = []
    
    
    medianerror = []
    lowpercentileerror = []
    highpercentileerror = []
    
    
    masked_medians = []
    masked_lowpercentile = []
    masked_highpercentile = []
    
    
    masked_medianerror = []
    masked_lowpercentileerror = []
    masked_highpercentileerror = []    
    
    
    
    
    flat_error_of_q = (Q_error_cutout.data).flatten()
    median_error_of_q = np.nanmedian(flat_error_of_q)


    masked_flat_error_of_q = (selected_pixels_Q_total_error).flatten()
    masked_median_error_of_q = np.nanmedian(masked_flat_error_of_q)
    



    for i in range(len(flat_gc)):
        if 0 <= i < len(flat_gc):
            data_window = flat_q[i:int(i+(bin_size))]  # Use bin_size instead of fixed 20
            lowpercentile.append(numpy.nanpercentile(data_window, 16))
            highpercentile.append(numpy.nanpercentile(data_window, 84))
        
            data_window_error = flat_q_error[i:int(i+bin_size)]
            lowpercentileerror.append(numpy.nanpercentile(data_window_error, 16))
            highpercentileerror.append(numpy.nanpercentile(data_window_error, 84))
        
        else:
            data_window = np.nan
            lowpercentile.append(np.nan)
            highpercentile.append(np.nan)
            
            data_window_error = np.nan
            lowpercentileerror.append(np.nan)
            highpercentileerror.append(np.nan)
           
    
        medians.append(np.nanmedian(data_window))
        medianerror.append(np.nanmedian(data_window_error))
        
        

    y_errorbarsize = median_error_of_q
    print("Y ERROR BAR SIZE", y_errorbarsize)
    




    for i in range(len(masked_flat_gc)):
        if 0 <= i < len(masked_flat_gc) - int(i+bin_size):  # Ensure within bounds
            masked_data_window = masked_flat_q[i:int(i+bin_size)]
            masked_lowpercentile.append(numpy.nanpercentile(masked_data_window, 16))
            masked_highpercentile.append(numpy.nanpercentile(masked_data_window, 84))
    
            masked_data_window_error = masked_flat_q_error[i:int(i+bin_size)]
            masked_lowpercentileerror.append(numpy.nanpercentile(masked_data_window_error, 16))
            masked_highpercentileerror.append(numpy.nanpercentile(masked_data_window_error, 84))
    
            data_window_error = flat_q_error[i:int(i+bin_size)]
            lowpercentileerror.append(numpy.nanpercentile(data_window_error, 16))
            highpercentileerror.append(numpy.nanpercentile(data_window_error, 84))
              
        else:
            masked_data_window = np.nan
            masked_lowpercentile.append(np.nan)
            masked_highpercentile.append(np.nan)
            
            masked_data_window_error = np.nan
            masked_lowpercentileerror.append(np.nan)
            masked_highpercentileerror.append(np.nan)
               
        masked_medians.append(np.nanmedian(masked_data_window))
        masked_medianerror.append(np.nanmedian(masked_data_window_error))
        
    masked_y_errorbarsize = masked_median_error_of_q
    print("Y ERROR BAR SIZE", masked_y_errorbarsize)
 



    # Example usage (replace with your actual data)
    x_data = flat_q
    width_data = flat_gc
    
    # Generate bin ranges dynamically up to index 162
    bin_ranges = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    def get_binned_x(x_data, width_data, bin_ranges):
        binned_x = {i: [] for i in range(len(bin_ranges))}  # Initialize all bins
        for x_value, width_value in zip(x_data, width_data):
            for bin_num, bin_range in enumerate(bin_ranges):
                if bin_range[0] <= width_value < bin_range[1]:
                    binned_x[bin_num].append(x_value)
        return binned_x
    
    binned_x = get_binned_x(x_data, width_data, bin_ranges)
    
    print(len(binned_x))
    print(len(bin_centers))
    
    # Compute mean and median bin data dynamically
    mean_bin_data = tuple(np.nanmean(binned_x[i]) for i in range(len(bin_edges)-1))
    median_bin_data = tuple(np.nanmedian(binned_x[i]) for i in range(len(bin_edges)-1))


    #radii of q data points is too small to get medians in size of bins
    
    plt.figure(figsize=(10,8))
    plt.scatter((flat_gc), flat_q)
    
    plt.figure(figsize=(10,8))
    plt.scatter((masked_flat_gc), masked_flat_q)
    plt.scatter((bin_centers), mean_bin_data, color='red')
    plt.scatter((bin_centers), median_bin_data, color='green')




    
    '''
    Plot each value of Q, followed by the 16th and 84th percentile fill, then the median of
    every Q at a given radius.
    '''
    
    funcqr, axqr = plt.subplots(figsize=(10, 8))
  
    
    
 
    '''    
    Place a representative error bar in corner of 'Q as function of R' plot.
    '''
    
    axis_coordinates_of_representative_error_bar = (120, 100) #position of example error bar
    screen_coordinates_of_representative_error_bar = axqr.transAxes.transform(axis_coordinates_of_representative_error_bar)
    screen_to_data_transform = axqr.transData.inverted().transform
    data_coordinates_of_representative_error_bar = screen_to_data_transform(screen_coordinates_of_representative_error_bar)
    foo = data_coordinates_of_representative_error_bar

    '''
    plot q and gas velocity dispersion on same graph
    '''
    cos_inclination = np.cos(math.radians(inclination_of_galaxy))

   
    '''
    Plot each value of Q, followed by the 16th and 84th percentile fill, then the median of
    every Q at a given radius.
    '''
    
    plt.grid(zorder=2)

    color1 = 'green'
    color2 = 'red'

    my_scatter = plt.scatter((flat_gc), flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', zorder=1, label='$Q_\mathrm{pix}$')
    my_fill = plt.fill_between((flat_gc), lowpercentile, highpercentile, alpha=0.1, color=color1, zorder=2)
    my_median, = plt.plot(flat_gc, medians, color=color1, zorder=3, label='Running median of $Q_\mathrm{pix}$')
        
    my_beam_scatter = plt.scatter((bin_centers), median_bin_data, color=color2, zorder=4, label='$Q_\mathrm{pix}$ beam')

    my_line = plt.axvline(x=black_hole_soi, color='black', ls='--', lw=2, zorder=5, label='$R_\mathrm{SOI}$') 
    #my_other_line = plt.axvline(x=299.634, color='blue', ls='--', lw=2, zorder=6, label='$R_\mathrm{disk}$') #extent of co disk


    #sets minor ticks
    minor_locator = MultipleLocator(5)
    axqr.xaxis.set_minor_locator(minor_locator)

    plt.errorbar(foo[0], foo[1],  yerr=y_errorbarsize, capsize=3, marker="", color='black')
    plt.legend(loc=4, frameon=True)


    plt.xlabel("$R_\mathrm{gal}$ (pc)", fontsize=14)
    plt.ylabel('$Q_{total}$', fontsize=14)
    
    plt.xlim([0, np.nanmax(masked_flat_gc)])
    plt.ylim([0, 100]) #q values
    
    plt.xticks(np.arange(0, np.nanmax(masked_flat_gc), 200))
    
    #extent of gcr according to galactocentric_radius, extent of masked_flat_q
    
      
    funcqr.savefig("Q as function of radius 0524 (strict mask).svg", format='svg', dpi=600)         
    funcqr.savefig("Q as function of radius 0524 (strict mask).png", format='png', dpi=600)                 
        
    
    plt.show()
    
    
    
    
    


    
    plt.figure(figsize=(5, 4))

    Q_total_fig = plt.figure(figsize=(5, 4))
    ax = Q_total_fig.add_subplot(111, projection=moment_2.wcs) #to get coordinates in ra and dec


    ax.set_xlabel('RA (J2000)')   # Set bottom x-axis label
    ax.set_ylabel('Dec (J2000)')  # Set left y-axis label

    # Access RA and Dec coordinates
    ra = ax.coords['ra']
    dec = ax.coords['dec']

    # Set the number of ticks for RA and Dec
    ra.set_ticks(number=1)   # Set 5 ticks for RA
    dec.set_ticks(number=3)  # Set 5 ticks for Dec

    textstr = 'NGC0524'

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    
    
    
    ''' 
    Takes Q values between 150:380 on the x- and y-axes to eliminate as much noise as 
    reasonably possible.
    
    '''
    

    image_of_Q_total = plt.imshow(Q_total_cutout.data, cmap='inferno', vmin=0, vmax=100, origin='lower') 

    plt.xlim(0, 181)  # Adjust x limits to zoom
    plt.ylim(0, 181)  # Adjust y limits to zoom

    cbar = plt.colorbar(image_of_Q_total, pad=0)
    cbar.set_label(r'$Q_{total}$', size=14, labelpad=10)
    
    

    
    #to get cordinates in ra and dec
    '''
    overlay = ax.get_coords_overlay('fk5')
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
    '''
        
    Q_total_fig.savefig("Map of Q_total parameter in NGC 0524 (strict mask).svg", format='svg', dpi=600)
    Q_total_fig.savefig("Map of Q_total parameter in NGC 0524 (strict mask).png", format='png', dpi=600)

    plt.show()
    
    
    
    
    
    
    
    
    

    
    
    
    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122, projection=moment_2.wcs)

    
    # Define a figure with two subplots side-by-side
    #fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Adjust width to your needs
    plt.subplots_adjust(wspace=0.3) 
    
    # =========================
    # Plot 1: Scatter and Line Plot
    # =========================
    ax1.grid(zorder=1)
    ax1.scatter(flat_gc, flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', zorder=1, label=r'$Q_{\rm pix}$')
    ax1.fill_between(flat_gc, lowpercentile, highpercentile, alpha=0.1, color=color1, zorder=2)
    ax1.plot(flat_gc, medians, color=color1, zorder=3, label='$Q_\mathrm{pix}$ running median')    
    ax1.scatter(bin_centers, median_bin_data, color=color2, zorder=4, label=r'$Q_{\rm pix}$ beam')
    ax1.tick_params(axis='both', direction='in', which='both')  # Applies to major and minor ticks
    
    # Add vertical lines
    #ax1.axvline(x=black_hole_soi, color='black', ls='--', lw=2, zorder=5, label=r'$R_{\rm SOI}$')
    #ax1.axvline(x=299.634, color='blue', ls='--', lw=2, zorder=6, label=r'$R_{\rm disk}$')
    
    # Minor ticks
    minor_locator = MultipleLocator(20)
    ax1.xaxis.set_minor_locator(minor_locator)
    
    # Error bars
    ax1.errorbar(foo[0], foo[1], yerr=y_errorbarsize, capsize=3, marker="", color='black')
    
    # Labels, legends, and limits for ax1
    ax1.set_xlabel(r"$R_{\rm gal}$ (pc)", fontsize=14)
    ax1.set_ylabel(r'$Q_{\rm T}$', fontsize=14)
    ax1.legend(loc=1, fontsize=11)
    ax1.set_xlim([black_hole_soi, np.nanmax(masked_flat_gc)])
    ax1.set_ylim([0, 120])
    ax1.set_xticks(np.arange(0, np.nanmax(masked_flat_gc), 200))
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    # =========================
    # Plot 2: Image with RA/Dec Coordinates
    # =========================
    # Here we define ax2 as a subplot with WCS projection if needed
    #ax2 = fig1.add_subplot(projection=moment_2.wcs)
    
    #fig, ax2 = plt.subplots()  # Create a standard Matplotlib axes without WCS
    
    ax2.set_xlabel('RA (J2000)')
    ax2.set_ylabel('Dec (J2000)')
    
    
    # Set the number of ticks for RA and Dec
    ra = ax2.coords['ra']
    dec = ax2.coords['dec']
    ra.set_ticks(number=2, direction='in')
    dec.set_ticks(number=3, direction='in')
    

    
    # Plot image in pixel coordinates
    image_of_Q_total = ax2.imshow(Q_total_cutout.data, cmap='inferno', vmin=0, vmax=60, origin='lower')
    
    
    
    
    
    
    from matplotlib.patches import Rectangle

    
    distance_to_source = 23300000
    the_px_scale = 0.146
    bar_len_pc = 500
    bar_length_pix = (bar_len_pc * 206265) / (distance_to_source * the_px_scale)  # Convert to pixels

    # Add scale bar
    bar_x, bar_y = 10, 8  # Adjust position as needed
    ax2.add_patch(Rectangle((bar_x, bar_y), bar_length_pix, 3, color='black'))

    # Label the scale bar
    ax2.text(bar_x + bar_length_pix / 2, bar_y + 7, f"{bar_len_pc} pc",
        color='black', fontsize=12, ha='center')
    
    
    
    # Text box with object label
    textstr = 'NGC0524'
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=props)

    cbar = plt.colorbar(image_of_Q_total, ax=ax2, pad=0)
    cbar.set_label(r'$Q_{\rm T}$', size=14, labelpad=10)
    
    # =========================
    # Save and Show the Figure
    # =========================
    # Save both plots in one figure
    fig1.savefig("Combined_plot_0524.svg", format='svg', dpi=600)
    fig1.savefig("Combined_plot_0524.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
        
    
    
    
    
    
    
    
    
    return data_window, medians, lowpercentile, highpercentile, mean_bin_data, median_bin_data, binned_x
    
    



def variance_from_median(masked_flat_q): #asymptotic_variance_median
    """Calculates the variance from the median of an array.

    Args:
        array: The input array.

    Returns:
        The variance from the median.
    """

    n = len(masked_flat_q)
    median = np.nanmedian(masked_flat_q)
    
    median_absolute_deviation = stats.median_abs_deviation(masked_flat_q)
    
    # Estimate the (Probability Density Function) PDF at the median using kernel density estimation  
    kde = gaussian_kde(masked_flat_q)
    pdf_at_median = kde(median)  
    
    # Calculate the asymptotic variance
    variance = 1 / (4 * n * pdf_at_median ** 2)
    
    standard_deviation = np.sqrt(variance)
    
    print("Variance", variance)
    print("standard_deviation", standard_deviation)
    
    return variance












    
    
def components_of_both_Qs(small_sigma_and_error, big_sigma_and_error, vrms_2d_with_error,
                          stellar_density_values_and_error, kappa_2d_with_error,
                          galactocentric_radius, gal_cutout, actual_size, moment_2,
                          position, size_of_cutout, medians, flat_gc, flat_q, lowpercentile,
                          highpercentile, inclination_of_galaxy, black_hole_soi, co_disk_radius, 
                          gcr_mask, masked_flat_q, gal_cutout_selected_pixels, bin_size):
    
    
    '''
    values for Q parameters
    '''
    two_dplot_gaseous_velocity_dispersion = unumpy.nominal_values(small_sigma_and_error)

    
    two_dplot_gaseous_surface_density = unumpy.nominal_values(big_sigma_and_error)


    two_dplot_stellar_velocity_dispersion = unumpy.nominal_values(vrms_2d_with_error)


    two_dplot_stellar_surface_density = unumpy.nominal_values(stellar_density_values_and_error)


    two_dplot_epicyclic_frequency = unumpy.nominal_values(kappa_2d_with_error)


    
    '''
    cutouts of Q parameters
    '''
    gvd_hdu = fits.ImageHDU(data=two_dplot_gaseous_velocity_dispersion, header=moment_2.hdu.header)
    gvd_cutout = Cutout2D(gvd_hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    gvd_cutout = np.array(gvd_cutout.data)
    gsv_idx = np.where(gvd_cutout == 0)
    gvd_cutout[gsv_idx] = np.nan

    gsd_hdu = fits.ImageHDU(data=two_dplot_gaseous_surface_density, header=moment_2.hdu.header)
    gsd_cutout = Cutout2D(gsd_hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    gsd_cutout = np.array(gsd_cutout.data)
    gsd_idx = np.where(gsd_cutout == 0)
    gsd_cutout[gsd_idx] = np.nan
    

    svd_hdu = fits.ImageHDU(data=two_dplot_stellar_velocity_dispersion, header=moment_2.hdu.header)
    svd_cutout = Cutout2D(svd_hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    svd_cutout = np.array(svd_cutout.data)
    svd_idx = np.where(svd_cutout == 0)
    svd_cutout[svd_idx] = np.nan
    

    ssd_hdu = fits.ImageHDU(data=two_dplot_stellar_surface_density, header=moment_2.hdu.header)
    ssd_cutout = Cutout2D(ssd_hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    ssd_cutout = np.array(ssd_cutout.data)
    ssd_idx = np.where(ssd_cutout == 0)
    ssd_cutout[ssd_idx] = np.nan
    
    
    ef_hdu = fits.ImageHDU(data=two_dplot_epicyclic_frequency, header=moment_2.hdu.header)
    ef_cutout = Cutout2D(ef_hdu.data, position, size_of_cutout, wcs=moment_2.wcs)
    ef_cutout = np.array(ef_cutout.data)
    ef_idx = np.where(ef_cutout == 0)
    ef_cutout[ef_idx] = np.nan   
    
    
    '''
    flattened arrays of Q parameters
    '''
    flat_gvd = (gvd_cutout).flatten()
    flat_gsd = (gsd_cutout).flatten()
    
    flat_svd = (svd_cutout).flatten()
    flat_ssd = (ssd_cutout).flatten()
    
    flat_ef = (ef_cutout).flatten()
    flat_gcr = (gal_cutout.data).flatten()  
   


    '''
    masked flattened arrays of Q parameters
    '''    

    flat_gvd_selected_pixels = np.where(gal_cutout_selected_pixels.data, gvd_cutout, np.nan).flatten()
    flat_gsd_selected_pixels = np.where(gal_cutout_selected_pixels.data, gsd_cutout, np.nan).flatten()
    
    flat_svd_selected_pixels = np.where(gal_cutout_selected_pixels.data, svd_cutout, np.nan).flatten()
    flat_ssd_selected_pixels = np.where(gal_cutout_selected_pixels.data, ssd_cutout, np.nan).flatten()
    
    flat_ef_selected_pixels = np.where(gal_cutout_selected_pixels.data, ef_cutout, np.nan).flatten()
    flat_gcr_selected_pixels = np.where(gal_cutout_selected_pixels.data, gal_cutout.data, np.nan).flatten()
    






    
    '''
    flattens to 1d so they can be plotted, sorts by the galactocentric radius.
    '''

    ind = np.argsort(flat_gcr)
    
    flat_gvd = flat_gvd[ind]
    flat_gsd = flat_gsd[ind]
    flat_svd = flat_svd[ind]
    flat_ssd = flat_ssd[ind]
    flat_ef = flat_ef[ind]
    flat_gcr = flat_gcr[ind]
    
 
    #masked
    selected_pixels_ind = np.argsort(flat_gcr_selected_pixels)
    
    flat_gvd_selected_pixels = flat_gvd_selected_pixels[selected_pixels_ind]
    flat_gsd_selected_pixels = flat_gsd_selected_pixels[selected_pixels_ind]
    flat_svd_selected_pixels = flat_svd_selected_pixels[selected_pixels_ind]
    flat_ssd_selected_pixels = flat_ssd_selected_pixels[selected_pixels_ind]
    flat_ef_selected_pixels = flat_ef_selected_pixels[selected_pixels_ind]
    flat_gcr_selected_pixels = flat_gcr_selected_pixels[selected_pixels_ind]    
        
 
    
 
    
    
    not_nan_idx = np.where(~np.isnan(flat_q))
    
    flat_gcr = flat_gcr[not_nan_idx]
    
    flat_gvd = flat_gvd[not_nan_idx]
    
    flat_gsd = flat_gsd[not_nan_idx]  
    
    flat_svd = flat_svd[not_nan_idx]      
    
    flat_ssd = flat_ssd[not_nan_idx]     
    
    flat_ef = flat_ef[not_nan_idx]      
    
    
    
    flat_gcr_gvd = flat_gcr[not_nan_idx]
    flat_gcr_gsd = flat_gcr[not_nan_idx]
    flat_gcr_svd = flat_gcr[not_nan_idx]
    flat_gcr_ssd = flat_gcr[not_nan_idx]
    flat_gcr_ef = flat_gcr[not_nan_idx]




    #masked
    selected_pixels_not_nan_idx = np.where(~np.isnan(masked_flat_q))
    
    flat_gcr_selected_pixels = flat_gcr_selected_pixels[selected_pixels_not_nan_idx]
    
    flat_gvd_selected_pixels = flat_gvd_selected_pixels[selected_pixels_not_nan_idx]
    
    flat_gsd_selected_pixels = flat_gsd_selected_pixels[selected_pixels_not_nan_idx]  
    
    flat_svd_selected_pixels = flat_svd_selected_pixels[selected_pixels_not_nan_idx]      
    
    flat_ssd_selected_pixels = flat_ssd_selected_pixels[selected_pixels_not_nan_idx]     
    
    flat_ef_selected_pixels = flat_ef_selected_pixels[selected_pixels_not_nan_idx]      






    medians_gvd = []
    medians_gsd = []
    medians_svd = []
    medians_ssd = []
    medians_ef = []


    lowpercentile_gvd = []
    lowpercentile_gsd = []
    lowpercentile_svd = []
    lowpercentile_ssd = []
    lowpercentile_ef = []

    highpercentile_gvd = []
    highpercentile_gsd = []
    highpercentile_svd = []
    highpercentile_ssd = []
    highpercentile_ef = []

  
    
            
    for i in range(len(flat_gcr_gvd)):
        if np.logical_and(i > 0, i < len(flat_gcr_gvd)):
            pad_gvd = numpy.pad(flat_gvd, (0, 0), mode='median')  # DO NOT PAD
            gvd_data_window = pad_gvd[i:i + int(bin_size)]  # Use bmaj_projected as the bin size
            lowpercentile_gvd.append(numpy.nanpercentile(gvd_data_window, 16))
            highpercentile_gvd.append(numpy.nanpercentile(gvd_data_window, 84))
        else:
            gvd_data_window = np.nan
            lowpercentile_gvd.append(np.nan)
            highpercentile_gvd.append(np.nan) 
        medians_gvd.append(np.nanmedian(gvd_data_window))


    for i in range(len(flat_gcr_gsd)):
        if np.logical_and(i > 0, i < len(flat_gcr_gsd)):
            pad_gsd = numpy.pad(flat_gsd, (0,0), mode='median') #affects how shifted up and down the median green line is. DO NOT PAD
            gsd_data_window = pad_gsd[i:i+int(bin_size)] #takes the median between these ranges
            lowpercentile_gsd.append(numpy.nanpercentile(gsd_data_window, 16))
            highpercentile_gsd.append(numpy.nanpercentile(gsd_data_window, 84))
        else:
            gsd_data_window = np.nan
            lowpercentile_gsd.append(np.nan)
            highpercentile_gsd.append(np.nan) 
        medians_gsd.append(np.nanmedian(gsd_data_window))


    for i in range(len(flat_gcr_svd)):
        if np.logical_and(i > 0, i < len(flat_gcr_svd)):
            pad_svd = numpy.pad(flat_svd, (0,0), mode='median') #affects how shifted up and down the median green line is. DO NOT PAD
            svd_data_window = pad_svd[i:i+int(bin_size)] #takes the median between these ranges
            lowpercentile_svd.append(numpy.nanpercentile(svd_data_window, 16))
            highpercentile_svd.append(numpy.nanpercentile(svd_data_window, 84))
        else:
            svd_data_window = np.nan
            lowpercentile_svd.append(np.nan)
            highpercentile_svd.append(np.nan) 
        medians_svd.append(np.nanmedian(svd_data_window))


    for i in range(len(flat_gcr_ssd)):
        if np.logical_and(i > 0, i < len(flat_gcr_ssd)):
            pad_ssd = numpy.pad(flat_ssd, (0,0), mode='median') #affects how shifted up and down the median green line is. DO NOT PAD
            ssd_data_window = pad_ssd[i:i+int(bin_size)] #takes the median between these ranges
            lowpercentile_ssd.append(numpy.nanpercentile(ssd_data_window, 16))
            highpercentile_ssd.append(numpy.nanpercentile(ssd_data_window, 84))
        else:
            ssd_data_window = np.nan
            lowpercentile_ssd.append(np.nan)
            highpercentile_ssd.append(np.nan) 
        medians_ssd.append(np.nanmedian(ssd_data_window))


    for i in range(len(flat_gcr_ef)):
        if np.logical_and(i > 0, i < len(flat_gcr_ef)):
            pad_ef = numpy.pad(flat_ef, (0,0), mode='median') #affects how shifted up and down the median green line is. DO NOT PAD
            ef_data_window = pad_ef[i:i+int(bin_size)] #takes the median between these ranges
            lowpercentile_ef.append(numpy.nanpercentile(ef_data_window, 16))
            highpercentile_ef.append(numpy.nanpercentile(ef_data_window, 84))
        else:
            ef_data_window = np.nan
            lowpercentile_ef.append(np.nan)
            highpercentile_ef.append(np.nan) 
        medians_ef.append(np.nanmedian(ef_data_window))








    '''
    plots each of the parameters as a function of galactocentric radius.
    '''
    
    '''
    plt.subplots(figsize=(8, 8))
    plt.scatter(flat_gcr_gvd, flat_gvd, marker='o', color='black', alpha=1, s=1.25, edgecolors='none')
    plt.plot(flat_gcr_gvd, medians_gvd, color='green')
    plt.xlim([36, 224.6])
    plt.ylim([0, 20]) #q values
    plt.xticks(np.arange(36, 224.6, 40))

    plt.subplots(figsize=(8, 8))
    plt.scatter(flat_gcr_gsd, flat_gsd, marker='o', color='black', alpha=1, s=1.25, edgecolors='none')
    plt.plot(flat_gcr_gsd, medians_gsd, color='green')
    plt.xlim([36, 224.6])
    plt.ylim([0, 20]) #q values
    plt.xticks(np.arange(36, 224.6, 40))
    
    plt.subplots(figsize=(8, 8))
    plt.scatter(flat_gcr_svd, flat_svd, marker='o', color='black', alpha=1, s=1.25, edgecolors='none')
    plt.plot(flat_gcr_svd, medians_svd, color='green')
    plt.xlim([36, 224.6])
    plt.ylim([250, 285]) #q values
    plt.xticks(np.arange(36, 224.6, 40))
    
    plt.subplots(figsize=(8, 8))
    plt.scatter(flat_gcr_ssd, flat_ssd, marker='o', color='black', alpha=1, s=1.25, edgecolors='none')
    plt.plot(flat_gcr_ssd, medians_ssd, color='green')
    plt.xlim([36, 224.6])
    plt.ylim([0, 700]) #q values
    plt.xticks(np.arange(36, 224.6, 40))
    
    
    plt.subplots(figsize=(8, 8))
    plt.plot(flat_gcr_ef, medians_ef, color='green')
    plt.scatter(flat_gcr_ef, flat_ef, marker='o', color='black', alpha=1, s=1.25, edgecolors='none')
    plt.xlim([36, 224.6])
    plt.ylim([0, 15]) #q values
    plt.xticks(np.arange(36, 224.6, 40))
    '''
    
    
    
    '''
    plot q and gas velocity dispersion on same graph
    '''
    cos_inclination = np.cos(math.radians(inclination_of_galaxy))


    
    data1 = medians #for Q
    data2 = medians_gvd #for gas velocity dispersion
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    fig.suptitle('Q vs. Gas Velocity Dispersion', fontsize=16)
    
    color = 'tab:red'
    ax1.set_xlabel('Radius (pc)')
    ax1.set_ylabel('Toomre Q', color=color)
    plt.fill_between((flat_gc), lowpercentile, highpercentile, alpha=0.1, color=color)
    plt.plot((flat_gc), medians, color=color, label='Toomre Q')
    plt.legend()
    plt.xlim([black_hole_soi, 500])
    plt.ylim([0, 80]) #q values

    plt.xticks(np.arange(black_hole_soi, co_disk_radius, 40))
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Gas Velocity Dispersion (km/s)', color=color)  # we already handled the x-label with ax1
    plt.fill_between((flat_gc), lowpercentile_gvd, highpercentile_gvd, alpha=0.1, color=color)

    plt.plot((flat_gc), medians_gvd, color=color, label='Gas Velocity Dispersion')
    plt.xlim([black_hole_soi, 500])
    plt.ylim([0, 10]) #gas velcoity dispersion

    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    fig.savefig("Q and Gaseous Velocity Dispersion NGC 0524 (strict mask).png", format='png', dpi=600)                 

    plt.show()
   
    
   
    
   
    
    '''
    plot q and stellar velocity dispersion on same graph
    '''
    

    #t = flat_gcr*actual_size
    data1 = medians #for Q
    data2 = medians_svd #for gas velocity dispersion
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    fig.suptitle('Q vs. Stellar Velocity Dispersion', fontsize=16)

    color = 'tab:red'
    ax1.set_xlabel('Radius (pc)')
    ax1.set_ylabel('Toomre Q', color=color)
    plt.fill_between((flat_gc), lowpercentile, highpercentile, alpha=0.1, color=color)

    plt.plot((flat_gc), medians, color=color, label='Toomre Q')
    plt.legend()
    plt.xlim([black_hole_soi, co_disk_radius])
    plt.ylim([0, 55]) #q values

    plt.xticks(np.arange(black_hole_soi, co_disk_radius, 40))
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Stellar Velocity Dispersion (km/s)', color=color)  # we already handled the x-label with ax1
    plt.plot((flat_gc), medians_svd, color=color, label='Stellar Velocity Dispersion')
    plt.xlim([black_hole_soi, co_disk_radius])
    plt.ylim([275, 280]) #q values

    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    fig.savefig("Q and Stellar Velocity Dispersion NGC 0524 (strict mask).png", format='png', dpi=600)                 

    plt.show()
      
    
    
    
    
    
    
    
    '''
    plot q and gas surface density on same graph
    
    # gas surface density (moment_0 * CO_conversion_factor)
    '''
    

    #t = flat_gcr*actual_size
    data1 = medians #for Q
    data2 = medians_gsd #for gas surface dispersion
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    fig.suptitle('Q vs. Gas Surface Density', fontsize=16)

    color = 'tab:red'
    ax1.set_xlabel('Radius (pc)')
    ax1.set_ylabel('Toomre Q', color=color)
    plt.fill_between((flat_gc), lowpercentile, highpercentile, alpha=0.1, color=color)

    plt.plot((flat_gc), medians, color=color, label='Toomre Q')
    plt.legend()

    plt.xlim([black_hole_soi, 500])
    plt.ylim([0, 80]) #q values

    plt.xticks(np.arange(black_hole_soi, co_disk_radius, 40))
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Gaseous Surface Density (K kms^-1)', color=color)  # we already handled the x-label with ax1
    plt.fill_between((flat_gc), lowpercentile_gsd, highpercentile_gsd, alpha=0.1, color=color)

    plt.plot((flat_gc), medians_gsd, color=color, label='Gas Surface Density')
    plt.xlim([black_hole_soi, 500])
    plt.ylim([0, 80]) 

    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    fig.savefig("Q and Gaseous Surface Density NGC 0524 (strict mask).png", format='png', dpi=600)                 

    plt.show()

    



    '''
    Kendall-Tau correlation between my two parameters, Q vs whatever other parameter it is.
    '''
    ktau_Q_gvd = stats.kendalltau([flat_gvd],[flat_q], nan_policy='omit')
    ktau_Q_svd = stats.kendalltau([flat_svd],[flat_q], nan_policy='omit')
    ktau_Q_gsd = stats.kendalltau([flat_gsd],[flat_q], nan_policy='omit')
    
    '''
    masked values for the kendall tau correlation
    '''
    masked_ktau_Q_gvd = stats.kendalltau([flat_gvd_selected_pixels],[masked_flat_q], nan_policy='omit', method='auto')
    masked_ktau_Q_svd = stats.kendalltau([flat_svd_selected_pixels],[masked_flat_q], nan_policy='omit')
    masked_ktau_Q_gsd = stats.kendalltau([flat_gsd_selected_pixels],[masked_flat_q], nan_policy='omit', method='auto')
    
    
    print("masked KTAU GVD", masked_ktau_Q_gvd)
    print("masked KTAU GSD", masked_ktau_Q_gsd)
    
    
    
    # m for masked
    mfuncqgvd, maxqgvd = plt.subplots(figsize=(5, 4))

    plt.scatter(flat_gvd_selected_pixels, masked_flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', label='ktau statistic = ' + str(round(masked_ktau_Q_gvd[0], 2)))
    plt.xlabel(r"Molecular Gas Velocity Dispersion - $\sigma_{g}$ ($\mathrm{km\,s^{-1}}$)", fontsize=14)
    plt.ylabel(r'$Q_{T}$', fontsize=14)
    plt.legend(loc=4)
    plt.xticks(np.arange(0, 35, 10))
    mfuncqgvd.savefig("masked q vs gvd ngc0524", dpi=300, bbox_inches='tight')

    plt.show()    
    
    
       
 
    funcqgvd, axqgvd = plt.subplots(figsize=(8, 8))

    plt.scatter(flat_gvd, flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', label='ktau statistic = ' + str(ktau_Q_gvd[0]))
    plt.xlabel("Gaseous Velocity Dispersion")
    plt.ylabel(r'$Q_{total}$')
    plt.legend(loc=4)
    #plt.xticks(np.arange(0, 20, 5))
    funcqgvd.savefig("q vs gvd ngc0524", dpi=800)

    plt.show()









    mfuncqsvd, maxqsvd = plt.subplots(figsize=(8, 8))

    plt.scatter(flat_svd_selected_pixels, masked_flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', label='ktau statistic = ' + str(masked_ktau_Q_gsd[0]))
    plt.xlabel("Stellar Velocity Dispersion masked")
    plt.ylabel(r'$Q_{total}$ masked')
    plt.legend(loc=4)
    #plt.xticks(np.arange(0, 20, 5))
    mfuncqsvd.savefig("masked q vs svd ngc0524", dpi=800)

    plt.show()



    funcqsvd, axqsvd = plt.subplots(figsize=(8, 8))

    plt.scatter(flat_svd, flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', label='ktau statistic = ' + str(ktau_Q_svd[0]))
    plt.xlabel("Stellar Velocity Dispersion")
    plt.ylabel(r'$Q_{total}$')
    plt.legend(loc=4)
    #plt.xticks(np.arange(0, 20, 5))
    funcqsvd.savefig("q vs svd ngc0524", dpi=800)

    plt.show()








    mfuncqgsd, maxqgsd = plt.subplots(figsize=(5, 4))

    plt.scatter(flat_gsd_selected_pixels, masked_flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', label='Kendall-Tau statistic = ' + str(round(ktau_Q_gsd[0], 2)))
    plt.xlabel(r"Molecular Gas Surface Density - $\Sigma_{g}$ (K $\mathrm{km\,s^{-1}}$)", fontsize=14) 
    plt.ylabel(r'$Q_{T}$', fontsize=14)
    plt.legend(loc=4)
    plt.xticks(np.arange(0, 401, 100))
    mfuncqgsd.savefig("masked q vs gsd ngc0524", dpi=300, bbox_inches='tight')
    plt.show()




    funcqgsd, axqgsd = plt.subplots(figsize=(8, 8))

    plt.scatter(flat_gsd, flat_q, marker='o', color='black', alpha=1, s=1.25, edgecolors='none', label='Kendall-Tau statistic = ' + str(round(ktau_Q_gsd[0], 2)))
    plt.xlabel("Gaseous Surface Density", fontsize=14)
    plt.ylabel(r'$Q_{total}$', fontsize=14)
    plt.legend(loc=4)
    #plt.xticks(np.arange(0, 20, 5))
    funcqgsd.savefig("q vs gsd ngc0524", dpi=300)
    plt.show()
    
    
    
    
    
    
    
    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)

    
    # Define a figure with two subplots side-by-side
    #fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Adjust width to your needs
    plt.subplots_adjust(wspace=0.3) 
    
    # =========================
    # Plot 1: Scatter and Line Plot
    # =========================
    
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.0)  # Adjust size and padding
    
    # Scatter plot
    sc = ax1.scatter(flat_gvd_selected_pixels, masked_flat_q, 
                     c=flat_gc, cmap='inferno', alpha=1, s=5,
                     edgecolors='none', label='Kendall-Tau statistic = ' + 
                     '-0.10' + r" $\pm$ 0.01")  # Example value
    
    # Add colorbar to the new axis
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(r'Galactocentric Radius (kpc)')
    
    
    tick_locator = plt.MultipleLocator(400)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    
    ax1.set_xlabel(r"Molecular gas effective width - $\rm \sigma_{EW}$ ($\rm {km\,s^{-1}}$)", fontsize=14)
    ax1.set_ylabel(r'$\rm Q_{T}$', fontsize=14)
    ax1.legend(loc=1)
    ax1.set_xticks(np.arange(0, 35, 10))
    ax1.tick_params(axis='both', direction='in')


    
    # =========================
    # Plot 2: Image with RA/Dec Coordinates
    # =========================
    # Here we define ax2 as a subplot with WCS projection if needed
    #ax2 = fig1.add_subplot(projection=moment_2.wcs)
    
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.0)  # Adjust size and padding
    
    sc2 = ax2.scatter(flat_gsd_selected_pixels, masked_flat_q,
                c=flat_gc, cmap='inferno', alpha=1, s=5,
                edgecolors='none', label='Kendall-Tau statistic = ' +
                '-0.20' + r" $\pm$ 0.01")
    
    cbar = plt.colorbar(sc2, cax=cax2)
    cbar.set_label(r'Galactocentric Radius (kpc)')
    
    
    tick_locator = plt.MultipleLocator(400)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    ax2.set_xlabel(r"Molecular gas surface density - $\rm \Sigma_{g}$ (K $\rm {km\,s^{-1}}$)", fontsize=14) 
    ax2.set_ylabel(r'$\rm Q_{T}$', fontsize=14)
    ax2.legend(loc=1)
    ax2.set_xticks(np.arange(0, 401, 100))
    ax2.tick_params(axis='both', direction='in')

    # =========================
    # Save and Show the Figure
    # =========================
    # Save both plots in one figure
    fig1.savefig("Combined_correlation_plot.svg", format='svg', dpi=600)
    fig1.savefig("Combined_correlation_plot full.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()
    










    '''
    second half of the arrays
    ''' 


    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)

    
    # Define a figure with two subplots side-by-side
    #fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Adjust width to your needs
    plt.subplots_adjust(wspace=0.3) 
    
    # =========================
    # Plot 1: Scatter and Line Plot
    # =========================
    
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.0)  # Adjust size and padding
    
    # Scatter plot
    sc = ax1.scatter(flat_gvd_selected_pixels[len(flat_gvd_selected_pixels)//2:], masked_flat_q[len(masked_flat_q)//2:], 
                     c=flat_gc[len(flat_gc)//2:], cmap='inferno', alpha=1, s=5,
                     edgecolors='none', label='Kendall-Tau statistic = ' + 
                     str(0.0) + r" $\pm$ 0.02")  # Example value
    
    # Add colorbar to the new axis
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(r'Galactocentric Radius (kpc)')
    
    
    tick_locator = plt.MultipleLocator(200)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    
    ax1.set_xlabel(r"Molecular gas effective width - $\rm \sigma_{EW}$ ($\rm {km\,s^{-1}}$)", fontsize=14)
    ax1.set_ylabel(r'$\rm Q_{T}$', fontsize=14)
    ax1.legend(loc=1)
    ax1.set_xticks(np.arange(0, 35, 10))
    ax1.tick_params(axis='both', direction='in')


    
    # =========================
    # Plot 2: Image with RA/Dec Coordinates
    # =========================
    # Here we define ax2 as a subplot with WCS projection if needed
    #ax2 = fig1.add_subplot(projection=moment_2.wcs)
    
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.0)  # Adjust size and padding
    
    sc2 = ax2.scatter(flat_gsd_selected_pixels[len(flat_gsd_selected_pixels)//2:], masked_flat_q[len(masked_flat_q)//2:],
                c=flat_gc[len(flat_gc)//2:], cmap='inferno', alpha=1, s=5,
                edgecolors='none', label='Kendall-Tau statistic = ' +
                str(0.00) + r" $\pm$ 0.02")
    
    cbar = plt.colorbar(sc2, cax=cax2)
    cbar.set_label(r'Galactocentric Radius (kpc)')
    
    
    tick_locator = plt.MultipleLocator(200)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    ax2.set_xlabel(r"Molecular gas surface density - $\rm \Sigma_{g}$ (K $\rm {km\,s^{-1}}$)", fontsize=14) 
    ax2.set_ylabel(r'$\rm Q_{T}$', fontsize=14)
    ax2.legend(loc=1)
    ax2.set_xticks(np.arange(0, 401, 100))
    ax2.tick_params(axis='both', direction='in')

    # =========================
    # Save and Show the Figure
    # =========================
    # Save both plots in one figure
    fig1.savefig("Combined_correlation_plot.svg", format='svg', dpi=600)
    fig1.savefig("Combined_correlation_plot second half.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()














    '''
    first half of the arrays
    ''' 

    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)

    
    # Define a figure with two subplots side-by-side
    #fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Adjust width to your needs
    plt.subplots_adjust(wspace=0.3) 
    
    # =========================
    # Plot 1: Scatter and Line Plot
    # =========================
    
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.0)  # Adjust size and padding
    
    # Scatter plot
    sc = ax1.scatter(flat_gvd_selected_pixels[:len(flat_gvd_selected_pixels)//2], masked_flat_q[:len(masked_flat_q)//2], 
                     c=flat_gc[:len(flat_gc)//2], cmap='inferno', alpha=1, s=5,
                     edgecolors='none', label='Kendall-Tau statistic = ' + 
                     str(-0.06) + r" $\pm$ 0.02")  # Example value
    
    # Add colorbar to the new axis
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(r'Galactocentric Radius (kpc)')
    
    
    tick_locator = plt.MultipleLocator(200)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    
    ax1.set_xlabel(r"Molecular gas effective width - $\rm \sigma_{EW}$ ($\rm {km\,s^{-1}}$)", fontsize=14)
    ax1.set_ylabel(r'$\rm Q_{T}$', fontsize=14)
    ax1.legend(loc=1)
    ax1.set_xticks(np.arange(0, 35, 10))
    ax1.tick_params(axis='both', direction='in')


    
    # =========================
    # Plot 2: Image with RA/Dec Coordinates
    # =========================
    # Here we define ax2 as a subplot with WCS projection if needed
    #ax2 = fig1.add_subplot(projection=moment_2.wcs)
    
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.0)  # Adjust size and padding
    
    sc2 = ax2.scatter(flat_gsd_selected_pixels[:len(flat_gsd_selected_pixels)//2], masked_flat_q[:len(masked_flat_q)//2],
                c=flat_gc[:len(flat_gc)//2], cmap='inferno', alpha=1, s=5,
                edgecolors='none', label='Kendall-Tau statistic = ' +
                str(-0.13) + r" $\pm$ 0.02")
    
    cbar = plt.colorbar(sc2, cax=cax2)
    cbar.set_label(r'Galactocentric Radius (kpc)')
    
    
    tick_locator = plt.MultipleLocator(200)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    ax2.set_xlabel(r"Molecular gas surface density - $\rm \Sigma_{g}$ (K $\rm {km\,s^{-1}}$)", fontsize=14) 
    ax2.set_ylabel(r'$\rm Q_{T}$', fontsize=14)
    ax2.legend(loc=1)
    ax2.set_xticks(np.arange(0, 401, 100))
    ax2.tick_params(axis='both', direction='in')

    # =========================
    # Save and Show the Figure
    # =========================
    # Save both plots in one figure
    fig1.savefig("Combined_correlation_plot.svg", format='svg', dpi=600)
    fig1.savefig("Combined_correlation_plot first half.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()









    return (two_dplot_gaseous_velocity_dispersion, two_dplot_gaseous_surface_density,
            two_dplot_stellar_velocity_dispersion, two_dplot_stellar_surface_density,
            two_dplot_epicyclic_frequency, flat_gvd, flat_gsd, flat_svd, flat_ssd,
            flat_ef, flat_gcr, ktau_Q_gvd, ktau_Q_svd, ktau_Q_gsd, flat_gvd_selected_pixels,
            flat_gsd_selected_pixels)

     
    
    
    
def plotderivedproducts(position, size_of_cutout, cube, sigma_map_data, moment_1_data, moment_0_data, moment_0, mean_velocity):
    
    image_data = moment_0.data          # Get the image data
    wcs = WCS(moment_0.header)          # Get WCS from the header
 
    # Generate some example images as numpy arrays
    image1 = (sigma_map_data)
    image2 = (moment_1_data - mean_velocity)
    image3 = (moment_0_data)
    
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, subplot_kw={'projection': wcs})




    # Display each image on a different axis with fixed aspect ratio
    im = axes[0].imshow(image1, cmap='inferno', vmin=0, vmax=20, aspect='equal')
    
    ra = axes[0].coords['ra']  # Access RA and Dec coordinates
    dec = axes[0].coords['dec']
    
    ra.set_ticks(number=2, direction='in')  # Set the number of ticks for RA and Dec
    dec.set_ticks(number=3, direction='in')  
    
    textstr = 'NGC0524'
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=14,
                 verticalalignment='top', bbox=props)    
    
    # Set the axis limits (zoom into the desired region)
    axes[0].set_xlim(0, 181)  # Adjust x limits to zoom
    axes[0].set_ylim(0, 181)  # Adjust y limits to zoom

    

    # Create the colorbar inside the plot
    # Adjust 'fraction' and 'pad' as needed
    cbar = plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 10, 20])

    # Optional: Adjust the position of the colorbar
    cbar.ax.set_position([0.1, 0.2, 0.428, 0.025])  # [left, bottom, width, height]











    # Repeat for the other images
    im1 = axes[1].imshow(image2, cmap='RdBu_r', vmin=-180, vmax=180, aspect='equal')
    ra1 = axes[1].coords['ra']
    dec1 = axes[1].coords['dec']
    ra1.set_ticks(number=2, direction='in')
    dec1.set_ticks(number=0, direction='in')
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)    
    
    # Set the axis limits (zoom into the desired region)
    axes[1].set_xlim(0, 181)  # Adjust x limits to zoom
    axes[1].set_ylim(0, 181)  # Adjust y limits to zoom
    

    
    # Create the colorbar inside the plot
    # Adjust 'fraction' and 'pad' as needed
    cbar1 = plt.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar1.set_ticks([-180, 0, 180])

    # Optional: Adjust the position of the colorbar
    cbar1.ax.set_position([0.1, 0.2, 0.8160, 0.025])  # [left, bottom, width, height]




    
    axes[2].set_facecolor("black")

    im2 = axes[2].imshow(image_data, cmap='inferno', vmin=0, vmax=50, aspect='equal')
    ra2 = axes[2].coords['ra']
    dec2 = axes[2].coords['dec']
    ra2.set_ticks(number=2, direction='in', color='white')
    dec2.set_ticks(number=0, direction='in')
    axes[2].text(0.05, 0.95, textstr, transform=axes[2].transAxes, fontsize=14,
        verticalalignment='top', bbox=props) 
    
    # Set the axis limits (zoom into the desired region)
    axes[2].set_xlim(0, 181)  # Adjust x limits to zoom
    axes[2].set_ylim(0, 181)  # Adjust y limits to zoom


    # Create the colorbar inside the plot
    # Adjust 'fraction' and 'pad' as needed
    cbar2 = plt.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar2.set_ticks([0, 25, 50])

    # Optional: Adjust the position of the colorbar
    cbar2.ax.set_position([0.1, 0.2, 1.2116, 0.025])  # [left, bottom, width, height]
    tick_color = "white"  # Choose any color here
    cbar2.ax.tick_params(colors=tick_color)  # Set tick color
    cbar2.ax.xaxis.set_tick_params(color=tick_color)  # Set color for x-axis ticks
    cbar2.ax.set_xticklabels(cbar2.get_ticks(), color=tick_color)  # Set color for labels
    cbar2.outline.set_edgecolor("white")




    axes[0].set_title('Effective width ($\mathrm{km\,s^{-1}}$)', fontsize=axes[0].xaxis.label.get_size(), pad=20)  # Remove "pos.eq.ra" and set custom label for first plot
    axes[1].set_title('Velocity ($\mathrm{km\,s^{-1}}$)', fontsize=axes[1].xaxis.label.get_size(), pad=20)  # Remove "pos.eq.ra" and set custom label for first plot
    axes[2].set_title('Intensity (K $\mathrm{km\,s^{-1}}$)', fontsize=axes[2].xaxis.label.get_size(), pad=20)  # Remove "pos.eq.ra" and set custom label for first plot


    axes[0].set_ylabel(' ')  # Remove "pos.eq.ra" and set custom label for first plot

    axes[0].set_xlabel(' ')  # Remove "pos.eq.ra" and set custom label for first plot
    axes[1].set_xlabel(' ')  # Remove "pos.eq.ra" and set custom label for first plot
    axes[2].set_xlabel(' ')  # Remove "pos.eq.ra" and set custom label for first plot


    plt.subplots_adjust(wspace=-0.5)
    
    

    

    fig.text(0.125, 0.5, 'DEC (J2000)', va='center', rotation='vertical', fontsize=14)  # y-axis label
    fig.text(0.5, -.05, 'RA (J2000)', ha='center', fontsize=14)  # x-axis label

    plt.show()
    fig.savefig("derived products ngc0524.png", format='png', dpi=300, bbox_inches='tight', facecolor='white') 
    





        
    
    
def perturbedktau(flat_gvd_selected_pixels, flat_gsd_selected_pixels, masked_flat_q,
                  masked_flat_q_error, flat_q, flat_q_error):
    

        


    # Define halves
    half = len(flat_gvd_selected_pixels) // 2
    
    # Split arrays into first and second halves
    gvd_first_half, gvd_second_half = flat_gvd_selected_pixels[:half], flat_gvd_selected_pixels[half:]
    gsd_first_half, gsd_second_half = flat_gsd_selected_pixels[:half], flat_gsd_selected_pixels[half:]
    Q_first_half, Q_second_half = flat_q[:half], flat_q[half:]
    Q_error_first_half, Q_error_second_half = flat_q[:half], flat_q[half:]
    
    # Full arrays
    gvd_full, gsd_full = flat_gvd_selected_pixels, flat_gsd_selected_pixels
    Q_full, Q_error_full = flat_q, flat_q_error
    
    num_samples = 100
    
    # Lists to store correlation coefficients and p-values
    perturbed_ktau_Q_gvd_first_list = []
    perturbed_ktau_Q_gvd_second_list = []
    perturbed_ktau_Q_gvd_full_list = []
    
    perturbed_ktau_Q_gsd_first_list = []
    perturbed_ktau_Q_gsd_second_list = []
    perturbed_ktau_Q_gsd_full_list = []
    
    perturbed_pval_Q_gvd_first_list = []
    perturbed_pval_Q_gvd_second_list = []
    perturbed_pval_Q_gvd_full_list = []
    
    perturbed_pval_Q_gsd_first_list = []
    perturbed_pval_Q_gsd_second_list = []
    perturbed_pval_Q_gsd_full_list = []
    
    # Monte Carlo simulation for perturbation
    for i in range(num_samples):
        # Perturb Q values with Gaussian noise
        Q_perturb_first = Q_first_half + np.random.normal(loc=0, scale=Q_error_first_half)
        Q_perturb_second = Q_second_half + np.random.normal(loc=0, scale=Q_error_second_half)
        Q_perturb_full = Q_full + np.random.normal(loc=0, scale=Q_error_full)
    
        # Compute Kendall’s tau for first half
        tau_gvd_first, pval_gvd_first = stats.kendalltau(gvd_first_half, Q_perturb_first, nan_policy='omit', method='auto')
        tau_gsd_first, pval_gsd_first = stats.kendalltau(gsd_first_half, Q_perturb_first, nan_policy='omit', method='auto')
    
        
        # Compute Kendall’s tau for second half
        tau_gvd_second, pval_gvd_second = stats.kendalltau(gvd_second_half, Q_perturb_second, nan_policy='omit', method='auto')
        tau_gsd_second, pval_gsd_second = stats.kendalltau(gsd_second_half, Q_perturb_second, nan_policy='omit', method='auto')
    
    
        # Compute Kendall’s tau for full array
        tau_gvd_full, pval_gvd_full = stats.kendalltau(gvd_full, Q_perturb_full, nan_policy='omit', method='auto')
        tau_gsd_full, pval_gsd_full = stats.kendalltau(gsd_full, Q_perturb_full, nan_policy='omit', method='auto')
        
    
        # Store correlation coefficients
        perturbed_ktau_Q_gvd_first_list.append(tau_gvd_first)
        perturbed_ktau_Q_gvd_second_list.append(tau_gvd_second)
        perturbed_ktau_Q_gvd_full_list.append(tau_gvd_full)
    
        perturbed_ktau_Q_gsd_first_list.append(tau_gsd_first)
        perturbed_ktau_Q_gsd_second_list.append(tau_gsd_second)
        perturbed_ktau_Q_gsd_full_list.append(tau_gsd_full)
    
        # Store p-values
        perturbed_pval_Q_gvd_first_list.append(pval_gvd_first)
        perturbed_pval_Q_gvd_second_list.append(pval_gvd_second)
        perturbed_pval_Q_gvd_full_list.append(pval_gvd_full)
    
        perturbed_pval_Q_gsd_first_list.append(pval_gsd_first)
        perturbed_pval_Q_gsd_second_list.append(pval_gsd_second)
        perturbed_pval_Q_gsd_full_list.append(pval_gsd_full)
        
        
    print("Kendall Tau First Half")
    print(tau_gvd_first, tau_gvd_first)
    print(tau_gsd_first, tau_gsd_first)
    
    print("Kendall Tau Second Half")
    print(tau_gvd_second, tau_gvd_second)
    print(tau_gsd_second, tau_gsd_second)
    

    print("Kendall Tau Full")
    print(tau_gvd_full, tau_gvd_full)
    print(tau_gsd_full, tau_gsd_full)    

    
    
    # Function to compute median, percentiles, and errors
    def compute_statistics(data):
        median = np.nanmedian(data)
        p16 = np.nanpercentile(data, 16)
        p84 = np.nanpercentile(data, 84)
        error = p84 - p16  # Compute error as 84th - 16th percentile
        return median, p16, p84, error
    
    # Compute statistics for Kendall’s tau
    perturbed_gvd_first_median, _, _, perturbed_gvd_first_error = compute_statistics(perturbed_ktau_Q_gvd_first_list)
    perturbed_gvd_second_median, _, _, perturbed_gvd_second_error = compute_statistics(perturbed_ktau_Q_gvd_second_list)
    perturbed_gvd_full_median, _, _, perturbed_gvd_full_error = compute_statistics(perturbed_ktau_Q_gvd_full_list)
    
    perturbed_gsd_first_median, _, _, perturbed_gsd_first_error = compute_statistics(perturbed_ktau_Q_gsd_first_list)
    perturbed_gsd_second_median, _, _, perturbed_gsd_second_error = compute_statistics(perturbed_ktau_Q_gsd_second_list)
    perturbed_gsd_full_median, _, _, perturbed_gsd_full_error = compute_statistics(perturbed_ktau_Q_gsd_full_list)
    
    # Compute statistics for p-values
    perturbed_pval_gvd_first_median, _, _, _ = compute_statistics(perturbed_pval_Q_gvd_first_list)
    perturbed_pval_gvd_second_median, _, _, _ = compute_statistics(perturbed_pval_Q_gvd_second_list)
    perturbed_pval_gvd_full_median, _, _, _ = compute_statistics(perturbed_pval_Q_gvd_full_list)
    
    perturbed_pval_gsd_first_median, _, _, _ = compute_statistics(perturbed_pval_Q_gsd_first_list)
    perturbed_pval_gsd_second_median, _, _, _ = compute_statistics(perturbed_pval_Q_gsd_second_list)
    perturbed_pval_gsd_full_median, _, _, _ = compute_statistics(perturbed_pval_Q_gsd_full_list)
    
    # Print results
    print("Perturbed KTAU GVD - First Half Median:", perturbed_gvd_first_median, "P-Value:", perturbed_pval_gvd_first_median)
    print("Perturbed KTAU GVD - Second Half Median:", perturbed_gvd_second_median, "P-Value:", perturbed_pval_gvd_second_median)
    print("Perturbed KTAU GVD - Full Array Median:", perturbed_gvd_full_median, "P-Value:", perturbed_pval_gvd_full_median)
    
    print("Perturbed KTAU GSD - First Half Median:", perturbed_gsd_first_median, "P-Value:", perturbed_pval_gsd_first_median)
    print("Perturbed KTAU GSD - Second Half Median:", perturbed_gsd_second_median, "P-Value:", perturbed_pval_gsd_second_median)
    print("Perturbed KTAU GSD - Full Array Median:", perturbed_gsd_full_median, "P-Value:", perturbed_pval_gsd_full_median)
    
    print("Perturbed KTAU GVD Error - First Half:", perturbed_gvd_first_error)
    print("Perturbed KTAU GVD Error - Second Half:", perturbed_gvd_second_error)
    print("Perturbed KTAU GVD Error - Full Array:", perturbed_gvd_full_error)
    
    print("Perturbed KTAU GSD Error - First Half:", perturbed_gsd_first_error)
    print("Perturbed KTAU GSD Error - Second Half:", perturbed_gsd_second_error)
    print("Perturbed KTAU GSD Error - Full Array:", perturbed_gsd_full_error)

    
    return (perturbed_gvd_first_median, perturbed_gvd_second_median, perturbed_gvd_full_median,
            perturbed_gsd_first_median, perturbed_gsd_second_median, perturbed_gsd_full_median,
            perturbed_gvd_first_error, perturbed_gvd_second_error, perturbed_gvd_full_error,
            perturbed_gsd_first_error, perturbed_gsd_second_error, perturbed_gsd_full_error,
            perturbed_pval_gvd_first_median, perturbed_pval_gvd_second_median, perturbed_pval_gvd_full_median,
            perturbed_pval_gsd_first_median, perturbed_pval_gsd_second_median, perturbed_pval_gsd_full_median)
    

