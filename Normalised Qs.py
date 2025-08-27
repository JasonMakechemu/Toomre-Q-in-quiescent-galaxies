#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:29:19 2024

@author: jason
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Data loading (Replace paths with actual locations)
flat_Q_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC0383.npy')
flat_Q_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC0524.npy')
flat_Q_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC1574.npy')
flat_Q_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC3607.npy')
flat_Q_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC4429.npy')
flat_Q_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC4435.npy')
flat_Q_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_Q_NGC4697.npy')

flat_Q_error_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC0383.npy')
flat_Q_error_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC0524.npy')
flat_Q_error_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC1574.npy')
flat_Q_error_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC3607.npy')
flat_Q_error_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC4429.npy')
flat_Q_error_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC4435.npy')
flat_Q_error_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_q_error_NGC4697.npy')

flat_Q_gas_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc0383.npy')
flat_Q_gas_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc0524.npy')
flat_Q_gas_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc1574.npy')
flat_Q_gas_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc3607.npy')
flat_Q_gas_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc4429.npy')
flat_Q_gas_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc4435.npy')
flat_Q_gas_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_qgas_ngc4697.npy')

flat_Q_star_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc0383.npy')
flat_Q_star_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc0524.npy')
flat_Q_star_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc1574.npy')
flat_Q_star_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc3607.npy')
flat_Q_star_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc4429.npy')
flat_Q_star_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc4435.npy')
flat_Q_star_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_ngc4697.npy')



# Define the arrays in groups
all_q_arrays = [
    'flat_Q_NGC0383', 'flat_Q_NGC0524', 'flat_Q_NGC1574',
    'flat_Q_NGC3607', 'flat_Q_NGC4429', 'flat_Q_NGC4435', 'flat_Q_NGC4697',
    'flat_Q_gas_NGC0383', 'flat_Q_gas_NGC0524', 'flat_Q_gas_NGC1574',
    'flat_Q_gas_NGC3607', 'flat_Q_gas_NGC4429', 'flat_Q_gas_NGC4435', 'flat_Q_gas_NGC4697',
    'flat_Q_star_NGC0383', 'flat_Q_star_NGC0524', 'flat_Q_star_NGC1574',
    'flat_Q_star_NGC3607', 'flat_Q_star_NGC4429', 'flat_Q_star_NGC4435', 'flat_Q_star_NGC4697'
]

# Convert 0s to np.nan in-place
for name in all_q_arrays:
    arr = globals()[name]
    arr[arr == 0] = np.nan




# Number of galaxies
galaxy_count = 7

# Normalized radii
radii = [
    np.linspace(0, 1290.12, 10835),  # NGC0383
    np.linspace(0, 903.599, 7124),  # NGC0524
    np.linspace(0, 231.511, 1945),  # NGC1574
    np.linspace(0, 1975.41, 5108),  # NGC3607
    np.linspace(0, 423.9, 7623),  # NGC4429
    np.linspace(0, 419.528, 466),  # NGC4435
    np.linspace(0, 408.595, 294)  # NGC4697
]

radii_gas = [
    np.linspace(0, 1290.12, 5137),  # NGC0383
    np.linspace(0, 903.599, 2838),  # NGC0524
    np.linspace(0, 231.511, 1945),  # NGC1574
    np.linspace(0, 1975.41, 5108),  # NGC3607
    np.linspace(0, 423.9, 3093),  # NGC4429
    np.linspace(0, 419.528, 437),  # NGC4435
    np.linspace(0, 408.595, 294)  # NGC4697
]


radii_star = [
    np.linspace(0, 1290.12, 6256),  # NGC0383
    np.linspace(0, 903.599, 3355),  # NGC0524
    np.linspace(0, 231.511, 1945),  # NGC1574
    np.linspace(0, 1975.41, 5108),  # NGC3607
    np.linspace(0, 423.9, 3234),  # NGC4429
    np.linspace(0, 419.528, 466),  # NGC4435
    np.linspace(0, 408.595, 294)  # NGC4697
]

# Sphere of influence for each galaxy
#r_soi = [11*322.89, 23.66*112.96, 21.01*96.48, 21.9*107.63, 48.84*79.99, 28.49*80.96, 39.51*55.27]
r_disk = [1290.12, 903.599, 231.511, 1975.41, 423.9, 419.528, 408.595]
#r_soi = [11, 23.66, 21.01, 21.9, 48.84, 28.49, 39.51]



# Q values for each galaxy
q_values_total = [flat_Q_NGC0383, flat_Q_NGC0524, flat_Q_NGC1574, flat_Q_NGC3607, flat_Q_NGC4429, flat_Q_NGC4435, flat_Q_NGC4697]
q_values_error_total = [flat_Q_error_NGC0383, flat_Q_error_NGC0524, flat_Q_error_NGC1574, flat_Q_error_NGC3607, flat_Q_error_NGC4429, flat_Q_error_NGC4435, flat_Q_error_NGC4697]
q_values_gas = [flat_Q_gas_NGC0383, flat_Q_gas_NGC0524, flat_Q_gas_NGC1574, flat_Q_gas_NGC3607, flat_Q_gas_NGC4429, flat_Q_gas_NGC4435, flat_Q_gas_NGC4697]
q_values_star = [flat_Q_star_NGC0383, flat_Q_star_NGC0524, flat_Q_star_NGC1574, flat_Q_star_NGC3607, flat_Q_star_NGC4429, flat_Q_star_NGC4435, flat_Q_star_NGC4697]



# Replace 0s with np.nan in each array in the lists
q_values_total = [np.where(arr == 0, np.nan, arr) for arr in q_values_total]
q_values_error_total = [np.where(arr == 0, np.nan, arr) for arr in q_values_error_total]
q_values_gas = [np.where(arr == 0, np.nan, arr) for arr in q_values_gas]
q_values_star = [np.where(arr == 0, np.nan, arr) for arr in q_values_star]


# Remove NaNs from each array in the lists
q_values_total = [arr[~np.isnan(arr)] for arr in q_values_total]
q_values_error_total = [arr[~np.isnan(arr)] for arr in q_values_error_total]
q_values_gas = [arr[~np.isnan(arr)] for arr in q_values_gas]
q_values_star = [arr[~np.isnan(arr)] for arr in q_values_star]

galaxy_names = ['NGC0383', 'NGC0524', 'NGC1574', 'NGC3607', 'NGC4429', 'NGC4435', 'NGC4697']



'''
We have Q gas a Q total values across the entire extent of the disk, however, because we are taking medians in bins,
the median is sometimes nan or 0, so we remove nans from each array in the list
'''


# Plot settings
num_bins = 20  # Number of bins for the radius

# Function to create the plot
def plot_q_values(q_values, title, radii_list):
    plt.figure(figsize=(10, 8))

    for i in range(galaxy_count):
        if title.lower().startswith("star"):
            q_values[i] = np.abs(q_values[i])  # Absolute value for stars if needed

        # Normalize using correct radius array
        normalized_radius = radii_list[i] / r_disk[i]

        # Trim to the shortest length
        min_length = min(len(normalized_radius), len(q_values[i]))
        normalized_radius = normalized_radius[:min_length]
        q_values[i] = q_values[i][:min_length]

        # Create bins for radius
        bins = np.linspace(0, max(normalized_radius), num_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate percentiles for each bin
        q_16th, q_84th, q_medians = [], [], []
        for j in range(len(bins) - 1):
            mask = (normalized_radius >= bins[j]) & (normalized_radius < bins[j + 1])
            if np.any(mask):
                q_16th.append(np.percentile(q_values[i][mask], 16))
                q_84th.append(np.percentile(q_values[i][mask], 84))
                q_medians.append(np.median(q_values[i][mask]))
            else:
                q_16th.append(np.nan)
                q_84th.append(np.nan)
                q_medians.append(np.nan)

        plt.fill_between(bin_centers, q_16th, q_84th, alpha=0.3)
        plt.plot(bin_centers, q_medians, label=galaxy_names[i])

    plt.tick_params(axis='both', which='major', direction='in', length=10, width=2, labelsize=16, top=True, right=True)
    plt.tick_params(axis='both', which='minor', direction='in', length=5, width=1.5, top=True, right=True)
    plt.xscale('log')
    plt.yscale('log')

    plt.axhline(y=1, color='grey', linestyle='--', label='Q = 1')
    plt.xlabel(r'log$_{10}(R/R_{\rm disc}$)', fontsize=18)
    
    if title.lower().startswith("gas"):
        plt.ylabel(r'$Q_{\rm g}$', fontsize=18, labelpad=20)
    elif title.lower().startswith("star"):
        plt.ylabel(r'$Q_{\ast}$', fontsize=18, labelpad=20)
    else:
        plt.ylabel(r'$Q_{\rm T}$', fontsize=18, labelpad=20)
        
    plt.legend(loc=3, bbox_to_anchor=(0.015, 0.01), fontsize=12)
    plt.grid()
    
    plt.savefig(f"{title}_Normalized_Qs_with_percentiles.png", dpi=300, format='png', bbox_inches='tight')
    plt.show()

    

# Generate plots
plot_q_values(q_values_total, "Total_Q", radii)
plot_q_values(q_values_gas, "Gas_Q", radii_gas)
plot_q_values(q_values_star, "Star_Q", radii_star)



# Compute medians
medians_total = [np.nanmedian(q) for q in q_values_total]
medians_gas = [np.nanmedian(q) for q in q_values_gas]
medians_star = [np.nanmedian(q) for q in q_values_star]

# Print the median values
print("Median Q values for each galaxy:")
for i, name in enumerate(galaxy_names):
    print(f"{name}: Q_total = {medians_total[i]:.2f}, Q_gas = {medians_gas[i]:.2f}, Q_star = {medians_star[i]:.2f}")


for i in range(len(q_values_total)):
    print(np.nanmedian(q_values_star[i]))


for i in range(len(q_values_total)):
    print(np.nanmedian(q_values_error_total[i]))


from scipy import stats
from scipy.stats import gaussian_kde

def variance_from_median(arr):
    """Calculates the variance from the median of an array using KDE."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]  # Remove NaNs
    n = len(arr)
    
    if n == 0:
        print("Empty array or only NaNs.")
        return None

    median = np.median(arr)
    mad = stats.median_abs_deviation(arr)

    kde = gaussian_kde(arr)
    pdf_at_median = kde(median)
    
    variance = 1 / (4 * n * pdf_at_median ** 2)
    std_dev = np.sqrt(variance)

    print(f"Variance: {variance[0]:.6f}, Std Dev: {std_dev[0]:.6f}")
    return variance[0], std_dev[0]

# Lists of Q values for each component
galaxy_names = ["NGC0383", "NGC0524", "NGC1574", "NGC3607", "NGC4429", "NGC4435", "NGC4697"]

print("Total Q values:")
for name, q_array in zip(galaxy_names, q_values_total):
    print(f"{name}: ", end="")
    variance_from_median(q_array)

print("\nGas Q values:")
for name, q_array in zip(galaxy_names, q_values_gas):
    print(f"{name}: ", end="")
    variance_from_median(q_array)

print("\nStar Q values:")
for name, q_array in zip(galaxy_names, q_values_star):
    print(f"{name}: ", end="")
    variance_from_median(q_array)























