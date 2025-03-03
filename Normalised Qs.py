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

flat_Q_gas_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC0383.npy')
flat_Q_gas_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC0524.npy')
flat_Q_gas_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC1574.npy')
flat_Q_gas_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC3607.npy')
flat_Q_gas_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC4429.npy')
flat_Q_gas_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC4435.npy')
flat_Q_gas_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qgas_NGC4697.npy')

flat_Q_star_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC0383.npy')
flat_Q_star_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC0524.npy')
flat_Q_star_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC1574.npy')
flat_Q_star_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC3607.npy')
flat_Q_star_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC4429.npy')
flat_Q_star_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC4435.npy')
flat_Q_star_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/Flat_qstar_NGC4697.npy')


#%%
# Number of galaxies
galaxy_count = 7

# Normalized radii
radii = [
    np.linspace(0, 5335.7, 15193),  # NGC0383
    np.linspace(0, 1560.55, 8102),  # NGC0524 (Fixed)
    np.linspace(0, 1370.55, 29057),  # NGC1574
    np.linspace(0, 1975.41, 5108),  # NGC3607
    np.linspace(0, 1269.06, 10779),  # NGC4429
    np.linspace(0, 2592.98, 788),  # NGC4435
    np.linspace(0, 2843.53, 432)  # NGC4697
]

# Sphere of influence for each galaxy
#r_soi = [11*322.89, 23.66*112.96, 21.01*96.48, 21.9*107.63, 48.84*79.99, 28.49*80.96, 39.51*55.27]
r_disk = [5335.7, 1560.55, 1370.55, 1975.41, 1269.06, 2592.98, 2843.53]
r_soi = [11, 23.66, 21.01, 21.9, 48.84, 28.49, 39.51]



# Q values for each galaxy
q_values_total = [flat_Q_NGC0383, flat_Q_NGC0524, flat_Q_NGC1574, flat_Q_NGC3607, flat_Q_NGC4429, flat_Q_NGC4435, flat_Q_NGC4697]
q_values_gas = [flat_Q_gas_NGC0383, flat_Q_gas_NGC0524, flat_Q_gas_NGC1574, flat_Q_gas_NGC3607, flat_Q_gas_NGC4429, flat_Q_gas_NGC4435, flat_Q_gas_NGC4697]
q_values_star = [flat_Q_star_NGC0383, flat_Q_star_NGC0524, flat_Q_star_NGC1574, flat_Q_star_NGC3607, flat_Q_star_NGC4429, flat_Q_star_NGC4435, flat_Q_star_NGC4697]


galaxy_names = ['NGC0383', 'NGC0524', 'NGC1574', 'NGC3607', 'NGC4429', 'NGC4435', 'NGC4697']

# Plot settings
num_bins = 25  # Number of bins for the radius

# Function to create the plot
def plot_q_values(q_values, title):
    plt.figure(figsize=(10, 6))

    for i in range(galaxy_count):
        q_values_star[i] = np.abs(q_values_star[i])

        # Normalize radii by sphere of influence
        normalized_radius = radii[i] / r_disk[i]

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

    plt.tick_params(axis='both', which='major', direction='in', length=10, width=2, labelsize=16)
    plt.tick_params(axis='both', which='minor', direction='in', length=5, width=1.5)
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(0, 100)

    plt.axhline(y=1, color='grey', linestyle='--', label='Q = 1')
    plt.xlabel(r'$log_{10}(R/R_{\rm disk}$)', fontsize=18)
    plt.ylabel(r'$Q_{\rm \ast}$', fontsize=18)
    plt.legend(loc=2, fontsize=10)
    plt.grid()
    
    plt.savefig(f"{title}_Normalized_Qs_with_percentiles.png", dpi=300, format='png', bbox_inches='tight')
    plt.show()

# Generate plots
plot_q_values(q_values_total, "Total_Q")
plot_q_values(q_values_gas, "Gas_Q")
plot_q_values(q_values_star, "Star_Q")




# Compute medians
medians_total = [np.median(q) for q in q_values_total]
medians_gas = [np.median(q) for q in q_values_gas]
medians_star = [np.median(q) for q in q_values_star]

# Print the median values
print("Median Q values for each galaxy:")
for i, name in enumerate(galaxy_names):
    print(f"{name}: Q_total = {medians_total[i]:.2f}, Q_gas = {medians_gas[i]:.2f}, Q_star = {medians_star[i]:.2f}")










