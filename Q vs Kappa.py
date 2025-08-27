#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:19:39 2025

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
import scipy.stats as stats


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



# Data loading (Replace paths with actual locations)
kappa_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC0383.npy')
kappa_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC0524.npy')
kappa_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC1574.npy')
kappa_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC3607.npy')
kappa_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC4429.npy')
kappa_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC4435.npy')
kappa_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_NGC4697.npy')

kappa_error_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC0383.npy')
kappa_error_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC0524.npy')
kappa_error_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC1574.npy')
kappa_error_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC3607.npy')
kappa_error_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC4429.npy')
kappa_error_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC4435.npy')
kappa_error_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/kappa_error_NGC4697.npy')

# Data loading
flat_gsd_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC0383.npy')
flat_gsd_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC0524.npy')
flat_gsd_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC1574.npy')
flat_gsd_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC3607.npy')
flat_gsd_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC4429.npy')
flat_gsd_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC4435.npy')
flat_gsd_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gsd_NGC4697.npy')


# Data loading
flat_gvd_NGC0383 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC0383.npy')
flat_gvd_NGC0524 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC0524.npy')
flat_gvd_NGC1574 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC1574.npy')
flat_gvd_NGC3607 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC3607.npy')
flat_gvd_NGC4429 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC4429.npy')
flat_gvd_NGC4435 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC4435.npy')
flat_gvd_NGC4697 = np.load('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/flat_gvd_NGC4697.npy')



# Function to compute nan-median
def nan_median(arrays):
    return {name: np.nanmedian(array) for name, array in arrays.items()}


# Dictionary of all loaded arrays

kappa_arrays = {
    "NGC0383": kappa_NGC0383,
    "NGC0524": kappa_NGC0524,
    "NGC1574": kappa_NGC1574,
    "NGC3607": kappa_NGC3607,
    "NGC4429": kappa_NGC4429,
    "NGC4435": kappa_NGC4435,
    "NGC4697": kappa_NGC4697,
}


q_arrays = {
    "NGC0383": flat_Q_NGC0383,
    "NGC0524": flat_Q_NGC0524,
    "NGC1574": flat_Q_NGC1574,
    "NGC3607": flat_Q_NGC3607,
    "NGC4429": flat_Q_NGC4429,
    "NGC4435": flat_Q_NGC4435,
    "NGC4697": flat_Q_NGC4697,
}


gsd_arrays = {
    "NGC0383": flat_gsd_NGC0383,
    "NGC0524": flat_gsd_NGC0524,
    "NGC1574": flat_gsd_NGC1574,
    "NGC3607": flat_gsd_NGC3607,
    "NGC4429": flat_gsd_NGC4429,
    "NGC4435": flat_gsd_NGC4435,
    "NGC4697": flat_gsd_NGC4697,
}

gvd_arrays = {
    "NGC0383": flat_gvd_NGC0383,
    "NGC0524": flat_gvd_NGC0524,
    "NGC1574": flat_gvd_NGC1574,
    "NGC3607": flat_gvd_NGC3607,
    "NGC4429": flat_gvd_NGC4429,
    "NGC4435": flat_gvd_NGC4435,
    "NGC4697": flat_gvd_NGC4697,
}


kappa_error_arrays = {
    "NGC0383": kappa_error_NGC0383,
    "NGC0524": kappa_error_NGC0524,
    "NGC1574": kappa_error_NGC1574,
    "NGC3607": kappa_error_NGC3607,
    "NGC4429": kappa_error_NGC4429,
    "NGC4435": kappa_error_NGC4435,
    "NGC4697": kappa_error_NGC4697,
}


q_error_arrays = {
    "NGC0383": flat_Q_error_NGC0383,
    "NGC0524": flat_Q_error_NGC0524,
    "NGC1574": flat_Q_error_NGC1574,
    "NGC3607": flat_Q_error_NGC3607,
    "NGC4429": flat_Q_error_NGC4429,
    "NGC4435": flat_Q_error_NGC4435,
    "NGC4697": flat_Q_error_NGC4697,
}


# Function to replace 0s with NaN in respective arrays array
def replace_zeros_with_nan(array):
    array[array == 0] = np.nan
    array[array == 0.1] = np.nan

    return array

# Apply the above function to all dictionaries
gsd_arrays = {key: replace_zeros_with_nan(arr) for key, arr in gsd_arrays.items()}
gvd_arrays = {key: replace_zeros_with_nan(arr) for key, arr in gvd_arrays.items()}
q_arrays = {key: replace_zeros_with_nan(arr) for key, arr in q_arrays.items()}
kappa_arrays = {key: replace_zeros_with_nan(arr) for key, arr in kappa_arrays.items()}

kappa_error_arrays = {key: replace_zeros_with_nan(arr) for key, arr in kappa_error_arrays.items()}
q_error_arrays = {key: replace_zeros_with_nan(arr) for key, arr in q_error_arrays.items()}


# Compute nan-median of each array.
gsd_medians = nan_median(gsd_arrays)
gvd_medians = nan_median(gvd_arrays)
q_medians = nan_median(q_arrays)
kappa_medians = nan_median(kappa_arrays)


kappa_error_medians = nan_median(kappa_error_arrays)
q_error_medians = nan_median(q_error_arrays)



# Print results
print("Kappa NaN-Medians:", kappa_medians)
print("Q NaN-Medians:", q_medians)
print("GSD NaN-Medians:", gsd_medians)
print("GVD NaN-Medians:", gvd_medians)

print("Kappa error NaN-Medians:", kappa_error_medians)
print("Q error NaN-Medians:", q_error_medians)



# Data dictionary (as before)
#median kappa, Q, gsd, gvd, SFR - and the respective associated errors
data = {
    "NGC 0383": (1.15, 002.16, 315.58, 06.11, 0.001, 0.32, 1.29, 3.35, 2.57, 0.2),
    "NGC 0524": (1.36, 018.47, 031.22, 04.76, -0.56, 0.45, 6.45, 0.35, 0.44, 0.2),
    "NGC 1574": (4.56, 005.42, 035.56, 01.41, -0.10, 0.97, 1.26, 1.70, 0.54, 0.2),
    "NGC 3607": (1.30, 009.98, 046.06, 03.54, -0.54, 0.02, 2.22, 0.95, 0.65, 0.2),
    "NGC 4429": (2.49, 002.02, 451.56, 04.79, -0.84, 0.04, 0.64, 2.21, 1.37, 0.2),
    "NGC 4435": (3.14, 038.38, 592.91, 21.46, -0.84, 0.22, 7.87, 0.84, 0.89, 0.2),
    "NGC 4697": (1.79, 017.12, 149.40, 10.18, -1.08, 0.05, 2.33, 0.63, 0.65, 0.2)
}






# Extract values and errors
kappa = np.array([point[0] for point in data.values()])
Q = np.array([point[1] for point in data.values()])
gsd = np.array([point[2] for point in data.values()])
gvd = np.array([point[3] for point in data.values()])
sfr = np.array([point[4] for point in data.values()])


kappa_err = np.array([point[5] for point in data.values()])
Q_err = np.array([point[6] for point in data.values()])
gsd_err = np.array([point[7] for point in data.values()])
gvd_err = np.array([point[8] for point in data.values()])
sfr_err = np.array([point[9] for point in data.values()])

labels = list(data.keys())



# Function to create scatter plots with Monte Carlo Pearson correlation coefficients
def plot_with_mc_subplot(ax, x, y, x_err, y_err, xlabel, ylabel, labels):
    num_samples = 10_000
    pearson_samples = []

    # Monte Carlo sampling
    for _ in range(num_samples):
        x_sampled = np.random.normal(x, x_err)
        y_sampled = np.random.normal(y, y_err)
        r_sampled = np.corrcoef(x_sampled, y_sampled)[0, 1]
        pearson_samples.append(r_sampled)

    # Compute mean and standard deviation of Pearson coefficients
    pearson_mean = np.mean(pearson_samples)
    pearson_std = np.std(pearson_samples)

    # Fit a linear regression line to the original data
    coeffs = np.polyfit(x, y, 1)
    poly_eq = np.poly1d(coeffs)
    y_fit = poly_eq(x)

    pearson = f'r = {pearson_mean:.2f} ± {pearson_std:.2f}'

    # Statistical significance (p-value)
    n = len(x)
    t_stat = pearson_mean * np.sqrt(n - 2) / np.sqrt(1 - pearson_mean**2)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))  # Two-tailed test

    # Plot in the given axis
    sns.scatterplot(ax=ax, x=x, y=y, color='blue', label='Galaxies')
    ax.plot(x, y_fit, color='red', label=f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    sns.scatterplot(ax=ax, x=x, y=y, color='white', label=pearson)

    # Error bars
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='blue', alpha=0.5, capsize=3)

    # Add labels to points
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], label, fontsize=8, verticalalignment='top', horizontalalignment='left')

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True)

    return pearson_mean, pearson_std, p_value  # Return values


# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot each relation in a separate subplot and store results
results = []

results.append(plot_with_mc_subplot(
    axes[0], gvd, Q, gvd_err, Q_err, 
    xlabel=r'Median $\rm \sigma_{EW}$ ($\rm {km\,s^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
))

results.append(plot_with_mc_subplot(
    axes[1], gsd, Q, gsd_err, Q_err, 
    xlabel=r"Median $\rm \Sigma_{g}$ (K $\rm {km\,s^{-1}}$)", ylabel=r'Median $\rm Q_{T}$', labels=labels
))



results.append(plot_with_mc_subplot(
    axes[2], kappa, Q, kappa_err, Q_err, 
    xlabel=r'Median $\kappa$ ($\rm {km\,s^{-1}kpc^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
))


'''
results.append(plot_with_mc_subplot(
    axes[3], sfr, Q, sfr_err, Q_err, 
    xlabel=r'Median $\rm SFR$ ($\rm {km\,s^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
))
'''

# Adjust layout
plt.tight_layout()
plt.savefig('three_panel_plot.png', dpi=300, bbox_inches='tight')
plt.show()







# Create individual plots for each relation

# Q vs kappa
plt.figure(figsize=(5, 4))
plot_with_mc_subplot(
    plt.gca(), kappa, Q, kappa_err, Q_err, 
    xlabel=r'Median $\kappa$ ($\rm {km\,s^{-1}kpc^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_kappa.png', dpi=300, bbox_inches='tight')
plt.show()

# Q vs gsd
plt.figure(figsize=(5, 4))
plot_with_mc_subplot(
    plt.gca(), gsd, Q, gsd_err, Q_err, 
    xlabel=r"Median $\rm \Sigma_{g}$ (K $\rm {km\,s^{-1}}$)", ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_gsd.png', dpi=300, bbox_inches='tight')
plt.show()

# Q vs gvd
plt.figure(figsize=(5, 4))
plot_with_mc_subplot(
    plt.gca(), gvd, Q, gvd_err, Q_err, 
    xlabel=r'Median $\rm \sigma_{EW}$ ($\rm {km\,s^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_gvd.png', dpi=300, bbox_inches='tight')
plt.show()


# Q vs SFR
plt.figure(figsize=(5, 4))
plot_with_mc_subplot(
    plt.gca(), sfr, Q, sfr_err, Q_err, 
    xlabel=r'Global SFR ($\log(\mathrm{SFR}/\mathrm{M_{\odot}}\,\mathrm{yr}^{-1})$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_sfr.png', dpi=300, bbox_inches='tight')
plt.show()



# Print Pearson correlation coefficients and p-values
plot_names = ["Q vs gvd", "Q vs gsd", "Q vs kappa", "Q vs SFR"]
for i, (r_mean, r_std, p_val) in enumerate(results):
    print(f"{plot_names[i]}: r = {r_mean:.2f} ± {r_std:.2f}, p-value = {p_val:.4f}")


















