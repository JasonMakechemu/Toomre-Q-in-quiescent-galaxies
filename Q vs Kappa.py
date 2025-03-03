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

# Compute nan-median
gsd_medians = nan_median(gsd_arrays)
gvd_medians = nan_median(gvd_arrays)

# Print results
print("GSD NaN-Medians:", gsd_medians)
print("GVD NaN-Medians:", gvd_medians)




# Data dictionary (as before)
#kappa, Q, gsd, gvd, SFR
data = {
    "NGC 0383": (0.56, 005.00, 303.59, 5.96, 0.001, 0.18, 0.54, 3.35, 2.57, 0.2),
    "NGC 0524": (1.21, 034.74, 29.25, 4.58, -0.56, 0.35, 5.64, 0.35, 0.44, 0.2),
    "NGC 1574": (0.74, 003.86, 25.30, 1.44, -0.1, 0.20, 5.18, 1.70, 0.54, 0.2),
    "NGC 3607": (1.17, 022.52, 46.06, 3.54, -0.54, 0.01, 2.77, 0.95, 0.65, 0.2),
    "NGC 4429": (1.93, 008.61, 407.50, 4.42, -0.84, 0.04, 0.63, 2.21, 1.37, 0.2),
    "NGC 4435": (1.61, 112.56, 464.44, 19.84, -0.84, 0.13, 3.02, 0.84, 0.89, 0.2),
    "NGC 4697": (0.90, 033.30, 157.38, 10.00, -1.08, 0.03, 2.13, 0.63, 0.65, 0.2)
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
sfr_err = np.array([point[8] for point in data.values()])

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

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True)

    return pearson_mean, pearson_std, p_value  # Return values


# Create figure with 3 subplots
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

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


results.append(plot_with_mc_subplot(
    axes[3], sfr, Q, sfr_err, Q_err, 
    xlabel=r'Median $\rm SFR$ ($\rm {km\,s^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
))


# Adjust layout
plt.tight_layout()
plt.savefig('three_panel_plot.png', dpi=300)
plt.show()







# Create individual plots for each relation

# Q vs kappa
plt.figure(figsize=(6, 5))
plot_with_mc_subplot(
    plt.gca(), kappa, Q, kappa_err, Q_err, 
    xlabel=r'Median $\kappa$ ($\rm {km\,s^{-1}kpc^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_kappa.png', dpi=300)
plt.show()

# Q vs gsd
plt.figure(figsize=(6, 5))
plot_with_mc_subplot(
    plt.gca(), gsd, Q, gsd_err, Q_err, 
    xlabel=r"Median $\rm \Sigma_{g}$ (K $\rm {km\,s^{-1}}$)", ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_gsd.png', dpi=300)
plt.show()

# Q vs gvd
plt.figure(figsize=(6, 5))
plot_with_mc_subplot(
    plt.gca(), gvd, Q, gvd_err, Q_err, 
    xlabel=r'Median $\rm \sigma_{EW}$ ($\rm {km\,s^{-1}}$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_gvd.png', dpi=300)
plt.show()



# Q vs SFR
plt.figure(figsize=(5, 4))
plot_with_mc_subplot(
    plt.gca(), sfr, Q, sfr_err, Q_err, 
    xlabel=r'Global SFR ($\log(\mathrm{SFR}/\mathrm{M_{\odot}}\,\mathrm{yr}^{-1})$)', ylabel=r'Median $\rm Q_{T}$', labels=labels
)
plt.savefig('q_vs_sfr.png', dpi=300)
plt.show()








# Print Pearson correlation coefficients and p-values
plot_names = ["Q vs gvd", "Q vs gsd", "Q vs kappa", "Q vs SFR"]
for i, (r_mean, r_std, p_val) in enumerate(results):
    print(f"{plot_names[i]}: r = {r_mean:.2f} ± {r_std:.2f}, p-value = {p_val:.4f}")




















