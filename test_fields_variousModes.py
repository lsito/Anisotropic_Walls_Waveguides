#%% Import packages
import anwg # My package

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.signal import find_peaks
import scipy.constants as sc

# Plot in separate window
%matplotlib qt

#%% Test Field distributions with analytical eq from textbook

# We want to check a number of modes 
modes = [[1,0], [2,0], [0,1], [1,1]]
f = [10e9, 14e9, 15.5e9, 17e9]

for mode, f in zip(modes,f):
    m = mode[1]
    n = mode[0]
    
    labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    freq = f
    a=22.86e-3
    b=10.16e-3

    WR90 = anwg.Waveguide(a=a, b=b, zz=0, zt=0, sigma=58e6)
    Dispersion_WR90 = anwg.DispersionCurve(waveguide=WR90, m=m, n=n, freq=freq)
    Fields_WR90 = anwg.Fields(waveguide=WR90, dispersionCurve=Dispersion_WR90, Nx=100, Ny=100)

    Fields_WR90.compute_fields()

    A = [Fields_WR90.Ex, Fields_WR90.Ey, Fields_WR90.Ez, 
        Fields_WR90.Hx, Fields_WR90.Hy, Fields_WR90.Hz]

    norm_factor_A = np.amax(np.amax(np.abs(A)))

    Fields_WR90.compute_fields_analytical(mode='TE')

    B = [Fields_WR90.Ex, Fields_WR90.Ey, Fields_WR90.Ez, 
        Fields_WR90.Hx, Fields_WR90.Hy, Fields_WR90.Hz]

    norm_factor_B = np.amax(np.amax(np.abs(B)))

    # Plotting TME
    fig, ax = plt.subplots(2,3, figsize=(10, 5))

    for idx, el in enumerate(labels):
        row = idx // 3  # Determine the row (0 or 1)
        col = idx % 3   # Determine the column (0, 1, or 2)

        contour = ax[row, col].contourf(Fields_WR90.x, Fields_WR90.y, np.abs(A[idx])/norm_factor_A, cmap='viridis', levels=100)
        fig.colorbar(contour, ax=ax[row, col])

        # Add labels and title
        ax[row, col].set_xlabel('x [m]')
        ax[row, col].set_ylabel('y [m]')
        ax[row, col].set_title(el)

        ax[row, col].set_aspect('equal')

    # Set a title for the whole figure
    fig.suptitle(f'abs TME TE{n}{m}, WR90 @{f/1e9} GHz', fontsize=16)
    # Show the plot
    plt.tight_layout()
    plt.savefig(f"imgs/abs_TME_TE{n}{m}_WR90_{f/1e9}.png")

    # Plotting analytical results
    fig, ax = plt.subplots(2,3, figsize=(10, 5))

    for idx, el in enumerate(labels):
        row = idx // 3  # Determine the row (0 or 1)
        col = idx % 3   # Determine the column (0, 1, or 2)

        contour = ax[row, col].contourf(Fields_WR90.x, Fields_WR90.y, np.abs(B[idx])/norm_factor_B, cmap='viridis', levels=100)
        fig.colorbar(contour, ax=ax[row, col])

        # Add labels and title
        ax[row, col].set_xlabel('x [m]')
        ax[row, col].set_ylabel('y [m]')
        ax[row, col].set_title(el)

        ax[row, col].set_aspect('equal')

    # Set a title for the whole figure
    fig.suptitle(f'abs analytical TE{n}{m}, WR90 @{f/1e9} GHz', fontsize=16)
    # Show the plot
    plt.tight_layout()
    plt.savefig(f"imgs/abs_analytical_TE{n}{m}_WR90_{f/1e9}.png")

# %% Doubt: when there exists a TM mode as well, since it turns on at the same 
# cutoff frequency of the TE, should I consider them overlapped?
    
# We want to check a number of modes
modes = [[0,1], [1,1]]
f = [15.5e9, 17e9]

for mode, f in zip(modes,f):
    m = mode[1]
    n = mode[0]
    
    labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    freq = f
    a=22.86e-3
    b=10.16e-3

    WR90 = anwg.Waveguide(a=a, b=b, zz=0, zt=0, sigma=58e6)
    Dispersion_WR90 = anwg.DispersionCurve(waveguide=WR90, m=m, n=n, freq=freq)
    Fields_WR90 = anwg.Fields(waveguide=WR90, dispersionCurve=Dispersion_WR90, Nx=100, Ny=100)

    Fields_WR90.compute_fields()

    A = [Fields_WR90.Ex, Fields_WR90.Ey, Fields_WR90.Ez, 
        Fields_WR90.Hx, Fields_WR90.Hy, Fields_WR90.Hz]

    norm_factor_A = np.amax(np.amax(np.abs(A)))

    Fields_WR90.compute_fields_analytical(mode='TE')

    B1 = [Fields_WR90.Ex, Fields_WR90.Ey, Fields_WR90.Ez, 
        Fields_WR90.Hx, Fields_WR90.Hy, Fields_WR90.Hz]
    
    Fields_WR90.compute_fields_analytical(mode='TM')

    B2 = [Fields_WR90.Ex, Fields_WR90.Ey, Fields_WR90.Ez, 
        Fields_WR90.Hx, Fields_WR90.Hy, Fields_WR90.Hz]
    
    B = []

    for el1, el2 in zip(B1, B2):
        B.append(el1+el2)

    norm_factor_B = np.amax(np.amax(np.abs(B)))

    # Plotting TME
    fig, ax = plt.subplots(2,3, figsize=(10, 5))

    for idx, el in enumerate(labels):
        row = idx // 3  # Determine the row (0 or 1)
        col = idx % 3   # Determine the column (0, 1, or 2)

        contour = ax[row, col].contourf(Fields_WR90.x, Fields_WR90.y, np.abs(A[idx])/norm_factor_A, cmap='viridis', levels=100)
        fig.colorbar(contour, ax=ax[row, col])

        # Add labels and title
        ax[row, col].set_xlabel('x [m]')
        ax[row, col].set_ylabel('y [m]')
        ax[row, col].set_title(el)

        ax[row, col].set_aspect('equal')

    # Set a title for the whole figure
    fig.suptitle(f'abs TME TE{n}{m}, WR90 @{f/1e9} GHz', fontsize=16)
    # Show the plot
    plt.tight_layout()
    #plt.savefig(f"imgs/abs_TME_TE{n}{m}_WR90_{f/1e9}.png")

    # Plotting analytical results
    fig, ax = plt.subplots(2,3, figsize=(10, 5))

    for idx, el in enumerate(labels):
        row = idx // 3  # Determine the row (0 or 1)
        col = idx % 3   # Determine the column (0, 1, or 2)

        contour = ax[row, col].contourf(Fields_WR90.x, Fields_WR90.y, np.abs(B[idx])/norm_factor_B, cmap='viridis', levels=100)
        fig.colorbar(contour, ax=ax[row, col])

        # Add labels and title
        ax[row, col].set_xlabel('x [m]')
        ax[row, col].set_ylabel('y [m]')
        ax[row, col].set_title(el)

        ax[row, col].set_aspect('equal')

    # Set a title for the whole figure
    fig.suptitle(f'abs analytical TE+TM{n}{m}, WR90 @{f/1e9} GHz', fontsize=16)
    # Show the plot
    plt.tight_layout()
    #plt.savefig(f"imgs/abs_analytical_TE{n}{m}_WR90_{f/1e9}.png")


# %%
