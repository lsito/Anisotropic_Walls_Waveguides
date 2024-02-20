#%% Import packages
import awg # My package

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.signal import find_peaks
import scipy.constants as sc

# Plot in separate window
%matplotlib qt

#%% Test Field distributions with analytical eq from textbook

fig, ax = plt.subplots(2,3, figsize=(10, 5))

labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

freq = 15e9
a=22.86e-3
b=10.16e-3

WR90 = awg.Waveguide(a=a, b=b, zz=0, zt=0, sigma=58e6)
Dispersion_WR90 = awg.DispersionCurve(waveguide=WR90, m=0, n=1, freq=freq)
Fields_WR90 = awg.Fields(waveguide=WR90, dispersionCurve=Dispersion_WR90, Nx=100, Ny=100)
Fields_WR90.compute_fields_analytical(mode='TE')
Fields_WR90.compute_fields()

A = [Fields_WR90.Ex, Fields_WR90.Ey, Fields_WR90.Ez, 
     Fields_WR90.Hx, Fields_WR90.Hy, Fields_WR90.Hz]

norm_factor = np.amax(np.amax(np.abs(A)))

for idx, el in enumerate(labels):
    row = idx // 3  # Determine the row (0 or 1)
    col = idx % 3   # Determine the column (0, 1, or 2)

    contour = ax[row, col].contourf(Fields_WR90.x, Fields_WR90.y, np.imag(A[idx])/norm_factor, cmap='viridis', levels=100)
    fig.colorbar(contour, ax=ax[row, col])

    # Add labels and title
    ax[row, col].set_xlabel('x [m]')
    ax[row, col].set_ylabel('y [m]')
    ax[row, col].set_title(el)

    ax[row, col].set_aspect('equal')

# Set a title for the whole figure
fig.suptitle(f'TE{1}{0}, WR90', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()
# %%
