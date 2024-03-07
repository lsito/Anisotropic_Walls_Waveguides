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

#%% Test Dispersion curves with analytical eq from textbook

fig, ax = plt.subplots()

# Geometrical dimension of the waveguide
a=22.86e-3
b=10.16e-3

# We define a simple copper waveguide
WR90 = anwg.Waveguide(a=a, b=b, zz=2*1j, zt=-2*1j, sigma=58e6)

# We want to check a number of modes
modes = [[1,0], [2,0], [0,1], [1,1], [2,1]]

for mode in modes:
    m = mode[1]
    n = mode[0]
    
    # I have to get curves in function of frequency
    freq = np.geomspace(1e6, 3e10, 101)
    
    # Initialize arrays
    k0 = np.zeros_like(freq)
    kc = np.zeros_like(freq)
    gamma = np.zeros_like(freq, dtype = complex)
    eps_eff = np.zeros_like(freq)
    
    # Sweep on all frequency points
    for idx, el in enumerate(freq):
        dispCurveParams = anwg.DispersionCurve(WR90, m=m, n=n, freq=el)

        k0[idx] = dispCurveParams.k0
        kc[idx] = dispCurveParams.kc
        gamma[idx] = dispCurveParams.gamma
        eps_eff[idx] = dispCurveParams.eps_eff
    
    # The analytical dispersion equation for a rectangular waveguide
    ky_a= m*np.pi/b
    kx_a= n*np.pi/a
    
    omega = 2*np.pi*freq
    k0_a = omega*np.sqrt(sc.epsilon_0*sc.mu_0)
    kc_a = np.ones_like(freq)*np.sqrt(kx_a**2 + ky_a**2)
    
    gamma_a = np.sqrt(kc_a**2 - k0_a**2, dtype=complex)
    eps_eff_a = (np.imag(gamma_a)/k0_a)**2
    
    # Plotting mode by mode
    ax.plot(k0*a, eps_eff, marker = "s", markevery=5, label =f"TE{n}{m} Mode Theory")
    ax.plot(k0_a*a, eps_eff_a, marker = "s", markevery=10, label =f"TE{n}{m} Analytical")
    
    ax.set_ylabel('$\epsilon_{eff}$')
    ax.set_xlabel('$k_0 a$')
    
    ax.set_ylim(0, 5)
    
    ax.grid(True, color='gray', linestyle=':')
    ax.legend()
    
plt.show()

# %% Plotting all the dispersion curves
fig, ax = plt.subplots(nrows=4, ncols=4)

# Geometrical dimension of the waveguide
a=22.86e-3
b=10.16e-3

zz_list = [0, 1j/2, 1j, 2*1j]
zt_list = [0, -1j/2, -1j, -2*1j]

for i, zz in enumerate(zz_list):
    # Loop through elements in zt_list
    for j, zt in enumerate(zt_list):
        
        # We define a simple copper waveguide
        WR90 = anwg.Waveguide(a=a, b=b, zz=zz, zt=zt, sigma=58e6)

        # We want to check a number of modes
        modes = [[1,0], [2,0], [0,1], [1,1], [2,1]]

        for mode in modes:
            m = mode[1]
            n = mode[0]
            
            # I have to get curves in function of frequency
            freq = np.geomspace(1e6, 3e10, 501)
            
            # Initialize arrays
            k0 = np.zeros_like(freq)
            kc = np.zeros_like(freq)
            gamma = np.zeros_like(freq, dtype = complex)
            eps_eff = np.zeros_like(freq)
            
            # Sweep on all frequency points
            for idx, el in enumerate(freq):
                dispCurveParams = anwg.DispersionCurve(WR90, m=m, n=n, freq=el)

                k0[idx] = dispCurveParams.k0
                kc[idx] = dispCurveParams.kc
                gamma[idx] = dispCurveParams.gamma
                eps_eff[idx] = dispCurveParams.eps_eff
            
        
            # Plotting mode by mode
            ax[i,j].plot(k0*a, eps_eff, marker = "s", markevery=5, label =f"TE{n}{m} Mode Theory")
            
            ax[i,j].set_ylabel('$\epsilon_{eff}$')
            ax[i,j].set_xlabel('$k_0 a$')
            
            ax[i,j].set_ylim(0, 5)
            ax[i,j].set_xlim(0, 7)
            
            ax[i,j].grid(True, color='gray', linestyle=':')
            ax[i,j].legend()
            
        plt.show()



# %%
