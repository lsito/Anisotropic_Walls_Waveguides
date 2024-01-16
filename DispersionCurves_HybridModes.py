# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:16:59 2024

@author: Leonardo Sito
"""

#%% Importing Modules

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.signal import find_peaks
import scipy.constants as sc

#%% Definition of global parmeters for testing

# WR90
a = 22.86e-3 # Long side of waveguide
b = 10.16e-3 # Short side of waveguide

zz = 0
zt = 0

#%% Compute kc for m=0

m = 0
n = 3 # Here I can choose only n

# Dpersion relation
def f_of_kc_me0(kc, freq):

    omega = 2*np.pi*freq
    k0 = omega*np.sqrt(sc.epsilon_0*sc.mu_0)
    X = np.exp(-1j*2*kc*a, dtype = complex)
    
    f = np.abs(zt**2*(1-X) - 2*k0/kc*zt*(1+X) + (k0/kc)**2 * (1-X))

    return f

def compute_kc_me0(a, b, zz, zt, n, freq):

    x = np.linspace(1, 1000, 10000)
    
    # Find the zeros as peaks: n.b. The "-" is mandatory, since it defines as peaks
    # the zeros (thanks to the abs)
    
    roots_idx = find_peaks(-f_of_kc_me0(x, freq))[0]
    roots = x[roots_idx]
    
    kc = roots[n-1]
  
    return kc
  
m = 0
n = 1 # Mode number

freq = np.geomspace(1e6, 2e10, 1001)
kc = np.zeros_like(freq)

for idx, el in enumerate(freq):
    kc[idx] = compute_kc_me0(a, b, zz, zt, n, el)
    
# Final computations for dispersion curves
k0 = 2*np.pi*freq*np.sqrt(sc.epsilon_0*sc.mu_0)

gamma = np.sqrt(kc**2 - k0**2, dtype=complex)

eps_eff = (np.imag(gamma)/k0)**2

#%% Plotting just to be sure
fig, ax = plt.subplots()
ax.plot(k0*a, eps_eff, marker = "s", markevery=20, label =f"TE{n}0")
ax.set_ylabel('$\epsilon_{eff}$')
ax.set_xlabel('$k_0 a$')

ax.set_ylim(0, 5)

ax.grid(True, color='gray', linestyle=':')
ax.legend()

plt.show()
