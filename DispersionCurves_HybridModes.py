# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:16:59 2024

@author: Leonardo Sito
"""

#%% Importing Modules

# Plot in separate window
%matplotlib qt

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

#%% Dispersion relation m = 0
def f_of_kc_me0(kc, freq):

    omega = 2*np.pi*freq
    k0 = omega*np.sqrt(sc.epsilon_0*sc.mu_0)
    X = np.exp(-1j*2*kc*a, dtype = complex)
    
    f = np.abs(zt**2*(1-X) - 2*k0/kc*zt*(1+X) + (k0/kc)**2 * (1-X))

    return f

#%% Dispersion relation m != 0
def f_of_kc_mne0(kc, m, freq):

        omega = 2*np.pi*freq
        k0 = omega*np.sqrt(sc.epsilon_0*sc.mu_0)

        ky = m*np.pi/b
        kx = np.sqrt(kc**2-ky**2, dtype = complex)
        gamma = np.sqrt(kc**2-k0**2, dtype = complex)

        X = np.exp(-1j*2*kx*a, dtype = complex)
        
        K0xc = k0*kx/kc**2
        Kzyc = gamma*ky/kc**2

        f = np.abs(
            2*Kzyc**2*zz*(-zt+(X+1)/(1-X)*K0xc*(zz*zt+1)-K0xc**2*zz)
            + (Kzyc**4+K0xc**4)*zz**2-2*K0xc**3*(X+1)/(1-X)*zz*(zz*zt+1)
            + K0xc**2*(1+(zz*zt)**2+4*((X+1)/(1-X))**2*zz*zt)
            - 2*K0xc*(X+1)/(1-X)*zt*(1+zz*zt)
            + zt**2
            )

        return f

#%% Compute alpha of hybrid modes
# Only once we have kc we can get alhpa coupling
"""
def compute_alpha(a, b, zz, zt, m, n, freq):
    
        Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
        omega = 2*np.pi*freq
        k0 = omega*np.sqrt(sc.epsilon_0*sc.mu_0)

        ky = m*np.pi/b
        kx = np.sqrt(kc**2-ky**2, dtype = complex)
        gamma = np.sqrt(kc**2-k0**2, dtype = complex)

        X = np.exp(-1j*2*kx*a, dtype = complex)
        
        K0xc = k0*kx/kc**2
        Kzyc = gamma*ky/kc**2
        
        if zt == 0:
            alpha = -Kzyc/K0xc
        else:
            alpha = (K0xc*Kzyc*(zz-1/zt))/(-2*(X+1)/(1-X)*K0xc+zt-zz*Kzyc**2+K0xc**2/zt)/Z0
            
        return alpha
"""
#%% Root finding of the dispersion equation and support variables
def compute_kc(a, b, zz, zt, m, n, freq):
    if m == 0:
        x = np.linspace(1, 1000, 10000)
        roots_idx = find_peaks(-f_of_kc_me0(x, freq))[0]
    else:
        x = np.linspace(1, 600, 20000)
        roots_idx = find_peaks(-f_of_kc_mne0(x, m, freq))[0]
    
    roots = x[roots_idx]
    kc_ = roots[n-1]
    
    k0_ = 2*np.pi*freq*np.sqrt(sc.epsilon_0*sc.mu_0)
    gamma_ = np.sqrt(kc_**2 - k0_**2, dtype=complex)
    eps_eff_ = (np.imag(gamma_)/k0_)**2

    return k0_, kc_, gamma_, eps_eff_

  
#%% Test Dispersion curves
m = 0
n = 1 # Mode number

freq = np.geomspace(1e6, 2e10, 1001)
k0 = np.zeros_like(freq)
kc = np.zeros_like(freq)
gamma = np.zeros_like(freq, dtype = complex)
eps_eff = np.zeros_like(freq)

for idx, el in enumerate(freq):
    k0[idx], kc[idx], gamma[idx], eps_eff[idx] = compute_kc(a, b, zz, zt, m, n, el)
    
fig, ax = plt.subplots()
ax.plot(k0*a, eps_eff, marker = "s", markevery=20, label =f"TE{n}0")
ax.set_ylabel('$\epsilon_{eff}$')
ax.set_xlabel('$k_0 a$')

ax.set_ylim(0, 5)

ax.grid(True, color='gray', linestyle=':')
ax.legend()

plt.show()

#%% Fields computation
# To solve the system of equation in [Byr16, Eq. 2.91] we need to:

# Solve the eq.
def solution(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)]     

def compute_fields(a, b, zz, zt, m, n, freq):
    
    k0, kc, gamma, eps_eff = compute_kc(a, b, zz, zt, m, n, freq)

    ky = m*np.pi/b
    kx = np.sqrt(kc**2-ky**2, dtype = complex)
    
    Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
    X = np.exp(-1j*2*kx*a, dtype = complex)
    
    # 1. Build the matrix
    M = np.array([[gamma*ky, gamma*ky, zt*Z0*kc**2-Z0*k0*kx, zt*Z0*kc**2+Z0*k0*kx],
                  [gamma*ky*X, gamma*ky, -(zt*Z0*kc**2+Z0*k0*kx)*X, -zt*Z0*kc**2+Z0*k0*kx],
                  [-zz*Z0*k0*kx+kc**2*Z0, zz*Z0*k0*kx+kc**2*Z0, zz*Z0*gamma*ky*Z0, zz*Z0*gamma*ky*Z0],
                  [-(zz*Z0*k0*kx+kc**2*Z0)*X, zz*Z0*k0*kx-kc**2*Z0, zz*Z0*gamma*ky*Z0*X, zz*Z0*gamma*ky*Z0]])
    
    # This is a vector [Gamma+ Gamma- Psi+ Psi-]
    [Gammap, Gammam, Psip, Psim] = solution(M)
    
    # Compute all the fields:
    # We still need the dependace from z
    def compute(x, y):
        # Supporting pieces
        Ap = (Gammam*np.exp(1j*kx*x, dtype = complex)+Gammap*np.exp(-1j*kx*x, dtype = complex))
        Bp = (Psim*np.exp(1j*kx*x, dtype = complex)+Psip*np.exp(-1j*kx*x, dtype = complex))
    
        Am = (Gammam*np.exp(1j*kx*x, dtype = complex)-Gammap*np.exp(-1j*kx*x, dtype = complex))
        Bm = (Psim*np.exp(1j*kx*x, dtype = complex)-Psip*np.exp(-1j*kx*x, dtype = complex))
    
    
        Ex = -1j*(gamma*kx*Am-Z0*k0*ky*Bp)*np.sin(ky*y)/(kc**2)
        Ey = -(gamma*ky*Ap-Z0*k0*kx*Bm)*np.cos(ky*y)/(kc**2)
        Ez = Ap*np.sin(ky*y)
    
        Hx = -1j*(k0*ky*Ap-Z0*gamma*kx*Bm)*np.cos(ky*y)/(Z0*kc**2)
        Hy = -(k0*kx*Am+Z0*gamma*ky*Bp)*np.sin(ky*y)/(Z0*kc**2)
        Hz = Bp*np.cos(ky*y)
    
        return [Ex, Ey, Ez, Hx, Hy, Hz]
    
    # Generate a grid of x and y values
    x_values = np.linspace(0, a, 100)  # 100 points between 0 and a
    y_values = np.linspace(0, b, 100)  # 100 points between 0 and b
    
    # Create a meshgrid from x and y values
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    
    # Evaluate the function for each combination of x and y
    fields = compute(x_mesh, y_mesh)
    mesh = [x_values, y_values]

    return [fields, mesh]

# Create a subplot with contourf
fig, ax = plt.subplots(2,3, figsize=(10, 5))

labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

freq = 10e9
A = compute_fields(a, b, zz, zt, m, n, freq)

for idx, el in enumerate(labels):
    row = idx // 3  # Determine the row (0 or 1)
    col = idx % 3   # Determine the column (0, 1, or 2)

    contour = ax[row, col].contourf(A[1][0], A[1][1], np.abs(A[0][idx]), cmap='viridis', levels=100)
    fig.colorbar(contour, ax=ax[row, col])

    # Add labels and title
    ax[row, col].set_xlabel('x [m]')
    ax[row, col].set_ylabel('y [m]')
    ax[row, col].set_title(el)

    ax[row, col].set_aspect('equal')

# Set a title for the whole figure
fig.suptitle(f'TE{n}{m}, WR90', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()