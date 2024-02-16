# -*- coding: utf-8 -*-
"""
Module to compute dispersion curve, field distribution and attenuation
coefficients in rectangular waveguides with vertical anisotropic impedance 
boundary conditions.

The code is based on the work from...

@author: Leonardo Sito
"""
#%% Importing Modules

# Plot in separate window
# %matplotlib qt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.signal import find_peaks
import scipy.constants as sc

#%% Waveguide class
class Waveguide:
    def __init__(self, a, b, zz, zt, sigma):
        self.a = a
        self.b = b
        self.zz = zz
        self.zt = zt
        self.sigma = sigma

#%% Dispersion Curve calculation
class DispersionCurve:
    '''Computing dispersion curve'''
    
    def __init__(self, waveguide, n, m, freq):
        
        self.waveguide = waveguide
        self.n = n
        self.m = m
        self.freq = freq

        self.k0 = 2*np.pi*self.freq*np.sqrt(sc.epsilon_0*sc.mu_0)
        self.kc = self.get_kc()
        self.gamma = self.get_gamma()
        self.eps_eff = self.get_eps_eff()

    def get_kc(self):
        # Dispersion relation m = 0, the mode is TE
        def f_of_kc_me0(kc, freq):
        
            omega = 2*np.pi*freq
            X = np.exp(-1j*2*kc*self.waveguide.a, dtype = complex)
            
            f = np.abs(self.waveguide.zt**2*(1-X) 
                       - 2*self.k0/kc*self.waveguide.zt*(1+X) 
                       + (self.k0/kc)**2*(1-X)
                       )
        
            return f
        
        # Dispersion relation m != 0 we do not know if the mode is TE, TM or Hybrid
        def f_of_kc_mne0(kc, m, freq):
        
                omega = 2*np.pi*freq
        
                ky = m*np.pi/self.waveguide.b
                kx = np.sqrt(kc**2-ky**2, dtype = complex)
                gamma = np.sqrt(kc**2-self.k0**2, dtype = complex)
        
                X = np.exp(-1j*2*kx*self.waveguide.a, dtype = complex)
                
                K0xc = self.k0*kx/kc**2
                Kzyc = gamma*ky/kc**2
        
                if np.abs(self.waveguide.zt)+np.abs(self.waveguide.zz)==0: # The metal waveguide condition
                    f = np.abs(1-X)
                else:
                    f = np.abs(
                        2*Kzyc**2*self.waveguide.zz*(-self.waveguide.zt+(X+1)/(1-X)*K0xc*(self.waveguide.zz*self.waveguide.zt+1)-K0xc**2*self.waveguide.zz)
                        + (Kzyc**4+K0xc**4)*self.waveguide.zz**2-2*K0xc**3*(X+1)/(1-X)*self.waveguide.zz*(self.waveguide.zz*self.waveguide.zt+1)
                        + K0xc**2*(1+(self.waveguide.zz*self.waveguide.zt)**2+4*((X+1)/(1-X))**2*self.waveguide.zz*self.waveguide.zt)
                        - 2*K0xc*(X+1)/(1-X)*self.waveguide.zt*(1+self.waveguide.zz*self.waveguide.zt)
                        + self.waveguide.zt**2
                        )
        
                return f
            
        if self.m == 0:
            x = np.linspace(1, 1000, 10000) # Hard coded, not nice!
            roots_idx = find_peaks(-f_of_kc_me0(x, self.freq))[0]
            
            roots = x[roots_idx]
            kc = roots[n-1] # Because there is no TEM
            
        else:
            x = np.linspace(1, 600, 10000) # Hard coded, not nice!
            roots_idx = find_peaks(-f_of_kc_mne0(x, self.m, self.freq))[0]
            
            roots = x[roots_idx]
            kc = roots[n]
        
        return kc
        
    def get_gamma(self):
        gamma = np.sqrt(self.kc**2 - self.k0**2, dtype=complex)
        return gamma
    
    def get_eps_eff(self):
        eps_eff = (np.imag(self.gamma)/self.k0)**2
        return eps_eff

#%% Test Dispersion curves with analytical eq from TE

fig, ax = plt.subplots()

a=22.86e-3
b=10.16e-3

WR90 = Waveguide(a=a, b=b, zz=0, zt=0, sigma=58e6)

modes = [[1,0], [2,0], [0,1], [1,1], [3,0], [2,1]]

for mode in modes:
    m = mode[1]
    n = mode[0]
    
    freq = np.geomspace(1e6, 3e10, 501)
    k0 = np.zeros_like(freq)
    kc = np.zeros_like(freq)
    gamma = np.zeros_like(freq, dtype = complex)
    eps_eff = np.zeros_like(freq)
    
    for idx, el in enumerate(freq):
        params = DispersionCurve(WR90, m=m, n=n, freq=el)
        k0[idx] = params.k0
        kc[idx] = params.kc
        gamma[idx] = params.gamma
        eps_eff[idx] = params.eps_eff
    
    # The analytical dispersion equation for a rectangular waveguide
    ky_a= m*np.pi/b
    kx_a= n*np.pi/a
    
    omega = 2*np.pi*freq
    k0_a = omega*np.sqrt(sc.epsilon_0*sc.mu_0)
    kc_a = np.ones_like(freq)*np.sqrt(kx_a**2 + ky_a**2)
    
    gamma_a = np.sqrt(kc_a**2 - k0_a**2, dtype=complex)
    eps_eff_a = (np.imag(gamma_a)/k0_a)**2
    
    ax.plot(k0*a, eps_eff, marker = "s", markevery=10, label =f"TE{n}{m} Mode Theory")
    ax.plot(k0_a*a, eps_eff_a, marker = "s", markevery=20, label =f"TE{n}{m} Analytical")
    ax.set_ylabel('$\epsilon_{eff}$')
    ax.set_xlabel('$k_0 a$')
    
    ax.set_ylim(0, 5)
    
    ax.grid(True, color='gray', linestyle=':')
    ax.legend()
    
plt.show()

#%% Fields computation
# To solve the system of equation in [Byr16, Eq. 2.91] we need to:
class Fields:
    def __init__(self, waveguide, dispersionCurve, Nx, Ny):
        self.waveguide = waveguide
        self.dispersionCurve = dispersionCurve
        self.Nx = Nx
        self.Ny = Ny

        def compute_params(self):

            ky = self.dispersionCurve.m*np.pi/self.waveguide.b
            kx = np.sqrt(self.dispersionCurve.kc**2-ky**2, dtype = complex)
            
            Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
            X = np.exp(-1j*2*kx*self.waveguide.a, dtype = complex)
            
            # 1. Build the matrix
            M = np.array([[gamma*ky, gamma*ky, zt*Z0*kc**2-Z0*k0*kx, zt*Z0*kc**2+Z0*k0*kx],
                        [gamma*ky*X, gamma*ky, -(zt*Z0*kc**2+Z0*k0*kx)*X, -zt*Z0*kc**2+Z0*k0*kx],
                        [-zz*Z0*k0*kx+kc**2*Z0, zz*Z0*k0*kx+kc**2*Z0, zz*Z0*gamma*ky*Z0, zz*Z0*gamma*ky*Z0],
                        [-(zz*Z0*k0*kx+kc**2*Z0)*X, zz*Z0*k0*kx-kc**2*Z0, zz*Z0*gamma*ky*Z0*X, zz*Z0*gamma*ky*Z0]])
            
            # This is a vector [Gamma+ Gamma- Psi+ Psi-]
            # [Gammap, Gammam, Psip, Psim] = solution(M)
            # print("Nontrivial solution 0:")
            # print([Gammap, Gammam, Psip, Psim])
        
            # Solve the homogeneous system Ax = 0 numerically
            _, _, V = np.linalg.svd(M)
            [Gammap, Gammam, Psip, Psim] = V[-1, :]
            # Print the nontrivial solution
            # print("Nontrivial solution 1:")
            # print([Gammap, Gammam, Psip, Psim])
        
            try:
                alpha = Psim/Gammam
            except ZeroDivisionError:
                alpha = float('inf')
            
            try:
                beta = Gammam/Psim
            except ZeroDivisionError:
                beta = float('inf')

# Solve the eq.

def solution(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)]     

def compute_fields(a, b, zz, zt, m, n, freq, flag=True):
    
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
    # [Gammap, Gammam, Psip, Psim] = solution(M)
    # print("Nontrivial solution 0:")
    # print([Gammap, Gammam, Psip, Psim])
 
    # Solve the homogeneous system Ax = 0 numerically
    _, _, V = np.linalg.svd(M)
    [Gammap, Gammam, Psip, Psim] = V[-1, :]
    # Print the nontrivial solution
    # print("Nontrivial solution 1:")
    # print([Gammap, Gammam, Psip, Psim])
 
    try:
        alpha = Psim/Gammam
    except ZeroDivisionError:
        alpha = float('inf')
    
    try:
        beta = Gammam/Psim
    except ZeroDivisionError:
        beta = float('inf')
    
    # Compute all the fields:
    # We still need the dependace from z
    def compute(x, y):
        # Supporting pieces
        Ap = (Gammam*np.exp(1j*kx*x, dtype = complex)+Gammap*np.exp(-1j*kx*x, dtype = complex))
        Bp = (Psim*np.exp(1j*kx*x, dtype = complex)+Psip*np.exp(-1j*kx*x, dtype = complex))
        Am = (Gammam*np.exp(1j*kx*x, dtype = complex)-Gammap*np.exp(-1j*kx*x, dtype = complex))
        Bm = (Psim*np.exp(1j*kx*x, dtype = complex)-Psip*np.exp(-1j*kx*x, dtype = complex))    
    
        Ex = -1j*(gamma*kx*Am-Z0*k0*ky*Bp)*np.sin(ky*y)/(kc**2)
        Ey = -(gamma*ky*Ap+Z0*k0*kx*Bm)*np.cos(ky*y)/(kc**2)
        Ez = Ap*np.sin(ky*y)
    
        Hx = -1j*(-k0*ky*Ap+Z0*gamma*kx*Bm)*np.cos(ky*y)/(Z0*kc**2)
        Hy = (k0*kx*Am+Z0*gamma*ky*Bp)*np.sin(ky*y)/(Z0*kc**2)
        Hz = Bp*np.cos(ky*y)
    
        return [Ex, Ey, Ez, Hx, Hy, Hz]
    
    if flag==True:
        # Generate a grid of x and y values
        x_values = np.linspace(0, a, 100)  # 100 points between 0 and a
        y_values = np.linspace(0, b, 100)  # 100 points between 0 and b
        
        # Create a meshgrid from x and y values
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        
        # Evaluate the function for each combination of x and y
        fields = compute(x_mesh, y_mesh)
        mesh = [x_values, y_values]
    
        return [fields, mesh, alpha, beta]
    else:
        return [alpha, beta]
    
#%% Testing field maps
# Create a subplot with contourf
fig, ax = plt.subplots(2,3, figsize=(10, 5))

labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

freq = 10e9
A = compute_fields(a=a, b=b, zz=0, zt=0, m=1, n=0, freq=15e9)

for idx, el in enumerate(labels):
    row = idx // 3  # Determine the row (0 or 1)
    col = idx % 3   # Determine the column (0, 1, or 2)

    contour = ax[row, col].contourf(A[1][0], A[1][1], np.imag(A[0][idx])/np.real(A[0][idx]), cmap='viridis', levels=100)
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

"""
#%% Mode checking example

m = 0
n = 1 # nth zero of f(kc)

freq = np.geomspace(1e6, 2e10, 1001)
k0 = np.zeros_like(freq)
kc = np.zeros_like(freq)
gamma = np.zeros_like(freq, dtype = complex)
eps_eff = np.zeros_like(freq)
alpha = np.zeros_like(freq, dtype = complex)
beta = np.zeros_like(freq, dtype = complex)

for idx, el in enumerate(freq):
    k0[idx], kc[idx], gamma[idx], eps_eff[idx] = compute_kc(a, b, zz, zt, m, n, el)
    alpha[idx], beta[idx] = compute_fields(a, b, zz, zt, m, n, el, flag=False)


fig, ax = plt.subplots()
ax.plot(k0*a, eps_eff, marker = "s", markevery=20, label =f"TE{n}0")
ax.set_ylabel('$\epsilon_{eff}$')
ax.set_xlabel('$k_0 a$')

ax.set_ylim(0, 5)

ax.grid(True, color='gray', linestyle=':')
ax.legend()

plt.show()

# Modify the eq. multipy by (1-X)**2
# I have to actually understand first if the mode is a TE or TM or HE or EH to 
# then name them in terms of n
# Check all modes now (all, go up to some orders)
"""

#%% The analytical dispersion equation for a rectangular waveguide

def compute_fields_a(a, b, m, n, freq, flag):

    ky= m*np.pi/b
    kx= n*np.pi/a
    
    k0 = 2*np.pi*freq*np.sqrt(sc.epsilon_0*sc.mu_0)
    kc = np.sqrt(kx**2 + ky**2)
    
    gamma_ = np.sqrt(k0**2 - kc**2, dtype=complex)
    Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
    
    def compute_fieldsTE(x,y):
        
        HzTE = np.cos(kx*x)*np.cos(ky*y)
        EzTE = 0*x*y
        
        HxTE = 1j*gamma_*kx/kc**2*np.sin(kx*x)*np.cos(ky*y)
        HyTE = 1j*gamma_*ky/kc**2*np.cos(kx*x)*np.sin(ky*y)
        
        ExTE = k0/gamma_*Z0*HyTE
        EyTE = -k0/gamma_*Z0*HxTE
        
        return [ExTE, EyTE, EzTE, HxTE, HyTE, HzTE]
        
    def compute_fieldsTM(x,y):
        
        HzTM = 0*x*y
        EzTM = np.sin(kx*x)*np.sin(ky*y)
        
        ExTM = 1j*gamma_*kx/kc**2*np.cos(kx*x)*np.sin(ky*y)
        EyTM = 1j*gamma_*ky/kc**2*np.sin(kx*x)*np.cos(ky*y)
        
        HxTM = -EyTM*k0/gamma_/Z0
        HyTM = ExTM*k0/gamma_/Z0
        
        return [ExTM, EyTM, EzTM, HxTM, HyTM, HzTM]
    
    # Generate a grid of x and y values
    x_values = np.linspace(0, a, 100)  # 100 points between 0 and a
    y_values = np.linspace(0, b, 100)  # 100 points between 0 and b
    
    # Create a meshgrid from x and y values
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    
    # Evaluate the function for each combination of x and y
    if flag=="TE":
        fields = compute_fieldsTE(x_mesh, y_mesh)
    else:
        fields = compute_fieldsTM(x_mesh, y_mesh)
    
    mesh = [x_values, y_values]
    return [fields, mesh]

# Create a subplot with contourf
fig, ax = plt.subplots(2,3, figsize=(10, 5))

labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

m = 1
n = 0
A = compute_fields_a(a=a, b=b, m=m, n=n, freq=20e9, flag="TE")

for idx, el in enumerate(labels):
    row = idx // 3  # Determine the row (0 or 1)
    col = idx % 3   # Determine the column (0, 1, or 2)

    contour = ax[row, col].contourf(A[1][0], A[1][1], np.real(A[0][idx]), cmap='viridis', levels=100)
    fig.colorbar(contour, ax=ax[row, col])

    # Add labels and title
    ax[row, col].set_xlabel('x [m]')
    ax[row, col].set_ylabel('y [m]')
    ax[row, col].set_title(el)

    ax[row, col].set_aspect('equal')

# Set a title for the whole figure
fig.suptitle(f'TE{n}{m} analytical, WR90', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# Benchmark
fig, ax = plt.subplots(2,3, figsize=(10, 5))

labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

a = 22.86e-3 # Long side of waveguide
b = 10.16e-3 # Short side of waveguide
m = 1
n = 1
TEanalytical = compute_fields_a(a=a, b=b, m=m, n=n, freq=20e9, flag="TE")
TManalytical = compute_fields_a(a=a, b=b, m=m, n=n, freq=20e9, flag="TM")
A = compute_fields(a=a, b=b, zz=0, zt=0, m=m, n=n, freq=20e9)

for idx, el in enumerate(labels):
    row = idx // 3  # Determine the row (0 or 1)
    col = idx % 3   # Determine the column (0, 1, or 2)

    contour = ax[row, col].contourf(A[1][0], A[1][1], np.abs(np.real(A[0][idx])/np.amax(np.real(A[0][idx]))-np.real(TEanalytical[0][idx])/np.amax(np.real(TEanalytical[0][idx]))), cmap='viridis', levels=100)
    fig.colorbar(contour, ax=ax[row, col])

    # Add labels and title
    ax[row, col].set_xlabel('x [m]')
    ax[row, col].set_ylabel('y [m]')
    ax[row, col].set_title(el)

    ax[row, col].set_aspect('equal')

# Set a title for the whole figure
fig.suptitle(f'TE{n}{m} diff, WR90', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

#%% Computation of alpha

# This is the version that benchmarks the idea of splitting the integrals
def alpha(a, b, zz, zt, m, n, freq, sigma):
    # To solve the system of equation in [Byr16, Eq. 2.91] we need to:
    # Here we follow [Benedikt Byrne. “Etude et conception de guides d’onde et 
# d’antennes cornets `a m ́etamat ́eriaux”. PhD thesis. Nov. 2016. url: https://oatao.univ-toulouse.fr/172 [Byr16]

    omega = 2*np.pi*freq
    
    [Ex, Ey, Ez, Hx, Hy, Hz] = compute_fields(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=freq)[0]
    #[Ex, Ey, Ez, Hx, Hy, Hz] = compute_fields_a(a=a, b=b, m=m, n=n, freq=freq, flag="TE")[0]
    # Power flow integral
    integrand = Ex*np.conjugate(Hy) - Ey*np.conjugate(Hx)
    # The integral you can do it as a simple sum
    Pnm = 1/2*np.real(np.sum(np.sum(integrand*a*b/100/100)))
    # I now need four currents, one for every side of the waveguide
    
    # Fields and currents for x = 0, normal is [-1, 0, 0]
    n = np.array([-1, 0, 0])
    H_xe0 = np.array([Hx[:,0], Hy[:,0], Hz[:,0]])

    J_xe0 = []

    for idx, el in enumerate(Hx[:,0]):
        J_xe0.append(np.cross(n, H_xe0[:,idx]))

    J_xe0 = np.array(J_xe0)

    # Fields and currents for x = a, normal is [1, 0, 0]
    n = np.array([1, 0, 0])
    H_xea = np.array([Hx[:,-1], Hy[:,-1], Hz[:,-1]])

    J_xea = []

    for idx, el in enumerate(Hx[:,0]):
        J_xea.append(np.cross(n, H_xea[:,idx]))

    J_xea = np.array(J_xea)

    # Fields and currents for y = 0, normal is [0, -1, 0]
    n = np.array([0, -1, 0])
    H_ye0 = np.array([Hx[0,:], Hy[0,:], Hz[0,:]])

    J_ye0 = []

    for idx, el in enumerate(Hx[:,0]):
        J_ye0.append(np.cross(n, H_ye0[:,idx]))

    J_ye0 = np.array(J_ye0)

    # Fields and currents for y = b, normal is [0, 1, 0]
    n = np.array([0, 1, 0])
    H_yeb = np.array([Hx[-1,:], Hy[-1,:], Hz[-1,:]])

    J_yeb = []

    for idx, el in enumerate(Hx[:,0]):
        J_yeb.append(np.cross(n, H_yeb[:,idx]))

    J_yeb = np.array(J_yeb)
    
    # Now we compute the losses
    # Resistance of the metal walls
    Rm = np.sqrt(omega*sc.mu_0/2/sigma)
    Rzz = np.real(zz)
    Rzt = np.real(zt)

    int1 = []
    for idx, el in enumerate(J_xe0[:,0]):
        int1.append(np.dot(J_xe0[idx], np.conjugate(J_xe0[idx]))*b/100)
    Pl1 = Rm/2*np.sum(int1)

    int2 = []
    for idx, el in enumerate(J_xea[:,0]):
        int2.append(np.dot(J_xea[idx], np.conjugate(J_xea[idx]))*b/100)
    Pl2 = Rm/2*np.sum(int2)

    int3 = []
    for idx, el in enumerate(J_ye0[:,0]):
        int3.append(np.dot(J_ye0[idx], np.conjugate(J_ye0[idx]))*a/100)
    Pl3 = Rm/2*np.sum(int3)

    int4z = []
    int4t = []
    for idx, el in enumerate(J_yeb[:,0]):
        Pl4t = np.dot(J_yeb[idx,0], np.conjugate(J_yeb[idx,0]))*a/100
        Pl4z = np.dot(J_yeb[idx,2], np.conjugate(J_yeb[idx,2]))*a/100
        int4z.append(Pl4z)
        int4t.append(Pl4t)
    Pl4 = Rm/2*np.sum(int4z) + Rm/2*np.sum(int4t) # Per ora è isotropa

    Pl = Pl1+Pl2+Pl3+Pl4
    
    # This is alpha in dB
    alpha = Pl/2/Pnm * 8.686
    if np.abs(alpha) > 10:
        alpha = np.nan
    
    return alpha

#%%
def alpha_a(a, b, zz, zt, m, n, freq, sigma):
    # To solve the system of equation in [Byr16, Eq. 2.91] we need to:
    # Here we follow [Benedikt Byrne. “Etude et conception de guides d’onde et 
# d’antennes cornets `a m ́etamat ́eriaux”. PhD thesis. Nov. 2016. url: https://oatao.univ-toulouse.fr/172 [Byr16]

    omega = 2*np.pi*freq
    
    #[Ex, Ey, Ez, Hx, Hy, Hz] = compute_fields(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=freq)[0]
    [Ex, Ey, Ez, Hx, Hy, Hz] = compute_fields_a(a=a, b=b, m=m, n=n, freq=freq, flag="TE")[0]
    # Power flow integral
    integrand = Ex*np.conjugate(Hy) - Ey*np.conjugate(Hx)
    # The integral you can do it as a simple sum
    Pnm = 1/2*np.real(np.sum(np.sum(integrand*a*b/100/100)))
    # I now need four currents, one for every side of the waveguide
    
    # Fields and currents for x = 0, normal is [-1, 0, 0]
    n = np.array([-1, 0, 0])
    H_xe0 = np.array([Hx[:,0], Hy[:,0], Hz[:,0]])

    J_xe0 = []

    for idx, el in enumerate(Hx[:,0]):
        J_xe0.append(np.cross(n, H_xe0[:,idx]))

    J_xe0 = np.array(J_xe0)

    # Fields and currents for x = a, normal is [1, 0, 0]
    n = np.array([1, 0, 0])
    H_xea = np.array([Hx[:,-1], Hy[:,-1], Hz[:,-1]])

    J_xea = []

    for idx, el in enumerate(Hx[:,0]):
        J_xea.append(np.cross(n, H_xea[:,idx]))

    J_xea = np.array(J_xea)

    # Fields and currents for y = 0, normal is [0, -1, 0]
    n = np.array([0, -1, 0])
    H_ye0 = np.array([Hx[0,:], Hy[0,:], Hz[0,:]])

    J_ye0 = []

    for idx, el in enumerate(Hx[:,0]):
        J_ye0.append(np.cross(n, H_ye0[:,idx]))

    J_ye0 = np.array(J_ye0)

    # Fields and currents for y = b, normal is [0, 1, 0]
    n = np.array([0, 1, 0])
    H_yeb = np.array([Hx[-1,:], Hy[-1,:], Hz[-1,:]])

    J_yeb = []

    for idx, el in enumerate(Hx[:,0]):
        J_yeb.append(np.cross(n, H_yeb[:,idx]))

    J_yeb = np.array(J_yeb)
    
    # Now we compute the losses
    # Resistance of the metal walls
    Rm = np.sqrt(omega*sc.mu_0/2/sigma)
    Rzz = np.real(zz)
    Rzt = np.real(zt)

    int1 = []
    for idx, el in enumerate(J_xe0[:,0]):
        int1.append(np.dot(J_xe0[idx], np.conjugate(J_xe0[idx]))*b/100)
    Pl1 = Rm/2*np.sum(int1)

    int2 = []
    for idx, el in enumerate(J_xea[:,0]):
        int2.append(np.dot(J_xea[idx], np.conjugate(J_xea[idx]))*b/100)
    Pl2 = Rm/2*np.sum(int2)

    int3 = []
    for idx, el in enumerate(J_ye0[:,0]):
        int3.append(np.dot(J_ye0[idx], np.conjugate(J_ye0[idx]))*a/100)
    Pl3 = Rm/2*np.sum(int3)

    int4z = []
    int4t = []
    for idx, el in enumerate(J_yeb[:,0]):
        Pl4t = np.dot(J_yeb[idx,0], np.conjugate(J_yeb[idx,0]))*a/100
        Pl4z = np.dot(J_yeb[idx,2], np.conjugate(J_yeb[idx,2]))*a/100
        int4z.append(Pl4z)
        int4t.append(Pl4t)
    Pl4 = Rm/2*np.sum(int4z) + Rm/2*np.sum(int4t) # Per ora è isotropa

    Pl = Pl1+Pl2+Pl3+Pl4
    
    # This is alpha in dB
    alpha = Pl/2/Pnm * 8.686

    return alpha


#%% Benchmark with multiple modes

fig, ax = plt.subplots(figsize=(4, 2))

modes = [[1,0], [2,0], [1,1], [0,1]]
#modes = [[1,0]]

for mode in modes:
    m = mode[1]
    n = mode[0]

    alphadB = []
    alphadB_a = []
    
    frequencies = np.linspace(7e9, 30e9, 100)
    
    # The waveguide (WR90)
    a = 22.86e-3 # Long side of waveguide
    b = 10.16e-3 # Short side of waveguide
    sigma = 5.8e7 
    Rm_ = []
    zt = 0
    zz = 0
    
    # For the analytical expression
    ky= m*np.pi/b
    kx= n*np.pi/a
    kc = np.sqrt(kx**2 + ky**2)
    Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
    
    for el in frequencies:
        #alphadB.append(alpha(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=el, sigma=sigma))
        alphadB.append(alpha(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=el, sigma=sigma))
            
        k0 = 2*np.pi*el*np.sqrt(sc.epsilon_0*sc.mu_0)
        
        Rm = np.sqrt(2*np.pi*el*sc.mu_0/2/sigma)
        Rm_.append(Rm)
        if m == 0 or n==0:
            app = 2*Rm/(b*Z0*np.sqrt(1-kc**2/k0**2))*((1+b/a)*kc**2/k0**2+b/a*(1/2-kc**2/k0**2)*(n**2*a*b+m**2*a**2)/(n**2*b**2+m**2*a**2))
            #app = 2*Rm/(b*Z0*np.sqrt(1-kc**2/k0**2))*(1/2+b/a*kc**2/k0**2)
        else:
            app = 2*Rm/(b*Z0*np.sqrt(1-kc**2/k0**2))*((1+b/a)*kc**2/k0**2+b/a*(1-kc**2/k0**2)*(n**2*a*b+m**2*a**2)/(n**2*b**2+m**2*a**2))
            #app = 2*Rm/(b*Z0*np.sqrt(1-kc**2/k0**2))*((1+b/a)*kc**2/k0**2+(1-kc**2/k0**2)*(m**2*b**2+n**2*a*b)/(n**2*a**2+m**2*b**2))
            
        alphadB_a.append(app*8.686)
        # alphadB_a.append(alpha_a(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=el, sigma=sigma))
        
    ax.plot(frequencies, alphadB, label = f"TE{n}{m} numerical")
    ax.plot(frequencies, alphadB_a, label = f"TE{n}{m}   analytical")
    # ax.plot(frequencies, Rm_, label = f"TE{n}{m} numerical")
    
    plt.legend()
#ax.set_ylim(0, 0.4)   
ax.grid(True)
