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
    '''Defining waveguide dimensions and material properties'''
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
    '''Computing fields in the waveguide with mode theory and analytically. 
    Computation of dissipation factor (alpha)'''

    def __init__(self, waveguide, dispersionCurve, Nx, Ny):
        self.waveguide = waveguide
        self.dispersionCurve = dispersionCurve
        self.Nx = Nx
        self.Ny = Ny

        # Meshgrid
        x_values = np.linspace(0, self.waveguide.a, self.Nx)  
        y_values = np.linspace(0, self.waveguide.b, self.Ny)
        self.x, self.y = np.meshgrid(x_values, y_values)

        # Params
        self.Gammap = None 
        self.Gammam = None 
        self.Pisp = None
        self.Psim = None 

        self.alpha = None
        self.beta = None

        self.compute_params()
        
        # Fields
        self.Ex = None
        self.Ey = None
        self.Ez = None
        self.Hx = None
        self.Hy = None
        self.Hz = None

        # Surface resistance
        self.R_surf = None
        self.alpha_att = None

    def compute_params(self):

        ky = self.dispersionCurve.m*np.pi/self.waveguide.b
        kx = np.sqrt(self.dispersionCurve.kc**2-ky**2, dtype = complex)
        
        Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
        X = np.exp(-1j*2*kx*self.waveguide.a, dtype = complex)
        
        # For better readability of the matrix
        gamma = self.dispersionCurve.gamma
        zt = self.waveguide.zt
        zz = self.waveguide.zz
        kc = self.dispersionCurve.kc
        k0 = self.dispersionCurve.k0

        # 1. Build the matrix
        M = np.array([[gamma*ky, gamma*ky, zt*Z0*kc**2-Z0*k0*kx, zt*Z0*kc**2+Z0*k0*kx],
                    [gamma*ky*X, gamma*ky, -(zt*Z0*kc**2+Z0*k0*kx)*X, -zt*Z0*kc**2+Z0*k0*kx],
                    [-zz*Z0*k0*kx+kc**2*Z0, zz*Z0*k0*kx+kc**2*Z0, zz*Z0*gamma*ky*Z0, zz*Z0*gamma*ky*Z0],
                    [-(zz*Z0*k0*kx+kc**2*Z0)*X, zz*Z0*k0*kx-kc**2*Z0, zz*Z0*gamma*ky*Z0*X, zz*Z0*gamma*ky*Z0]])
        
        # 2. Solve the homogeneous system Ax = 0 numerically
        _, _, V = np.linalg.svd(M)
        [self.Gammap, self.Gammam, self.Psip, self.Psim] = V[-1, :]
    
        # 3. Evaluate coupling of the modes
        try:
            self.alpha = self.Psim/self.Gammam
        except ZeroDivisionError:
            self.alpha = float('inf')
        
        try:
            self.beta = self.Gammam/self.Psim
        except ZeroDivisionError:
            self.beta = float('inf')
    
    # Compute all the fields: we still need the dependace from z
    def compute_fields(self):

        ky = self.dispersionCurve.m*np.pi/self.waveguide.b
        kx = np.sqrt(self.dispersionCurve.kc**2-ky**2, dtype = complex)

        Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
        X = np.exp(-1j*2*kx*self.waveguide.a, dtype = complex)

        # For better readability
        gamma = self.dispersionCurve.gamma
        kc = self.dispersionCurve.kc
        k0 = self.dispersionCurve.k0

        # Supporting pieces
        Ap = (self.Gammam*np.exp(1j*kx*self.x, dtype = complex)+self.Gammap*np.exp(-1j*kx*self.x, dtype = complex))
        Bp = (self.Psim*np.exp(1j*kx*self.x, dtype = complex)+self.Psip*np.exp(-1j*kx*self.x, dtype = complex))
        Am = (self.Gammam*np.exp(1j*kx*self.x, dtype = complex)-self.Gammap*np.exp(-1j*kx*self.x, dtype = complex))
        Bm = (self.Psim*np.exp(1j*kx*self.x, dtype = complex)-self.Psip*np.exp(-1j*kx*self.x, dtype = complex))    
    
        self.Ex = -1j*(gamma*kx*Am-Z0*k0*ky*Bp)*np.sin(ky*self.y)/(kc**2)
        self.Ey = -(gamma*ky*Ap+Z0*k0*kx*Bm)*np.cos(ky*self.y)/(kc**2)
        self.Ez = Ap*np.sin(ky*self.y)
    
        self.Hx = -1j*(-k0*ky*Ap+Z0*gamma*kx*Bm)*np.cos(ky*self.y)/(Z0*kc**2)
        self.Hy = (k0*kx*Am+Z0*gamma*ky*Bp)*np.sin(ky*self.y)/(Z0*kc**2)
        self.Hz = Bp*np.cos(ky*self.y)
    
    # Compute all the fields only for fully metal waveguides
    def compute_fields_analytical(self, mode='TE'):
        
        ky = self.dispersionCurve.m*np.pi/self.waveguide.b
        kx = self.dispersionCurve.n*np.pi/self.waveguide.a
    
        k0 = 2*np.pi*self.dispersionCurve.freq*np.sqrt(sc.epsilon_0*sc.mu_0)
        kc = np.sqrt(kx**2 + ky**2)
        
        gamma_ = np.sqrt(k0**2 - kc**2, dtype=complex)
        Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
        
        if mode=='TE':
            self.Hz = np.cos(kx*self.x)*np.cos(ky*self.y)
            self.Ez = 0*self.x*self.y
            
            self.Hx = 1j*gamma_*kx/kc**2*np.sin(kx*self.x)*np.cos(ky*self.y)
            self.Hy = 1j*gamma_*ky/kc**2*np.cos(kx*self.x)*np.sin(ky*self.y)
            
            self.Ex = k0/gamma_*Z0*self.Hy
            self.Ey = -k0/gamma_*Z0*self.Hx
            
        elif mode=='TM':    
            self.Hz = 0*self.x*self.y
            self.Ez = np.sin(kx*self.x)*np.sin(ky*self.y)
            
            self.Ex = 1j*gamma_*kx/kc**2*np.cos(kx*self.x)*np.sin(ky*self.y)
            self.Ey = 1j*gamma_*ky/kc**2*np.sin(kx*self.x)*np.cos(ky*self.y)
            
            self.Hx = -self.Ey*k0/gamma_/Z0
            self.Hy = self.Ex*k0/gamma_/Z0

    def compute_alpha(self):
        # This is the version that benchmarks the idea of splitting the integrals
        omega = 2*np.pi*self.dispersionCurve.freq
    
        # Power flow integral
        integrand = self.Ex*np.conjugate(self.Hy) - self.Ey*np.conjugate(self.Hx)

        # The integral is fone as a simple sum
        Pnm = 1/2*np.real(np.sum(np.sum(integrand*self.waveguide.a*self.waveguide.b/self.Nx/self.Ny)))
        
        # Computing one current density for every side of the waveguide
        
        # Fields and currents for x = 0, normal is [-1, 0, 0]
        n = np.array([-1, 0, 0])
        H_xe0 = np.array([self.Hx[:,0], self.Hy[:,0], self.Hz[:,0]])

        J_xe0 = []
        for idx, el in enumerate(self.Hx[:,0]):
            J_xe0.append(np.cross(n, H_xe0[:,idx]))
        J_xe0 = np.array(J_xe0)

        # Fields and currents for x = a, normal is [1, 0, 0]
        n = np.array([1, 0, 0])
        H_xea = np.array([self.Hx[:,-1], self.Hy[:,-1], self.Hz[:,-1]])

        J_xea = []

        for idx, el in enumerate(self.Hx[:,0]):
            J_xea.append(np.cross(n, H_xea[:,idx]))
        J_xea = np.array(J_xea)

        # Fields and currents for y = 0, normal is [0, -1, 0]
        n = np.array([0, -1, 0])
        H_ye0 = np.array([self.Hx[0,:], self.Hy[0,:], self.Hz[0,:]])

        J_ye0 = []

        for idx, el in enumerate(self.Hx[:,0]):
            J_ye0.append(np.cross(n, H_ye0[:,idx]))
        J_ye0 = np.array(J_ye0)

        # Fields and currents for y = b, normal is [0, 1, 0]
        n = np.array([0, 1, 0])
        H_yeb = np.array([self.Hx[-1,:], self.Hy[-1,:], self.Hz[-1,:]])

        J_yeb = []

        for idx, el in enumerate(Hx[:,0]):
            J_yeb.append(np.cross(n, H_yeb[:,idx]))
        J_yeb = np.array(J_yeb)
        
        # Now we compute the losses
        # Resistance of the metal walls
        self.R_surf = np.sqrt(omega*sc.mu_0/2/self.waveguide.sigma)
        Rzz = np.real(self.waveguide.zz)
        Rzt = np.real(self.waveguide.zt)

        int1 = []
        for idx, el in enumerate(J_xe0[:,0]):
            int1.append(np.dot(J_xe0[idx], np.conjugate(J_xe0[idx]))*self.waveguide.b/self.Ny)
        Pl1 = self.R_surf/2*np.sum(int1)

        int2 = []
        for idx, el in enumerate(J_xea[:,0]):
            int2.append(np.dot(J_xea[idx], np.conjugate(J_xea[idx]))*self.waveguide.b/self.Ny)
        Pl2 = self.R_surf/2*np.sum(int2)

        int3 = []
        for idx, el in enumerate(J_ye0[:,0]):
            int3.append(np.dot(J_ye0[idx], np.conjugate(J_ye0[idx]))*self.waveguide.a/self.Nx)
        Pl3 = self.R_surf/2*np.sum(int3)

        # Testing on int 4 the separation of the components, this should be done
        # on the vertical walls...
        int4z = []
        int4t = []
        for idx, el in enumerate(J_yeb[:,0]):
            Pl4t = np.dot(J_yeb[idx,0], np.conjugate(J_yeb[idx,0]))*self.waveguide.a/self.Nx
            Pl4z = np.dot(J_yeb[idx,2], np.conjugate(J_yeb[idx,2]))*self.waveguide.a/self.Nx

            int4z.append(Pl4z)
            int4t.append(Pl4t)
        Pl4 = self.R_surf/2*np.sum(int4z) + self.R_surf/2*np.sum(int4t) # Per ora è isotropa

        Pl = Pl1+Pl2+Pl3+Pl4
        
        # This is alpha in dB
        self.alpha_att = Pl/2/Pnm * 8.686
        
        # Just for visualization
        if np.abs(self.alpha_att) > 10:
            self.alpha_att = np.nan
        
    
#%% Testing field maps
# Create a subplot with contourf
fig, ax = plt.subplots(2,3, figsize=(10, 5))

labels = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

freq = 15e9
a=22.86e-3
b=10.16e-3

WR90 = Waveguide(a=a, b=b, zz=0, zt=0, sigma=58e6)
Dispersion_WR90 = DispersionCurve(waveguide=WR90, m=0, n=1, freq=freq)
Fields_WR90 = Fields(waveguide=WR90, dispersionCurve=Dispersion_WR90, Nx=100, Ny=100)
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

#%% Mode checking example
"""
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

#%% Plot
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
