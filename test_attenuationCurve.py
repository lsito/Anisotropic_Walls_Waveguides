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

#%% 
fig, ax = plt.subplots()

a=22.86e-3
b=10.16e-3

WR90 = awg.Waveguide(a=a, b=b, zz=0, zt=0, sigma=58e6)

modes = [[1,0], [2,0], [1,1], [0,1]]

for mode in modes:
    m = mode[1]
    n = mode[0]

    alphadB = []
    alphadB_a = []
    Rm = []
    
    freq = np.geomspace(1e6, 3e10, 501)
    
    # For the analytical expression
    ky= m*np.pi/b
    kx= n*np.pi/a
    kc = np.sqrt(kx**2 + ky**2)
    Z0 = np.sqrt(sc.mu_0/sc.epsilon_0)
    
    for el in freq: 
        #alphadB.append(alpha(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=el, sigma=sigma))

        Dispersion_WR90 = awg.DispersionCurve(waveguide=WR90, m=m, n=n, freq=el)
        Fields_WR90 = awg.Fields(waveguide=WR90, dispersionCurve=Dispersion_WR90, Nx=100, Ny=100)
        
        Fields_WR90.compute_fields()
        Fields_WR90.compute_alpha()

        alphadB.append(Fields_WR90.alpha_att)
            
        k0 = 2*np.pi*el*np.sqrt(sc.epsilon_0*sc.mu_0)
        
        Rm.append(Fields_WR90.R_surf)

        if m == 0 or n==0:
            app = 2*Fields_WR90.R_surf/(b*Z0*np.sqrt(1-kc**2/k0**2))*((1+b/a)*kc**2/k0**2+b/a*(1/2-kc**2/k0**2)*(n**2*a*b+m**2*a**2)/(n**2*b**2+m**2*a**2))
            #app = 2*Rm/(b*Z0*np.sqrt(1-kc**2/k0**2))*(1/2+b/a*kc**2/k0**2)
        else:
            app = 2*Fields_WR90.R_surf/(b*Z0*np.sqrt(1-kc**2/k0**2))*((1+b/a)*kc**2/k0**2+b/a*(1-kc**2/k0**2)*(n**2*a*b+m**2*a**2)/(n**2*b**2+m**2*a**2))
            #app = 2*Rm/(b*Z0*np.sqrt(1-kc**2/k0**2))*((1+b/a)*kc**2/k0**2+(1-kc**2/k0**2)*(m**2*b**2+n**2*a*b)/(n**2*a**2+m**2*b**2))
            
        alphadB_a.append(app*8.686)
        # alphadB_a.append(alpha_a(a=a, b=b, zz=zz, zt=zt, m=m, n=n, freq=el, sigma=sigma))
        
    ax.plot(freq, alphadB, label = f"TE{n}{m} numerical")
    ax.plot(freq, alphadB_a, label = f"TE{n}{m}   analytical")
    # ax.plot(frequencies, Rm_, label = f"TE{n}{m} numerical")
    
    plt.legend()
#ax.set_ylim(0, 0.4)   
ax.grid(True)
# %%
