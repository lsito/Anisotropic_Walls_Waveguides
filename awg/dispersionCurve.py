from awg.waveguide import Waveguide 

import numpy as np

from scipy.optimize import fsolve
from scipy.signal import find_peaks
import scipy.constants as sc


class DispersionCurve:
    '''
    Computing dispersion curve
    '''
    
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
            kc = roots[self.n-1] # Because there is no TEM
            
        else:
            x = np.linspace(1, 600, 10000) # Hard coded, not nice!
            roots_idx = find_peaks(-f_of_kc_mne0(x, self.m, self.freq))[0]
            
            roots = x[roots_idx]
            kc = roots[self.n]
        
        return kc
        
    def get_gamma(self):
        gamma = np.sqrt(self.kc**2 - self.k0**2, dtype=complex)
        return gamma
    
    def get_eps_eff(self):
        eps_eff = (np.imag(self.gamma)/self.k0)**2
        return eps_eff