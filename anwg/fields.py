from anwg.waveguide import Waveguide 
from anwg.dispersionCurve import DispersionCurve

import numpy as np

from scipy.optimize import fsolve
from scipy.signal import find_peaks
import scipy.constants as sc


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
        Ap = (self.Gammam*np.exp(-1j*kx*self.x, dtype = complex)+self.Gammap*np.exp(1j*kx*self.x, dtype = complex))
        Bp = (self.Psim*np.exp(-1j*kx*self.x, dtype = complex)+self.Psip*np.exp(1j*kx*self.x, dtype = complex))
        Am = (self.Gammam*np.exp(-1j*kx*self.x, dtype = complex)-self.Gammap*np.exp(1j*kx*self.x, dtype = complex))
        Bm = (self.Psim*np.exp(-1j*kx*self.x, dtype = complex)-self.Psip*np.exp(1j*kx*self.x, dtype = complex))    
    
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

        """
        if self.Ex == None:
            compute_fields()
        """

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

        for idx, el in enumerate(self.Hx[:,0]):
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