class Waveguide:
    '''
    Defining parameters of waveguide
    
    Class to define the physical parameters of a rectangular waveguide

    Attributes
    ----------
    a : float
        Dimension of the long side of the waveguide in [m]
    b : float
        Dimension of the short side of the waveguide in [m]    
    zz : complex
        Longitudinal component of the anisotropic surface impedance on the
        vertical walls of the waveguide. Normalized by the vacuum impedance Z0.
    zt : complex
        Transverse component of the anisotropic surface impedance on the
        vertical walls of the waveguide. Normalized by the vacuum impedance Z0.
    sigma : float
        Surface conductivity of the horizontal walls in [S/m]
    '''

    def __init__(self, a, b, zz, zt, sigma):
        self.a = a
        self.b = b
        self.zz = zz
        self.zt = zt
        self.sigma = sigma
