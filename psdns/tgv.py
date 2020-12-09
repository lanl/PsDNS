import numpy

from .fft import PSFourier
from .diagnostics import Diagnostics
from .solver import Euler

class TGV(Euler, Diagnostics, PSFourier):
    def __init__(self, Re, **kwargs):
        super().__init__(**kwargs)

        self.Re = Re
        self.nu = 1/Re

        self.u = self.physical_array(rank=1)
        self.vorticity = self.physical_array(rank=1)
        self.uhat = self.spectral_array(rank=1)
        self.nl = self.spectral_array(rank=1)
        self.du = self.spectral_array(rank=1)
        self.P = (numpy.eye(3)[:,:,None,None,None]
                 -self.k[None,...]*self.k[:,None,...]/numpy.where(self.k2==0, 1, self.k2))

        self.u[0] = numpy.cos(self.x[0])*numpy.sin(self.x[1])*numpy.sin(self.x[2])
        self.u[1] = -numpy.sin(self.x[0])*numpy.cos(self.x[1])*numpy.sin(self.x[2])
        self.u[2] = 0

        for i in range(3):
            self.to_spectral(self.u[i], self.uhat[i])

    def rhs(self):
        # Compute du/dx in physical space
        self.curl(self.uhat, self.vorticity)
    
        # Compute "non-linear" term in spectral space
        self.cross(self.u, self.vorticity, self.nl)
        
        numpy.einsum("ij...,j...->i...", self.P, self.nl, out=self.du)
        self.du -= self.nu*self.k2*self.uhat
        
        return self.du
