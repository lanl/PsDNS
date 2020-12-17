import mpi4py
import numpy

from .bases import FFT
from .diagnostics import Diagnostics


class HomogeneousDecay(Diagnostics, FFT):
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

    def rhs(self):
        for i in range(3):
            self.to_physical(self.uhat[i], self.u[i])

        # Compute du/dx in physical space
        self.curl(self.uhat, self.vorticity)
    
        # Compute "non-linear" term in spectral space
        self.cross(self.u, self.vorticity, self.nl)
        
        numpy.einsum("ij...,j...->i...", self.P, self.nl, out=self.du)
        self.du -= self.nu*self.k2*self.uhat
        
        return self.du


class HomogeneousDecaySmagorinsky(HomogeneousDecay):
    def __init__(self, Cs=0.17, **kwargs):
        super().__init__(**kwargs)
        self.Cs = Cs
        self.dx = (2*numpy.pi/self.N)
        self.nu_t = 0
        
    def rhs(self):
        SijSij = \
            [ self.spectral_norm(-1j*self.k[i]*self.uhat[i])
              for i in range(3) ] + \
            [ self.spectral_norm(-1j*self.k[0]*self.uhat[1]-1j*self.k[1]*self.uhat[0]),
              self.spectral_norm(-1j*self.k[0]*self.uhat[2]-1j*self.k[2]*self.uhat[0]),
              self.spectral_norm(-1j*self.k[1]*self.uhat[2]-1j*self.k[2]*self.uhat[1]), ]
        if self.rank == 0:
            SijSij = sum(SijSij)
        SijSij = mpi4py.MPI.COMM_WORLD.bcast(SijSij)
        self.nu_t = (self.Cs*self.dx)**2*numpy.sqrt(2*SijSij)
        self.nu = 1/self.Re + self.nu_t

        rhs = super().rhs()

        self.nu = 1/self.Re

        return rhs


class TaylorGreenIC(object):
    def __init__(self, A=1, B=-1, C=0, a=1, b=1, c=1, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(a, int), "Wavenumbers must be integers."
        assert isinstance(b, int), "Wavenumbers must be integers."
        assert isinstance(c, int), "Wavenumbers must be integers."
        assert A*a+B*b+C*c == 0, "Initial condition does not satisfy continuity."

        self.u[0] = A*numpy.cos(a*self.x[0])*numpy.sin(b*self.x[1])*numpy.sin(c*self.x[2])
        self.u[1] = B*numpy.sin(a*self.x[0])*numpy.cos(b*self.x[1])*numpy.sin(c*self.x[2])
        self.u[2] = C*numpy.sin(a*self.x[0])*numpy.sin(b*self.x[1])*numpy.cos(c*self.x[2])

        for i in range(3):
            self.to_spectral(self.u[i], self.uhat[i])
        
