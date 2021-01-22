import numpy

from .bases import spectral_grid, PhysicalArray


class HomogeneousDecay(object):
    def __init__(self, Re, N, padding, **kwargs):
        super().__init__(**kwargs)

        self.Re = Re
        self.nu = 1/Re

        k, x = spectral_grid(N, padding)
        self.u = PhysicalArray([3,], k, x)
        self.k2 = numpy.sum(k*k, axis=0)
        self.P = (numpy.eye(3)[:,:,None,None,None]
                 -k[None,...]*k[:,None,...]/numpy.where(self.k2==0, 1, self.k2))

    def rhs(self):
        u = self.uhat.to_physical()
        vorticity = self.uhat.curl().to_physical()
        nl = numpy.cross(u, vorticity, axis=0)
        nl = PhysicalArray(nl, self.u.k, self.u.x).to_spectral()
        du = numpy.einsum("ij...,j...->i...", self.P, nl)
        du -= self.nu*self.k2*self.uhat
        return du


# class HomogeneousDecaySmagorinsky(HomogeneousDecay):
#     def __init__(self, Cs=0.17, **kwargs):
#         super().__init__(**kwargs)
#         self.Cs = Cs
#         self.dx = (2*numpy.pi/self.N)
#         self.nu_t = 0
        
#     def rhs(self):
#         SijSij = \
#             [ self.spectral_norm(-1j*self.k[i]*self.uhat[i])
#               for i in range(3) ] + \
#             [ self.spectral_norm(-1j*self.k[0]*self.uhat[1]-1j*self.k[1]*self.uhat[0]),
#               self.spectral_norm(-1j*self.k[0]*self.uhat[2]-1j*self.k[2]*self.uhat[0]),
#               self.spectral_norm(-1j*self.k[1]*self.uhat[2]-1j*self.k[2]*self.uhat[1]), ]
#         if self.rank == 0:
#             SijSij = sum(SijSij)
#         SijSij = mpi4py.MPI.COMM_WORLD.bcast(SijSij)
#         self.nu_t = (self.Cs*self.dx)**2*numpy.sqrt(2*SijSij)
#         self.nu = 1/self.Re + self.nu_t

#         rhs = super().rhs()

#         self.nu = 1/self.Re

#         return rhs


class TaylorGreenIC(object):
    def __init__(self, A=1, B=-1, C=0, a=1, b=1, c=1, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(a, int), "Wavenumbers must be integers."
        assert isinstance(b, int), "Wavenumbers must be integers."
        assert isinstance(c, int), "Wavenumbers must be integers."
        assert A*a+B*b+C*c == 0, "Initial condition does not satisfy continuity."

        x = self.u.x
        
        self.u[0] = A*numpy.cos(a*x[0])*numpy.sin(b*x[1])*numpy.sin(c*x[2])
        self.u[1] = B*numpy.sin(a*x[0])*numpy.cos(b*x[1])*numpy.sin(c*x[2])
        self.u[2] = C*numpy.sin(a*x[0])*numpy.sin(b*x[1])*numpy.cos(c*x[2])

        self.uhat = self.u.to_spectral()
