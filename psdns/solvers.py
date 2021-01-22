import numpy

from .bases import spectral_grid, PhysicalArray


class HomogeneousDecay(object):
    def __init__(self, Re, **kwargs):
        super().__init__(**kwargs)

        self.Re = Re
        self.nu = 1/Re

        k = self.uhat.k
        self.k2 = numpy.sum(k*k, axis=0)
        self.P = (numpy.eye(3)[:,:,None,None,None]
                 -k[None,...]*k[:,None,...]/numpy.where(self.k2==0, 1, self.k2))

    def rhs(self):
        u = self.uhat.to_physical()
        vorticity = self.uhat.curl().to_physical()
        nl = numpy.cross(u, vorticity, axis=0)
        nl = PhysicalArray(nl, self.uhat.k, self.uhat.x).to_spectral()
        du = numpy.einsum("ij...,j...->i...", self.P, nl)
        du -= self.nu*self.k2*self.uhat
        return du


class HomogeneousDecaySmagorinsky(HomogeneousDecay):
    def __init__(self, Cs=0.17, **kwargs):
        super().__init__(**kwargs)
        self.Cs = Cs
        self.dx = (2*numpy.pi/self.uhat.x.shape[1])
        self.nu_t = 0
        
    def rhs(self):
        gradu = self.uhat.grad()
        SijSij = (gradu + gradu.transpose(1,0,2,3,4))/2
        SijSij = sum([ SijSij[i,j].norm() for i in range(3) for j in range(3) ])
        self.nu_t = (self.Cs*self.dx)**2*numpy.sqrt(2*SijSij)
        self.nu = 1/self.Re + self.nu_t

        rhs = super().rhs()

        self.nu = 1/self.Re

        return rhs


class TaylorGreenIC(object):
    def __init__(self, N, padding, A=1, B=-1, C=0, a=1, b=1, c=1, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(a, int), "Wavenumbers must be integers."
        assert isinstance(b, int), "Wavenumbers must be integers."
        assert isinstance(c, int), "Wavenumbers must be integers."
        assert A*a+B*b+C*c == 0, "Initial condition does not satisfy continuity."

        k, x = spectral_grid(N, padding)
        u = PhysicalArray([3,], k, x)
        
        u[0] = A*numpy.cos(a*x[0])*numpy.sin(b*x[1])*numpy.sin(c*x[2])
        u[1] = B*numpy.sin(a*x[0])*numpy.cos(b*x[1])*numpy.sin(c*x[2])
        u[2] = C*numpy.sin(a*x[0])*numpy.sin(b*x[1])*numpy.cos(c*x[2])

        self.uhat = u.to_spectral()
