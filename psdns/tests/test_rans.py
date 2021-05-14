import unittest


import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy
from numpy import testing as nptest


from psdns import *
from psdns.equations.navier_stokes import KEpsilon


class ShearLayer(KEpsilon):
    def __init__(self, **kwargs):
        kwargs['Pr_k'] = 1.0
        kwargs['Pr_e'] = 1.0
        super().__init__(**kwargs)

        self.dU = 1
        self.dhdt = self.dU*(self.Ce2-self.Ce1)/2*numpy.sqrt(self.Cmu/(self.Ce1*self.Ce2))
        self.kstar = (self.Ce2-self.Ce1)/(8*self.Ce2)
        self.estar = (self.Ce2-self.Ce1)/(16*self.Ce2)*numpy.sqrt(self.Cmu*self.Ce1/self.Ce2)

    def exact(self, grid, t):
        U = PhysicalArray([5,], grid)
        h = self.dhdt*t
        eta = numpy.arcsin(numpy.sin(grid.x[1]))/h
        U[0] = self.dU/2*eta.clip(min=-1, max=1)
        U[3] = (self.kstar*self.dU**2*(1-eta**2)).clip(min=1e-12)
        U[4] = (self.estar*self.dU**3/h*(1-eta**2)).clip(min=1e-12)
        return U


class TestRANS(unittest.TestCase):
    def test_shear_layer(self):
        """Finite thickness one-dimensional shear-layer exact solution (Israel, 2018)
        """
        equations = ShearLayer(Re=1e6)
        t0 = 20.0
        solver = ImplicitEuler(
            dt=0.01,
            t0=t0,
            tfinal=21.0,
            equations=equations,
            ic=equations.exact(
                SpectralGrid(sdims=[1, 2**6, 1]), t0
                ).to_spectral(),
        )
        solver.run()

        exact = equations.exact(solver.uhat.grid, solver.time)
        u = solver.uhat.to_physical()
        plt.subplot(311)
        plt.plot(u.grid.x[1,0,:,0], exact[0,0,:,0], label="Exact solution")
        plt.plot(u.grid.x[1,0,:,0], u[0,0,:,0], '+', label="Computed solution")
        plt.title("Comparision to exact solution")
        plt.xticks([])
        plt.ylabel("Velocity")
        plt.legend()
        plt.subplot(312)
        plt.plot(u.grid.x[1,0,:,0], exact[3,0,:,0], label="Exact solution")
        plt.plot(u.grid.x[1,0,:,0], u[3,0,:,0], '+', label="Computed solution")
        plt.xticks([])
        plt.ylabel("TKE")
        plt.subplot(313)
        plt.plot(u.grid.x[1,0,:,0], exact[4,0,:,0], label="Exact solution")
        plt.plot(u.grid.x[1,0,:,0], u[4,0,:,0], '+', label="Computed solution")
        plt.xlabel("Y")
        plt.ylabel("Dissipation")
        plt.savefig("ShearLayer.pdf")
        nptest.assert_allclose(
            u,
            exact,
            rtol=0.0, atol=1e-2,
            )
