import unittest


import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy
from numpy import testing as nptest


from psdns.bases import PhysicalArray, SpectralArray, SpectralGrid
from psdns.integrators import ImplicitEuler
from psdns.solvers import KEpsilon


class ShearLayer(KEpsilon):
    def __init__(self, sdims, pdims=None, **kwargs):
        grid = SpectralGrid(sdims, pdims)
        self.uhat = SpectralArray([3,], grid)

        kwargs['Pr_k'] = 1.0
        kwargs['Pr_e'] = 1.0
        super().__init__(**kwargs)

        self.dU = 1
        self.dhdt = self.dU*(self.Ce2-self.Ce1)/2*numpy.sqrt(self.Cmu/(self.Ce1*self.Ce2))
        self.kstar = (self.Ce2-self.Ce1)/(8*self.Ce2)
        self.estar = (self.Ce2-self.Ce1)/(16*self.Ce2)*numpy.sqrt(self.Cmu*self.Ce1/self.Ce2)

        u, K, e = self.exact(20)
        U = PhysicalArray([5,], grid)
        U[0] = u
        U[1] = 0
        U[2] = 0
        U[3] = K
        U[4] = e
        self.uhat = U.to_spectral()

    def exact(self, time):
        h = self.dhdt*(time)
        eta = numpy.arcsin(numpy.sin(self.uhat.grid.x[1]))/h
        u = self.dU/2*eta.clip(min=-1, max=1)
        K = (self.kstar*self.dU**2*(1-eta**2)).clip(min=1e-12)
        e = (self.estar*self.dU**3/h*(1-eta**2)).clip(min=1e-12)
        return u, K, e


class TestRANS(unittest.TestCase):
    def test_shear_layer(self):
        """Finite thickness one-dimensional shear-layer exact solution (Israel, 2018)
        """
        solver = ImplicitEuler(
            dt=0.01,
            tfinal=21.0,
            t0=20.0,
            equations=ShearLayer(
                Re=1e6,
                sdims=[1, 2**6, 1],
            ),
        )
        solver.run()

        exact = solver.equations.exact(solver.time)
        u = solver.equations.uhat.to_physical()
        plt.subplot(311)
        plt.plot(u.grid.x[1,0,:,0], exact[0][0,:,0], label="Exact solution")
        plt.plot(u.grid.x[1,0,:,0], u[0,0,:,0], '+', label="Computed solution")
        plt.title("Comparision to exact solution")
        plt.xticks([])
        plt.ylabel("Velocity")
        plt.legend()
        plt.subplot(312)
        plt.plot(u.grid.x[1,0,:,0], exact[1][0,:,0], label="Exact solution")
        plt.plot(u.grid.x[1,0,:,0], u[3,0,:,0], '+', label="Computed solution")
        plt.xticks([])
        plt.ylabel("TKE")
        plt.subplot(313)
        plt.plot(u.grid.x[1,0,:,0], exact[2][0,:,0], label="Exact solution")
        plt.plot(u.grid.x[1,0,:,0], u[4,0,:,0], '+', label="Computed solution")
        plt.xlabel("Y")
        plt.ylabel("Dissipation")
        plt.savefig("ShearLayer.pdf")
        nptest.assert_allclose(
            u[[0,3,4]],
            numpy.array(exact),
            rtol=0.0, atol=1e-2,
            )
