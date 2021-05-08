import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy

from psdns.bases import spectral_grid, PhysicalArray
from psdns.integrators import ImplicitEuler, RungeKutta


class Burgers(object):
    A = 2
    
    def __init__(self, N, padding, nu=1.0, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        u = PhysicalArray((), *spectral_grid(N, padding))
        k = u.k
        self.k2 = numpy.sum(k*k, axis=0)
        u[...] = self.exact(u.x[0], 0)
        self.uhat = u.to_spectral()
        
    def rhs(self):
        u = self.uhat.to_physical()
        return -self.k2*self.nu*self.uhat + self.uhat.k[0]*(u*u).to_spectral()/2

    def exact(self, x, t):
        return 2*self.nu*numpy.sin(x)/(self.A*numpy.exp(self.nu*t)+numpy.cos(x))
    
    
class TestBurgers(unittest.TestCase):
    def test_Burgers(self):
        solver = ImplicitEuler(
            dt=0.02,
            tol=1e-4,
            niter=100,
            tfinal=1.0,
            alpha=0.2,
            equations=Burgers(
                nu=0.1,
                N=[ 2**8, 1, 1 ],
                padding=1,
            ),
        )
        plt.plot(
            solver.equations.uhat.x[0,:,0,0],
            solver.equations.uhat.to_physical()[:,0,0],
            "-o",
            label="Initial Condition",
            )
        solver.run()
        x = solver.equations.uhat.x[0,:,0,0]
        plt.plot(
            x,
            solver.equations.uhat.to_physical()[:,0,0],
            "-+",
            label="Computed Solution",
            )
        plt.plot(
            x,
            solver.equations.exact(x, solver.time),
            "-+",
            label="Exact Solution",
            )
        plt.legend()
        plt.savefig("Burgers.pdf")
        plt.clf()

    def test_convergence(self):
        for n in range(2, 6):
            solver = RungeKutta(
                dt=0.01,
                tfinal=1.0,
                equations=Burgers(
                    nu=1.0,
                    N=[ 2**n, 1, 1 ],
                    padding=1,
                    ),
                )
            solver.run()
            x = solver.equations.uhat.x[0]
            plt.loglog(
                n,
                numpy.linalg.norm(
                    solver.equations.uhat.to_physical()-solver.equations.exact(x, solver.time)
                    )/2**n,
                'ko',
                )
        plt.savefig("convergence.pdf")
        plt.clf()
