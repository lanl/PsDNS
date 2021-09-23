"""Code verificcation tests

Code verification is the process of showing the correctness of a
solver by demonstrating that it converges to the exact analytic
solution under grid and time step refinement.  For PsDNS, this
formally needs to be done individually for each combination of
equations and integrator.  This module provides sample studies for
some basic equations, with standard schemes.  This partly serves as an
integration test for the psuedo-spectral implementation in the context
of a full solver.  It also can be used as a template from which users
can build tests for other solvers.

A more detailed discussion of how to perform code verification is
beyond the scope of this documentations.  For a good discussion, see
[Oberkampf2010]_.
"""
from mpi4py import MPI

import numpy
import scipy.optimize

from psdns import *
from psdns.equations.basic import Wave, Burgers


class ExactWave(Wave):
    A = 0.9
    
    def exact(self, grid, t):
        r"""An exact solution to be used for testing purposes

        Potentially any function can be used as a solution for the
        wave equation.  For purposes of code verification, we wish to
        use a solution which is periodic, mathematically simple,
        :math:`C^\infty`, and has spectral content at all wave
        numbers.  We chose

        .. math::

            f(x, y, z) = \frac{\sin(x)}{2+\cos(x)}
        """
        eta = grid.x - self.c[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]*t
        #u = 1/(1+self.A*numpy.sin(eta[0])*numpy.sin(eta[1])*numpy.sin(eta[2]))
        u = numpy.cos(6*eta[0])
        return PhysicalArray(grid, u)


class ExactBurgers(Burgers):
    A = 2
    
    def exact(self, grid, t):
        r"""An exact solution for testing purposes

        Utilizing the Cole-Hopf transformation, we pick a solution of the
        Burgers equation such that

        .. math::

            u = - \frac{2 \nu}{\phi} \frac{\partial \phi}{\partial x}

        where :math:`\phi` is a solution to the diffusion equation,

        .. math::

            \frac{\partial \phi}{\partial t}
            = \nu \frac{\partial^2 \phi}{\partial x^2}

        We could pick any solution to this equation, but we want one that
        is simple and periodic, so we choose

        .. math::

            \phi = A + \exp - \nu t \cos x

        or

        .. math::

            u = \frac{2 \nu}{A \exp \nu t + \cos x} \sin x
        """
        return PhysicalArray(
            grid,
            2*self.nu*numpy.sin(grid.x[0])
            / (self.A*numpy.exp(self.nu*t)+numpy.cos(grid.x[0])),
            )
    

class TestConvergence(tests.TestCase):
    """Test convergence for several equation sets

    This class includes convergence tests for several standard PDEs.
    Currently these are

      * Unidirectional wave-eqaution
      * Viscous Burgers equation
    """
    def convergence_test(self, equations, grids, solver_args):
        """Generic convergence test

        An abstract implementation of a convergence test, which runs
        the *equations* using a :class:`~psdns.integrators.RungeKutta`
        integrator on each grid in the list *grids*.  *solver_args* is
        a dictionary containing additional arguments to pass to the
        integrator.

        In addition to the interface described in
        :mod:`~psdns.equations`, the *equations* object must include a
        method :meth:`exact` which takes two arguments, a grid and a
        time, and returns the exact solution of the equations at the
        specified grid locations and time.  This exact solution is
        used both to generate the initial conditions, and to compute
        the error in the simulated solution.
        """
        errs = []
        for grid in grids:
            solver = RungeKutta(
                equations=equations,
                ic=equations.exact(grid, 0).to_spectral(),
                **solver_args,
            )
            solver.run()
            errs.append(
#                (solver.uhat.to_physical()
#                 - equations.exact(solver.uhat.grid, solver.time)).norm(),
                (solver.uhat
                 - equations.exact(solver.uhat.grid, solver.time).to_spectral()).norm(),
                )
        if MPI.COMM_WORLD.rank == 0:
            ns = [grid.pdims[0] for grid in grids]
            fit = numpy.poly1d(
                numpy.polyfit(numpy.log(ns), numpy.log(errs), 1)
                )
            with self.subplots() as (fig, ax):
                ax.loglog(
                    ns,
                    errs,
                    'ko',
                    )
                ax.plot(
                    ns,
                    numpy.exp(fit(numpy.log(ns))),
                    'r-',
                    label=f"p={fit.coeffs[0]:0.2g}",
                    )
                ax.set_xlabel("Number of points")
                ax.set_ylabel("Error")
                ax.legend()
            self.assertLess(fit.coeffs[0], -1)

    def test_wave(self):
        """Grid convergence for the wave equation

        Typical results for the grid convergence of the wave equation
        are shown in :numref:`fig-wave`.  Note that the convergence rate
        is absurdly high compared to a standard finite-difference or
        finite-volume scheme.  The developers are not aware of a
        theoretical estimate for how the convergence of a spectral
        scheme should behave.  The roll-off for very fine grids is
        presumably due to reaching the level of round-off error.

        .. _fig-wave:

        .. figure:: fig/test_convergence.TestConvergence.test_wave.png

            Grid convergence of the wave equation
        """
        self.convergence_test(
            equations=ExactWave(c=[1, 1, 1]),
            grids=[SpectralGrid(2**n) for n in range(2, 9)],
            solver_args={'dt': 0.001, 'tfinal': 0.001},
            )

    def test_burgers(self):
        """Grid convergence for Burgers equation

        Typical results for the grid convergence of Burgers equation
        are shown in :numref:`fig-burgers`.  The results are very
        similar to those seen for :meth:`test_wave`.

        .. _fig-burgers:

        .. figure:: fig/test_convergence.TestConvergence.test_burgers.png

           Grid convergence for Burgers equation
        """
        self.convergence_test(
            equations=ExactBurgers(),
            grids=[SpectralGrid(2**n) for n in range(3, 7)],
            solver_args={'dt': 0.001, 'tfinal': 1.0},
            )
