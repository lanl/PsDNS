r"""Test of the RANS solver

This module tests the two-equation :math:`k-\varepsilon` solver,
:class:`~psdns.equations.navier_stokes.KEpsilon`.
"""
import numpy
from numpy import testing as nptest

from mpi4py import MPI

from psdns import *
from psdns.equations.navier_stokes import KEpsilon


class ShearLayer(KEpsilon):
    r"""Exact solution for a turbulent shear layer

    This subclass of :class:`~psdns.equations.navier_stokes.KEpsilon`
    includes an exact solution of the :math:`k-\varepsilon` for a
    turbulent shear layer, which can be used for testing purposes.
    """
    def __init__(self, **kwargs):
        """Initialize a equation object for a turbulent shear layer.

        The constructor for the :class:`ShearLayer` takes exactly the
        same parameters as the
        :class:`~psdns.equations.navier_stokes.KEpsilon`, except that
        the provided values of *Pr_k* and *Pr_e* are ignored and set
        to one, since the exact solution is only valid for those
        values.
        """
        kwargs['Pr_k'] = 1.0
        kwargs['Pr_e'] = 1.0
        super().__init__(**kwargs)

        self.dU = 1
        self.dhdt = self.dU * (self.Ce2-self.Ce1) / 2 \
            * numpy.sqrt(self.Cmu/(self.Ce1*self.Ce2))
        self.kstar = (self.Ce2-self.Ce1)/(8*self.Ce2)
        self.estar = (self.Ce2-self.Ce1) / (16*self.Ce2) \
            * numpy.sqrt(self.Cmu*self.Ce1/self.Ce2)

    def exact(self, grid, t):
        r"""Return the exact solution for the turbulent shear layer.

        Return a :class:`~psdns.bases.PhysicalArray` containing the
        exact solution for the :math:`k-\varepsilon` turbulence model
        of a shear layer on the grid *grid* at time *t*.
        """
        U = PhysicalArray(grid, (5,))
        h = self.dhdt*t
        eta = numpy.arcsin(numpy.sin(grid.x[1]))/h
        U[0] = self.dU/2*eta.clip(min=-1, max=1)
        U[3] = (self.kstar*self.dU**2*(1-eta**2)).clip(min=1e-12)
        U[4] = (self.estar*self.dU**3/h*(1-eta**2)).clip(min=1e-12)
        return U


class TestRANS(tests.TestCase):
    def test_shear_layer(self):
        r"""Finite thickness one-dimensional shear-layer exact solution (Israel, 2019)

        This test runs a finite-thickness one-dimensional shear-layer
        for which the :math:`k-\varepsilon` equations have an exact
        solution [Israel2019]_.  Since the solution is a linear profile
        in velocity and quadratic profiles in the turbulence quantities,
        it is not ideal for testing typical finite-difference codes,
        however, for a psuedo-spectral code, it does activate all the
        modes.  The one potential issue is that, since the solution is
        only :math:`C^\infty` piecewise, with discontinous derivatives
        at the boundary of the layer, it may generate high wavenumber
        ringing.

        .. figure:: fig/test_rans.TestRANS.test_shear_layer.png

            Test results for comparison to an exact solution for a
            finite-thickness shear layer.
        """
        equations = ShearLayer(Re=1e6)
        t0 = 20.0
        solver = ImplicitEuler(
            dt=0.01,
            t0=t0,
            tfinal=21.0,
            equations=equations,
            ic=equations.exact(
                SpectralGrid(sdims=[MPI.COMM_WORLD.size, 2**6, MPI.COMM_WORLD.size]), t0
                ).to_spectral(),
        )
        solver.run()
        exact = equations.exact(solver.uhat.grid, solver.time)
        u = solver.uhat.to_physical()
        if MPI.COMM_WORLD.rank != 0:
            return
        with self.subplots(3, 1) as (fig, axs):
            axs[0].plot(
                u.grid.x[1, 0, :, 0],
                exact[0, 0, :, 0],
                label="Exact solution"
                )
            axs[0].plot(
                u.grid.x[1, 0, :, 0],
                u[0, 0, :, 0],
                '+',
                label="Computed solution"
                )
            axs[0].set_title("Comparision to exact solution")
            axs[0].set_xticks([])
            axs[0].set_ylabel("Velocity")
            axs[1].plot(
                u.grid.x[1, 0, :, 0],
                exact[3, 0, :, 0],
                label="Exact solution"
                )
            axs[1].plot(
                u.grid.x[1, 0, :, 0],
                u[3, 0, :, 0],
                '+',
                label="Computed solution")
            axs[1].set_xticks([])
            axs[1].set_ylabel("TKE")
            axs[2].plot(
                u.grid.x[1, 0, :, 0],
                exact[4, 0, :, 0],
                label="Exact solution")
            axs[2].plot(
                u.grid.x[1, 0, :, 0],
                u[4, 0, :, 0],
                '+',
                label="Computed solution")
            axs[2].set_xlabel("Y")
            axs[2].set_ylabel("Dissipation")
            axs[0].legend()
        nptest.assert_allclose(
            u,
            exact,
            rtol=0.0, atol=1e-2,
            )
