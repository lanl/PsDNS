"""Sample implementations of basic partial differential equations

This module provides sample implementations of some classical partial
differential equations.
"""
import numpy

from psdns import *


class Wave(object):
    r"""Wave equation

    The first-order three-dimensional wave equation is

    .. math::

      \frac{\partial u}{\partial t}
      + c_i \frac{\partial u}{\partial x_i}
      = 0

    and has the solution

    .. math::

      u(x_i,t) = f(\boldsymbol{x - c t})

    for any initial condition.

    In spectral space, the equation transforms to

    .. math::

      \frac{\partial \hat{u}}{\partial t}
      - i c_i k_i \hat{u}
      = 0
    """
    def __init__(self, c=[1.0, 0.0, 0.0]):
        """Return a wave equation object.

        Instantiating returns a wave-equation object with a propogation
        speed *c*, which must be of length 3.
        """
        self.c = numpy.asarray(c)

    def rhs(self, uhat):
        return -1j*numpy.tensordot(self.c, uhat.grid.k, 1)*uhat

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
        u = numpy.sin(eta[0])/(2+numpy.cos(eta[0]))
        return PhysicalArray(grid, u)


class Burgers(object):
    r"""One-dimensional viscous Burgers equation

    The viscous Burgers equation in one dimensions is

    .. math::

      \frac{\partial u}{\partial t}
      + u \frac{\partial u}{\partial x}
      = \nu \frac{\partial^2 u}{\partial x^2}

    This is a non-linear, damped wave-equation, propogating along the first
    index.
    """
    def __init__(self, nu=1.0, A=2):
        """Return a Burgers equation object

        Instantiating returns a Burgers equation object with the
        specified viscosity, *nu*.  *A* is a parameter in the exact
        solution function.
        """
        self.nu = nu
        self.A = A

    def rhs(self, uhat):
        u = uhat.to_physical()
        return -1j*uhat.grid.k[0]*(u*u).to_spectral()/2 \
            - uhat.grid.k2*self.nu*uhat

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
