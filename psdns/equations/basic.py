"""Sample implementations of basic partial differential equations

This module provides sample implementations of some classical partial
differential equations.
"""
import numpy

from psdns import *
from psdns.equations import Equation


class Wave(Equation):
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

    def rhs(self, time, uhat):
        return -1j*numpy.tensordot(self.c, uhat.grid.k, 1)*uhat


class Burgers(Equation):
    r"""One-dimensional viscous Burgers equation

    The viscous Burgers equation in one dimensions is

    .. math::

      \frac{\partial u}{\partial t}
      + u \frac{\partial u}{\partial x}
      = \nu \frac{\partial^2 u}{\partial x^2}

    This is a non-linear, damped wave-equation, propogating along the first
    index.
    """
    def __init__(self, nu=1.0):
        """Return a Burgers equation object

        Instantiating returns a Burgers equation object with the
        specified viscosity, *nu*.
        """
        self.nu = nu

    def rhs(self, time, uhat):
        u = uhat.to_physical()
        return -1j*uhat.grid.k[0]*(u*u).to_spectral()/2 \
            - uhat.grid.k2*self.nu*uhat

