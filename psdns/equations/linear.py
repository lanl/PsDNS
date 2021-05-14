import numpy

from psdns import *


class Wave(object):
    """The 3-d wave equation.

    The three-dimensional wave equation is

    ..math::

      \frac{\partial u}{\partial t}
      + c_i \frac{\partial u}{\partial x_i}
      = 0

    and has the solution

    ..math::

      u(x_i,t) = u(x_i - c_i t, 0)

    for any initial condition.

    In spectral space, the equation transforms to

    ..math::

      \frac{\partial \hat{u}}{\partial t}
      - i c_i k_i \hat{u}
      = 0
    """
    def __init__(self, c=[1.0, 0.0, 0.0]):
        self.c = numpy.asarray(c)

    def rhs(self, uhat):
        return -1j*numpy.tensordot(self.c, uhat.grid.k, 1)*uhat

    def exact(self, grid, t):
        eta = grid.x - self.c[:,numpy.newaxis,numpy.newaxis,numpy.newaxis]*t
        #u = numpy.sign((eta[0] % ( 2 * numpy.pi )) - numpy.pi)
        #u = numpy.sin(eta[0])
        u = numpy.sin(eta[0])/(2+numpy.cos(eta[0]))
        return PhysicalArray(u, grid)


class Burgers(object):
    """
    The viscous Burgers equation is

    ..math::

      \frac{\partial u}{\partial t}
      + u \frac{\partial u}{\partial x}
      = \nu \frac{\partial^2 u}{\partial x^2}

    Utilizing the Cole-Hopf transformation, we pick a solution of the
    Burgers equation such that

    ..math::

      u = - \frac{2 \nu}{\phi} \frac{\partial \phi}{\partial x}

    where :math:`\phi` is a solution to the diffusion equation,

    ..math::

      \frac{\partial \phi}{\partial t}
      = \nu \frac{\partial^2 \phi}{\partial x^2}

    We could pick any solution to this equation, but we want one that
    is simple and periodic, so we choose

    ..math::

      \phi = A + \exp - \nu t \cos x

    or

    ..math::

      u = \frac{2 \nu}{A \exp \nu t + \cos x} \sin x
    """
    def __init__(self, nu=1.0, A=2):
        self.nu = nu
        self.A = A
        
    def rhs(self, uhat):
        u = uhat.to_physical()
        return -1j*uhat.grid.k[0]*(u*u).to_spectral()/2 - uhat.grid.k2*self.nu*uhat

    def exact(self, grid, t):
        return PhysicalArray(
            2*self.nu*numpy.sin(grid.x[0])/(self.A*numpy.exp(self.nu*t)+numpy.cos(grid.x[0])),
            grid
            )
