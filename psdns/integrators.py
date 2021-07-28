"""Time integrators

This module defines the time integrators that can be used for solving
the PDEs.
"""
import sys
from time import time as walltime

import numpy


class Integrator(object):
    """Base class for integrators

    The base class for explicit integrators, which provides
    functionality that is common to all integrators.  Actual integrators
    shoud override the :meth:`step` method to implement the time
    advancement scheme.
    """
    def __init__(self, equations, ic, dt, tfinal, t0=0.0, diagnostics=[]):
        """Initialize an Integrator

        :params equations: the equations
        :params SpectralArray ic: the initial conditions
        :params real dt: The timestep
        :params real tfinal: The simulation stop time
        :params real t0: The simulation start time
        :params list(Diagnostics) diagnostics: diagnostics
        """
        self.equations = equations
        self.uhat = ic
        #: Timestep
        self.dt = dt
        #: Simulation stop time
        self.tfinal = tfinal
        #: Current simulation time
        self.time = t0
        self.diagnostics_list = diagnostics

    def diagnostics(self):
        for diagnostic in self.diagnostics_list:
            diagnostic(self.time, self.equations, self.uhat)
        
    def run(self):
        """Run a simulations to completion
        """
        self.diagnostics()
        time0 = walltime()
        while self.time<self.tfinal-1e-8:
            self.step()
            self.diagnostics()
        #self.runtime = mpi4py.MPI.COMM_WORLD.reduce(walltime()-time0)
        self.runtime = walltime()-time0
        
    def step(self):
        """Advance a single timestep
        """
        raise NotImplementedError

    def print_statistics(self):
        """Print run statistics
        """
        print("Total compute time = ", self.runtime, file=sys.stderr)


class Euler(Integrator):
    """A simple forward Euler integrator
    """
    def step(self):
        self.time += self.dt
        self.uhat += self.dt*self.equations.rhs(self.uhat)


class ImplicitEuler(Integrator):
    """Implicit-Euler with fixed point iteration.

    An implementation of the implicit Euler method, which uses a
    fixed-point iteration with a relaxation factor.
    """
    def __init__(self, alpha=0.5, niter=100, tol=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.niter = niter
        self.tol = tol
        self.uhat0 = self.uhat.copy()
        self.resfile = open("residual.dat", 'w')

    def __del__(self):
        self.resfile.close()

    def step(self):
        self.resfile.write("# Time = {}\n".format(self.time))
        self.uhat0[...] = self.uhat
        self.time += self.dt
        for i in range(self.niter):
            dU = self.uhat0 - self.uhat + self.dt*self.equations.rhs(self.uhat)
            self.uhat += self.alpha*dU
            res = numpy.sqrt(self.uhat.grid.comm.bcast(dU.norm(), root=0))
            if self.uhat.grid.comm.rank == 0:
                self.resfile.write("{} {}\n".format(i, res))
                self.resfile.flush()
            if res<self.tol:
                break
        if self.uhat.grid.comm.rank == 0:
            self.resfile.write("\n\n")


class RungeKutta(Integrator):
    """An Runge-Kutta integrator

    This version of the Runge-Kutta method is that implemented by the
    `spectralDNS <https://github.com/spectralDNS/spectralDNS>`_
    package.  It is neither the full arbitrary Runge-Kutta, nor least
    memory optimized version.  I believe that with the default
    coefficient settings, it is intended to be a reduced memory
    equivalent of the standard Runge-Kutta.

    The method is (in psuedo-code):

    .. code-block:: python

        U[0] := U
        U[1] := U
        for i in [0, ... N]
          U := U[0] + b[i] dt F(U)
          U1[1] := U[1] + a[i] dt F(U)
        U = U[1]
    
    N is the number of sub-steps, and is determined by the length of
    :attr:`a`.

    Note that the solution time also needs to be appropriately advanced
    in the intermediate steps, but as it is not used (absent a
    time-dependent forcing), it is neglected in this implementation.
    """
    #: Coefficients of the integrator
    a = [ 1/6, 1/3, 1/3, 1/6 ]
    #: Coefficients of the accumulator (This must be of the same length
    #: as :attr:`a`.  The last, and only the last, element must be
    #: zero.)
    b = [ 1/2, 1/2, 1, 0 ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: First intermediate storage array
        self.uhat0 = self.uhat.copy()
        #: Second intermediate storage array
        self.uhat1 = self.uhat.copy()
    
    def step(self):
        self.uhat1[...] = self.uhat0[...] = self.uhat
        self.time += self.dt            
        for a, b in zip(self.a, self.b):
            self.dU = self.equations.rhs(self.uhat)
            if b:
                self.uhat[...] = self.uhat0 + b*self.dt*self.dU
            self.uhat1 += a*self.dt*self.dU
        self.uhat[...] = self.uhat1
