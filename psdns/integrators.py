r"""Time integrators

The top level object in PsDNS is an :class:`Integrator`.  Integrators
take an equation class, and some initial conditions, and numerically
integrate the equations in time.  A typical PsDNS script simply
instantiates a conrete integrator subclass, and then invokes the
:meth:`~Integrator.run` method to run the simulation.

For the remainder of this section, in describing time integration
schemes, we adopt the convention that the PDE we wish to solve can be
written as

.. math::

    \frac{\partial}{\partial t} \boldsymbol{U}(t)
    = \boldsymbol{F}[\boldsymbol{U}]

where :math:`\boldsymbol{U}(t)` is the solution vector, which is discretized
as :math:`u^i`.  The index :math:`i` is the discrete time step,
:math:`t=t^i`, and the discrete vector :math:`u^i` will be of of some
dimension higher than three, corresponding to three space dimensions and
then the dimensionality of the solution vector.
"""
import sys
from time import time as walltime

import cupy as  numpy


class Integrator(object):
    """Base class for integrators

    The base class for explicit integrators, which provides
    functionality that is common to all integrators.  Users should not
    instantiate this class directly, as it does not define an acutal
    time advancement scheme.  Instead, use one of the concrete base
    classes.

    Developers should override the :meth:`step` method to implement a
    new time advancement scheme.
    """
    def __init__(self, equations, ic, dt, tfinal, t0=0.0, diagnostics=[]):
        """Initialize an Integrator

        Instantiating an integrator returns an object which will
        integrate the given *equations* from time *t0* to *tfinal* with
        a timestep *dt*.

        The *equations* must be an object that implements a :meth:`rhs`
        method, as described in :mod:`~psdns.equations`.

        The *diagnostics* are a list of objects, typically instances of
        child classes of :class:`~psdns.diagnostics.Diagnostic`, which
        implement a :meth:`~psdns.diagnostics.Diagnostic.diagnostic`
        method, as described in :mod:`~psdns.diagnostics`.

        The *ic* is a :class:`~psdns.bases.SpectralArray` holding the
        initial conditions in spectral space.
        """
        #: The equation object that implements the right-hand side of
        #: the PDE.
        self.equations = equations
        #: The current value of the dependent variable.
        self.uhat = ic
        #: The timestep for the time advancement scheme.
        self.dt = dt
        #: The time at which the simulations should terminate.
        self.tfinal = tfinal
        #: The current value of the simulation time.
        self.time = t0
        #: A list of :class:`~psdns.diagnostics.Diagnostic` objects to
        #: produce diagnostics output for the current simulation.
        self.diagnostics_list = diagnostics

    def diagnostics(self):
        """Run the diagnostics for the current time step.
        """
        for diagnostic in self.diagnostics_list:
            diagnostic(self.time, self.equations, self.uhat)

    def run(self):
        """Run a simulation to completion.

        Advance the timestep repeatedly, performing diagnostic output at
        each step, until the simulation time reaches :attr:`tfinal`.
        """
        self.diagnostics()
        time0 = walltime()
        while self.time < self.tfinal - 1e-8:
            self.step()
            self.diagnostics()
        self.total_walltime = self.uhat.grid.comm.reduce(walltime() - time0)

    def step(self):
        """Advance a single timestep.
        """
        raise NotImplementedError

    def print_statistics(self):
        """Print the run statistics.
        """
        if self.uhat.grid.comm.rank == 0:
            print(
                "Total compute time = ", self.total_walltime,
                file=sys.stderr
                )
            return self.total_walltime


class Euler(Integrator):
    """Forward Euler integrator."""
    def step(self):
        r"""Perform a single forward-Euler step.

        The :class:`Euler` integrator implements the forward Euler time
        advancement scheme,

        .. math::

            u^{n+1} = u^n + \Delta t F[u^n]
        """
        self.time += self.dt
        self.uhat += self.dt*self.equations.rhs(self.uhat)


class ImplicitEuler(Integrator):
    """Implicit-Euler with fixed point iteration.

    The backward-Euler scheme is a first-order in time implicit method
    that is chosen for its stability properties.  The implicit solve is
    performed using an iterative relaxation scheme.
    """

    def __init__(
            self, alpha=0.5, niter=100, tol=1e-6,
            resfile="residual.dat", **kwargs):
        r"""Return an :class:`ImplicitEuler` integrator.

        The integrator returned will use a relaxation factor of
        *alpha*.  Iteration will terminate when the residual is less
        than *tol* or after *niter* iteration loops.

        Residual information will be dumped to *resfile*.

        The remaining arguments are the same as for
        :class:`Integrator`.
        """
        super().__init__(**kwargs)
        #: The relaxation factor.
        self.alpha = alpha
        #: The maximum number of iteration loops.
        self.niter = niter
        #: The tolerance for the iteration residual.
        self.tol = tol
        self.uhat0 = self.uhat.copy()
        self.resfile = open(resfile, 'w')

    def __del__(self):
        self.resfile.close()

    def step(self):
        r"""Advance one backward-Euler time step.

        An implementation of the implicit Euler method, which uses a
        fixed-point iteration with a relaxation factor.  The backward,
        or implicit, Euler scheme is

        .. math::
            u^{n+1} = u^n + \Delta t F[u^{n+1}]

        This is implemented as an iteration loop,

        .. math::

            u_{0} = u^n

            u^* = u^n + \Delta t F[u_i]

            u_{i+1} = \alpha u^* + ( 1 - \alpha ) u_i

        where the subscript index :math:`i` is the iteration step, and the loop
        terminates when the residual
        :math:`\left|| u^* - u^n \right||_2` is smaller than :attr:`niter`.

        This can be re-written in delta form, by defining
        :math:`\Delta u^* = u^* - u^n`, and writing

        .. math::

           \Delta u^* = u^n - u_{i} + \Delta t F[u_i]

           u_{i+1} = u_i + \alpha \Delta u^*
        """
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
            if res < self.tol:
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
    equivalent of the standard fourth-order Runge-Kutta.
    """
    def __init__(
            self,
            a=[1/6, 1/3, 1/3, 1/6],
            b=[1/2, 1/2, 1, 0],
            **kwargs):
        """Return a RungeKutta integrator.

        Returns an ``N`` step integrator, where ``N == len(a)`` and
        *a* and *b* are the weights for the integrator and accumulator,
        respectively.

        The remainging arguments are the same as for :class:`Integrator`.
        """
        super().__init__(**kwargs)
        #: Coefficients of the integrator
        self.a = a
        #: Coefficients of the accumulator (This must be of the same length
        #: as :attr:`a`.  The last, and only the last, element must be
        #: zero.)
        self.b = b
        #: First intermediate storage array
        self.uhat0 = self.uhat.copy()
        #: Second intermediate storage array
        self.uhat1 = self.uhat.copy()

    def step(self):
        r"""Advance one Runge-Kutta time step.

        The method is (in psuedo-code):

        .. code-block:: python

          U[0] := U
          U[1] := U
          for i in [0, ... N]
            U := U[0] + b[i] dt F(U)
            U1[1] := U[1] + a[i] dt F(U)
          U = U[1]

        N is the number of sub-steps, and is determined by the length
        of :attr:`a`.

        Note that the solution time also needs to be appropriately
        advanced in the intermediate steps, but as it is not used
        (absent a time-dependent forcing), it is neglected in this
        implementation.
        """
        self.uhat1[...] = self.uhat0[...] = self.uhat
        self.time += self.dt
        for a, b in zip(self.a, self.b):
            self.dU = self.equations.rhs(self.uhat)
            if b:
                self.uhat[...] = self.uhat0 + b*self.dt*self.dU
            self.uhat1 += a*self.dt*self.dU
        self.uhat[...] = self.uhat1
