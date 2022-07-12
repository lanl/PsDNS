"""Diagnostics to use with PsDNS solvers

In order to make it easier to obtain data from simulations, PsDNS
provides a standardized diagnostic interface.
:class:`~psdns.integrators.Intergrator` classes a provided with a user
specified list of diagnostics which are run after each time step.
This module contains a :class:`Diagnostic` base class, from which
users can create their own diagnostics, as well as several
library diagnostics to use.
"""
import csv
import sys

import numpy as np
import cupy as numpy

from psdns import *


class Diagnostic(object):
    """Base class for diagnostics

    This is a base class which does not actually perform any
    diagnostic output, but contains some useful infrastructure.  The
    actual diagnostics are performed by overriding the
    :meth:`diagnostic` method.  The base class takes care of opening
    and closing a file to use for output, and triggering only at the
    correct output interval.

    In principle, any callable which takes the correct arguments can
    be used as a diagnostic, however, it is recommended that
    diagnostics be derived from this base class.
    """
    def __init__(self, tdump, grid, outfile=sys.stderr):
        """Create a diagnostic

        Each time the diagnostic is called, it checks to see if the
        simulation time has increased more than *tdump* since the last
        output, and if so, it triggers another diagnostic output.  The
        :class:`~psdns.bases.SpectralGrid` on which the data to be
        output is stored is given by *grid*.  *outfile* is either a
        file object to use for writing, or a filename which is opened
        for writing.
        """
        #: The dump interval
        self.tdump = tdump
        #: The simulation time when the most recent dump occured
        self.lastdump = -1e9
        self._needs_close = False
        if grid.comm.rank != 0:
            return
        if hasattr(outfile, 'write'):
            #: The file in which to write output
            self.outfile = outfile
        else:
            self.outfile = open(outfile, 'w')
            self._needs_close = True

    def __del__(self):
        if self._needs_close:
            self.outfile.close()

    def __call__(self, time, equations, uhat):
        if time - self.lastdump < self.tdump - 1e-8:
            return
        self.diagnostic(time, equations, uhat)
        self.lastdump = time

    def diagnostic(self, time, equations, uhat):
        """Empty diagnostic

        This is the method that actually computes whatever diagnostic
        quantities are desired and writes them to the output file.  In
        subclasses this should be over-ridden to perform whatever
        analysis is required.  The :meth:`diagnostics` method must
        take exactly three arguments, the current simulation *time*,
        an *equations* object representing the equations being solved,
        and a spectral data array, *uhat*, containing the PDE
        independent variables.
        """
        return NotImplemented


class StandardDiagnostics(Diagnostic):
    """Write statistical averages to a CSV file

    This is a flexible, extensible diagnostic routine for standard
    statistical properties of interesnt in Navier-Stokes simulations.
    It assumes that ``uhat[:3]`` will be the vector velocity field.
    """
    def __init__(self, fields=['tke', 'dissipation'], **kwargs):
        """Create a standard diagnostics object

        The *fields* argument is a list of names of methods, each one
        of which takes *equations* and *uhat* as arguments, and
        returns a scalar which is the value to write to the output.
        Users can sub-class :class:`StandardDiagnostics` to add any
        other scalar methods they need to compute.  The remainder of
        the arguments are the same as for the :class:`Diagnostics`
        class.
        """
        super().__init__(**kwargs)
        #: A list of names of methods to run corresponding to the
        #: columns of the output.
        self.fields = fields
        if kwargs['grid'].comm.rank != 0:
            return
        #: A Python standard library :class:`csv.DictWriter` for
        #: writing CSV output files.
        self.writer = csv.DictWriter(
            self.outfile,
            ['time'] + self.fields
            )
        self.writer.writeheader()

    def tke(self, equations, uhat):
        """Compute the turbulent kinetic energy"""
        u2 = uhat[:3].norm()
        if uhat.grid.comm.rank == 0:
            return 0.5*u2

    def dissipation(self, equations, uhat):
        r"""Compute the dissipation rate of the turbulent kinetic energy

        The dissipation rate is given by

        .. math::
            :label:

            \varepsilon
            = 2 \nu
            \left<
              \frac{\partial u_i}{\partial x_j}
              \frac{\partial u_i}{\partial x_j}
            \right>
        """
         
        enstrophy =  [ (uhat[j]*1j*uhat.grid.k[i]).norm()
                     for i in range(3) for j in range(3)]
        if uhat.grid.comm.rank == 0:
            return 2*equations.nu*sum(enstrophy)

    def urms(self, equations, uhat):
        """Compute <uu> velocity fluctuations"""
        urms = uhat[0].norm()
        if uhat.grid.comm.rank == 0:
            return urms

    def vrms(self, equations, uhat):
        """Compute <vv> velocity fluctuations"""
        vrms = uhat[1].norm()
        if uhat.grid.comm.rank == 0:
            return vrms

    def wrms(self, equations, uhat):
        """Compute <ww> velocity fluctuations"""
        wrms = uhat[2].norm()
        if uhat.grid.comm.rank == 0:
            return wrms

    def S(self, equations, uhat):
        r"""Compute the skewness

        The general tensor form for the skewness is

        .. math::
            :label: def-skewness

            S =
            \left<
                \frac{\partial u_i}{\partial x_k}
                \frac{\partial u_j}{\partial x_k}
                \frac{\partial u_i}{\partial x_j}
            \right>
        """
        gradu = uhat.grad().to_physical()
        S = [
            (gradu[i, k]*gradu[j, k]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(S)

    def S2(self, equations, uhat):
        r"""Compute a generalized skewness

        Typically :math:`S` is generalized to higher-order moments of
        the first-derivative.  However, it can also be generalized in
        terms of higher-order derivatives, as

        .. math::
           :label:

           S_N =
           \left<
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial^N u_j}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial u_i}{\partial x_j}
           \right>

        Note that the regular skewness (eq. :eq:`def-skewness`) is
        given by :math:`S = S_1`.
        """
        gradu = uhat.grad()
        grad2u = gradu.grad().to_physical()
        gradu = gradu.to_physical()
        S = [
            (grad2u[i, k, l]*grad2u[j, k, l]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            for l in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum([s.real for s in S])

    def S3(self, equations, uhat):
        gradu = uhat.grad()
        grad3u = gradu.grad().grad().to_physical()
        gradu = gradu.to_physical()
        S = [
            (grad3u[i, k, l, m]*grad3u[j, k, l, m]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            for l in range(3) for m in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum([s.real for s in S])

    def S4(self, equations, uhat):
        gradu = uhat.grad()
        grad4u = gradu.grad().grad().grad().to_physical()
        gradu = gradu.to_physical()
        S = [
            (grad4u[i, k, l, m, n]*grad4u[j, k, l, m, n]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            for l in range(3) for m in range(3) for n in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum([s.real for s in S])

    def G(self, equations, uhat):
        r"""Compute the palinstrophy

        We can define a generalized enstrophy,

        .. math::
            :label: palinstrophy

            G_N =
            \left<
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
            \right>

        The palinstrophy is :math:`G = G_2`.
        """
        G = [
            (-uhat.grid.k[j]*uhat.grid.k[l]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G3(self, equations, uhat):
        """Compute the :math:`G_3`

        Compute the generalized palinstrophy of the third order,
        :math:`G_3`, (see equation :eq:`palinstrophy`).
        """
        G = [
            (-1j*uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G4(self, equations, uhat):
        """Compute the :math:`G_4`

        Compute the generalized palinstrophy of the third order,
        :math:`G_4`, (see equation :eq:`palinstrophy`).
        """
        G = [
            (uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat.grid.k[n]
             *uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G5(self, equations, uhat):
        """Compute the :math:`G_5`

        Compute the generalized palinstrophy of the third order,
        :math:`G_5`, (see equation :eq:`palinstrophy`).
        """
        G = [
            (1j*uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat.grid.k[n]
             *uhat.grid.k[o]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3) for o in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def H(self, equations, uhat):
        r"""Compute the H

        :math:`H` is a non-isotropic generalization of the :math:`h(r)`
        function in the von Karmen-Howarth equation.  It is defined by

        .. math::
            :label:

            H_n =
            \left<
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial^{N+1} u_k u_i}{\partial x_k \partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
            \right>
        """
        uu = PhysicalArray(
            uhat.grid,
            numpy.einsum(
                "i...,j...->ij...",
                uhat.to_physical(),
                uhat.to_physical()
                )
            ).to_spectral()
        H = [
            ((1j*uhat.grid.k[p]*uhat[j]).to_physical()
             *(-uhat.grid.k[m]*uhat.grid.k[p]*uu[j,m]).to_physical()).average()
            for j in range(3) for m in range(3) for p in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(H)

    def H2(self, equations, uhat):
        uu = PhysicalArray(
            uhat.grid,
            numpy.einsum(
                "i...,j...->ij...",
                uhat.to_physical(),
                uhat.to_physical()
                )
            ).to_spectral()
        H = [
            ((-uhat.grid.k[p]*uhat.grid.k[q]*uhat[j]).to_physical()
             *(-1j*uhat.grid.k[m]*uhat.grid.k[p]*uhat.grid.k[q]*uu[j,m]).to_physical()).average()
            for j in range(3) for m in range(3) for p in range(3)
            for q in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(H)
        
    def H3(self, equations, uhat):
        uu = PhysicalArray(
            uhat.grid,
            numpy.einsum(
                "i...,j...->ij...",
                uhat.to_physical(),
                uhat.to_physical()
                )
            ).to_spectral()
        H = [
            ((-1j*uhat.grid.k[p]*uhat.grid.k[q]*uhat.grid.k[r]*uhat[j]).to_physical()
             *(uhat.grid.k[m]*uhat.grid.k[p]*uhat.grid.k[q]*uhat.grid.k[r]*uu[j,m]).to_physical()).average()
            for j in range(3) for m in range(3) for p in range(3)
            for q in range(3) for r in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(H)
        
    def diagnostic(self, time, equations, uhat):
        """Write the diagnostics specified in :attr:`fields`"""
        row = dict(
            time=time,
            **{field: getattr(self, field)(equations, uhat)
               for field in self.fields}
            )
        if uhat.grid.comm.rank == 0:
            self.writer.writerow(row)
            self.outfile.flush()


class Spectra(Diagnostic):
    r"""A diagnostic class for velocity spectra

    The full velocity spectrum is a tensor function of a
    three-dimensional wave-number,

    .. math::

        \Phi_{ij}(\boldsymbol{k}) 
        = \hat{u}_i(\boldsymbol{k}) \hat{u}_j^*(\boldsymbol{k})

    There are a number of different one-dimensional spectra that can
    be obtained from this quantity, which can all be related to each
    other in the simple case of isotropic turbulence.

    This diagnostic computes the one-dimensional energy spectrum
    function, defined as

    .. math::

        E(k)
        = \iint_{|\boldsymbol{k}|=k} \Phi_{ii}(\boldsymbol{k}) dS
    """
    def integrate_shell(self, u, dk):
        nbins = int(u.grid.kmax/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
       # uu1 = numpy.asnumpy(u.grid.kmag)
       # uu2 = numpy.asnumpy(u._data)
        shape1 = numpy.shape(u.grid.kmag)
       # for k,v in numpy._iter_([u.grid.kmag, u._data]):
        for inx1 in range(shape1[0]):
            for inx2 in range(shape1[1]):
                for inx3 in range(shape1[2]):
                  k = u.grid.kmag[inx1,inx2,inx3]
                  v = u._data[:,inx1,inx2,inx3]
                  spectrum[int(k)/dk] = sum(v)
                  ispectrum[int(k)/dk] += 1
        k = numpy.arange(nbins)*dk
        spectrum = u.grid.comm.reduce(spectrum)
        ispectrum = u.grid.comm.reduce(ispectrum)
        if u.grid.comm.rank == 0:
            spectrum *= 4*numpy.pi*k**2/ispectrum
        return k, spectrum
        
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        k, spectrum = self.integrate_shell(
            (uhat[:3]*uhat[:3].conjugate()).real(),
            1,
            )
        if uhat.grid.comm.rank == 0:
            for i, s in zip(k, spectrum):
                self.outfile.write("{} {}\n".format(i, s))
            self.outfile.write("\n\n")
            self.outfile.flush()


class FieldDump(Diagnostic):
    """Full spectral field file dumps"""
    def diagnostic(self, time, equations, uhat):
        """Write the solution fields in MPI format"""
       # uhat.checkpoint("data{t:04g}".format(t = time))
