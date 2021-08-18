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

import numpy


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
        enstrophy = [(1j*uhat.grid.k[i]*uhat[j]).norm()
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
            :label:

            S =
            \left<
                \frac{\partial u_i}{\partial x_k}
                \frac{\partial u_i}{\partial x_l}
                \frac{\partial u_l}{\partial x_k}
            \right>
        """
        gradu = uhat.grad().to_physical()
        S = [
            (gradu[i, k]*gradu[i, l]*gradu[l, k]).to_spectral().get_mode([0, 0, 0])
            for i in range(3) for k in range(3) for l in range(3)
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
    """A diagnostic class for velocity spectra."""
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file"""
        kmag = numpy.sqrt(numpy.sum(uhat.grid.k**2, axis=0))
        kmax = numpy.amax(kmag)
        nbins = int(max(kmag.shape)/2)
        dk = kmax/nbins
        wavenumbers = dk*numpy.arange(nbins+1)
        spectrum = numpy.zeros([nbins+1, 3])
        ispectrum = numpy.zeros([nbins+1], dtype=int)

        for k, u in numpy.nditer([kmag, (uhat[:3]*uhat[:3].conjugate()).real()]):
            spectrum[int(k/dk)] += u
            ispectrum[int(k/dk)] += 1
        spectrum *= 4*numpy.pi*(wavenumbers**2/ispectrum)[:, numpy.newaxis]

        for i, s in zip(wavenumbers, spectrum):
            self.outfile.write("{} {}\n".format(i, sum(s)))
        self.outfile.write("\n\n")
        self.outfile.flush()


class FieldDump(Diagnostic):
    """Full spectral field file dumps"""
    def diagnostic(self, time, equations, uhat):
        """Write the solution fields in numpy format"""
        numpy.save(
            "data{:04g}".format(time),
            uhat
            )
