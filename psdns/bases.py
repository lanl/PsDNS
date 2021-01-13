"""Spectral bases for solvers.

This module defines the various spectral bases that can be used to
write solvers.
"""
import numpy


class FFT(object):
    """Fast Fourier Transform

    This is an interface to the FFTW library, which currently is
    implemented on top of the `mpi4py_fft
    <https://mpi4py-fft.readthedocs.io>`_ package.
    """
    def __init__(self, N, padding=1, **kwargs):
        """Initialize an FFT.

        :param int N: Domain size (see :attr:`N` for details)
        :param int padding: Factor for de-aliasing (1.5 corresponds to
        the 2/3-rule)
        """
        super().__init__(**kwargs)
        #: The size of the arrays.  The domain is always a cube, so
        #: total number of points is :math:`N^3`.  Note that *N*
        #: is the number of modes retained after de-aliasing, however,
        #: the actual number of points used in the transform includes
        #: the :param:`padding`.  
        self.N = int(padding*N)
        #: MPI process rank
        self.rank = 0
        self.x = (2*numpy.pi/self.N)*numpy.mgrid[:N,:N,:N]
        k = numpy.mgrid[:N,:N,:N//2+1]
        # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
        fftfreq = numpy.fft.fftfreq(N, 1/N)
        rfftfreq = numpy.fft.rfftfreq(N, 1/N)
        #: The spectral wave number coordinates of the local array
        self.k = numpy.array( [
            fftfreq[k[0]],
            fftfreq[k[1]],
            rfftfreq[k[2]]
        ] )
        self.k2 = numpy.sum(self.k*self.k, axis=0)

    def curl(self, a, b):
        """Curl of a spectral variable, in physical space.
        """
        b[0] = self.to_physical(1j*(self.k[1]*a[2]-self.k[2]*a[1]))
        b[1] = self.to_physical(1j*(self.k[2]*a[0]-self.k[0]*a[2]))
        b[2] = self.to_physical(1j*(self.k[0]*a[1]-self.k[1]*a[0]))
        return b

    def cross(self, a, b, c):
        """Cross-product of two physical space variables, in spectral space.
        """
        c[0] = self.to_spectral(a[1]*b[2]-a[2]*b[1])
        c[1] = self.to_spectral(a[2]*b[0]-a[0]*b[2])
        c[2] = self.to_spectral(a[0]*b[1]-a[1]*b[0])
        return c

    def to_spectral(self, u):
        """Transform from physical to spectral space.
        """
        return numpy.fft.rfftn(u, norm="ortho")

    def to_physical(self, uhat):
        """Transform from spectral to physical space.
        """
        return numpy.fft.irfftn(uhat, norm="ortho")

    def spectral_array(self, *args, **kwargs):
        """Return a new local spectral array.
        """
        return numpy.zeros(shape=self.k.shape, dtype=numpy.complex)
        
    def physical_array(self, *args, **kwargs):
        """Return a new local physical array.
        """
        return numpy.zeros(shape=self.x.shape)

    def spectral_norm(self, u):
        """Return the L2 norm of a spectral array.
        """
        return numpy.sum(
            (u[...,0]*numpy.conjugate(u[...,0])).real
            +2*numpy.sum(
               (u[...,1:]*numpy.conjugate(u[...,1:])).real,
               axis=-1
               )
            )

    def physical_norm(self, u):
        """Return the L2 norm of a physical array.
        """
        return numpy.sum(u*u)/self.N**3
