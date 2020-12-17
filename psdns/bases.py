"""Spectral bases for solvers.

This module defines the various spectral bases that can be used to
write solvers.
"""
import mpi4py
import mpi4py_fft
import numpy


class FFT(object):
    """Fast Fourier Transform

    This is an interface to the FFTW library, which currently is
    implemented on top of the `mpi4py_fft
    <https://mpi4py-fft.readthedocs.io>`_ package.
    """
    def __init__(self, N, padding=1.5, **kwargs):
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
        self.rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        # The mpi4py PFFT instance for this transform
        self.fft = mpi4py_fft.PFFT(
            comm=mpi4py.MPI.COMM_WORLD,
            shape=[N, N, N],
            padding=3*[padding],
            dtype=numpy.double,
        )
        #: The physical space grid coordinates of the local array
        self.x = (2*numpy.pi/self.N)*numpy.mgrid[self.fft.local_slice(False)]
        k = numpy.mgrid[self.fft.local_slice(True)]
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
        self.to_physical(1j*(self.k[1]*a[2]-self.k[2]*a[1]), b[0])
        self.to_physical(1j*(self.k[2]*a[0]-self.k[0]*a[2]), b[1])
        self.to_physical(1j*(self.k[0]*a[1]-self.k[1]*a[0]), b[2])
        return b

    def cross(self, a, b, c):
        """Cross-product of two physical space variables, in spectral space.
        """
        self.to_spectral(a[1]*b[2]-a[2]*b[1], c[0])
        self.to_spectral(a[2]*b[0]-a[0]*b[2], c[1])
        self.to_spectral(a[0]*b[1]-a[1]*b[0], c[2])
        return c

    def to_spectral(self, *args):
        """Transform from physical to spectral space.
        """
        return self.fft.forward(*args)

    def to_physical(self, *args):
        """Transform from spectral to physical space.
        """
        return self.fft.backward(*args)

    def spectral_array(self, *args, **kwargs):
        """Return a new local spectral array.
        """
        return mpi4py_fft.newDistArray(self.fft, True, *args, **kwargs)
        
    def physical_array(self, *args, **kwargs):
        """Return a new local physical array.
        """
        return mpi4py_fft.newDistArray(self.fft, False, *args, **kwargs)

    def spectral_norm(self, u):
        """Return the L2 norm of a spectral array.
        """
        return mpi4py.MPI.COMM_WORLD.reduce(numpy.sum(
            (u[...,0]*numpy.conjugate(u[...,0])).real
            +2*numpy.sum(
               (u[...,1:]*numpy.conjugate(u[...,1:])).real,
               axis=-1
               )
            ))

    def physical_norm(self, u):
        """Return the L2 norm of a physical array.
        """
        return mpi4py.MPI.COMM_WORLD.reduce(numpy.sum(u*u)/self.N**3)
