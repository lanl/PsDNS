"""Spectral bases for solvers.

This module defines the various spectral bases that can be used to
write solvers.
"""
import numpy

"""
There are three scenarios for creating a SpectralArray or PhysicalArray.

    1. At the beginning of the simulation a compatible spectral and
       physical grid need to be created.

         k, x = spectral_grid(...)
         PhysicalArray([3], k, x)

    2. A new array might be created using the grids from an existing
       array.

         PhysicalArray([3], uhat.k, uhat.x)

    3. A array view of existing data.

         PhysicalArray(data, uhat.k, uhat.x)

    f = lambda N: [*range(0, (N+1)//2), *range(-(N//2), 0)]

"""
def spectral_grid(N, padding=1):
    # To do:
    #   1. Arguments for domain size
    #   2. Allow array arguments
    #   3. Support for 1, 2, or 3-d.
    Nx = int(N*padding)
    x = (2*numpy.pi/Nx)*numpy.mgrid[:Nx,:Nx,:Nx]
    k = numpy.mgrid[:N,:N,:N//2+1]
    # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
    fftfreq = numpy.fft.fftfreq(N, 1/N)
    rfftfreq = numpy.fft.rfftfreq(N, 1/N)
    #: The spectral wave number coordinates of the local array
    k = numpy.array( [
        fftfreq[k[0]],
        fftfreq[k[1]],
        rfftfreq[k[2]]
        ] )
    return k, x


class PhysicalArray(numpy.ndarray):
    def __new__(cls, shape_or_data, k=None, x=None):
        try:
            if shape_or_data.shape[-3:] != x.shape[1:]:
                raise ValueError(
                    "data array shape {} does not match grid shape {}".format(
                        shape_or_data.shape[-3:], x.shape[1:]
                        )
                    )
            obj = shape_or_data.view(cls)
        except AttributeError:
            obj = super().__new__(
                cls,
                shape=list(shape_or_data)+list(x.shape[1:]),
                dtype=numpy.float
                )
        obj.k = k
        obj.x = x
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # This is a kludge.  Viewcasting shouldn't work if no k, x are
        # available, but if we disable this, then we cannot create a
        # view object on our way to adding attributes in to_spectral.
        self.k = getattr(obj, 'k', None)
        self.x = getattr(obj, 'x', None)

    def to_spectral(self):
        return SpectralArray(
            numpy.fft.rfftn(
                self,
                s=self.x.shape[1:],
                ), # Need to add downselect for padding
            self.k,
            self.x
            )/self.x[0].size
        
        
class SpectralArray(numpy.ndarray):
    def __new__(cls, shape_or_data, k=None, x=None):
        try:
            if shape_or_data.shape[-3:] != k.shape[1:]:
                raise ValueError(
                    "data array shape {} does not match grid shape {}".format(
                        shape_or_data.shape[-3:], k.shape[1:]
                        )
                    )
            obj = shape_or_data.view(cls)
        except AttributeError:
            obj = super().__new__(
                cls,
                shape=list(shape_or_data)+list(k.shape[1:]),
                dtype=numpy.complex
                )
        obj.k = k
        obj.x = x
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.k = getattr(obj, 'k', None)
        self.x = getattr(obj, 'x', None)

    def to_physical(self):
        return PhysicalArray(
            numpy.fft.irfftn(
                self,
                s=self.x.shape[1:],
                ),
            self.k,
            self.x
            )*self.x[0].size

    def curl(self):
        """Curl of a spectral variable, in physical space.
        """
        return SpectralArray(
            1j*numpy.cross(self.k, self, axis=0),
            self.k,
            self.x
            )
    
    def norm(self):
        """Return the L2 norm of a spectral array.
        """
        return numpy.sum(
            (self[...,0]*numpy.conjugate(self[...,0])).real
            +2*numpy.sum(
               (self[...,1:]*numpy.conjugate(self[...,1:])).real,
               axis=-1
               )
            )


# class FFT(object):
#     """Fast Fourier Transform

#     This is an interface to the FFTW library, which currently is
#     implemented on top of the `mpi4py_fft
#     <https://mpi4py-fft.readthedocs.io>`_ package.
#     """
#     def __init__(self, N, padding=1, **kwargs):
#         """Initialize an FFT.

#         :param int N: Domain size (see :attr:`N` for details)
#         :param int padding: Factor for de-aliasing (1.5 corresponds to
#         the 2/3-rule)
#         """
#         super().__init__(**kwargs)
#         #: The size of the arrays.  The domain is always a cube, so
#         #: total number of points is :math:`N^3`.  Note that *N*
#         #: is the number of modes retained after de-aliasing, however,
#         #: the actual number of points used in the transform includes
#         #: the :param:`padding`.  
#         self.N = int(padding*N)
#         #: MPI process rank
#         self.rank = 0
#         self.x = (2*numpy.pi/self.N)*numpy.mgrid[:N,:N,:N]
#         k = numpy.mgrid[:N,:N,:N//2+1]
#         # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
#         fftfreq = numpy.fft.fftfreq(N, 1/N)
#         rfftfreq = numpy.fft.rfftfreq(N, 1/N)
#         #: The spectral wave number coordinates of the local array
#         self.k = numpy.array( [
#             fftfreq[k[0]],
#             fftfreq[k[1]],
#             rfftfreq[k[2]]
#         ] )
#         self.k2 = numpy.sum(self.k*self.k, axis=0)

#     def curl(self, a, b):
#         """Curl of a spectral variable, in physical space.
#         """
#         b[0] = self.to_physical(1j*(self.k[1]*a[2]-self.k[2]*a[1]))
#         b[1] = self.to_physical(1j*(self.k[2]*a[0]-self.k[0]*a[2]))
#         b[2] = self.to_physical(1j*(self.k[0]*a[1]-self.k[1]*a[0]))
#         return b

#     def cross(self, a, b, c):
#         """Cross-product of two physical space variables, in spectral space.
#         """
#         c[0] = self.to_spectral(a[1]*b[2]-a[2]*b[1])
#         c[1] = self.to_spectral(a[2]*b[0]-a[0]*b[2])
#         c[2] = self.to_spectral(a[0]*b[1]-a[1]*b[0])
#         return c

#     def to_spectral(self, u):
#         """Transform from physical to spectral space.
#         """
#         return numpy.fft.rfftn(u, norm="ortho")

#     def to_physical(self, uhat):
#         """Transform from spectral to physical space.
#         """
#         return numpy.fft.irfftn(uhat, norm="ortho")

#     def spectral_array(self, *args, **kwargs):
#         """Return a new local spectral array.
#         """
#         return numpy.zeros(shape=self.k.shape, dtype=numpy.complex)
        
#     def physical_array(self, *args, **kwargs):
#         """Return a new local physical array.
#         """
#         return numpy.zeros(shape=self.x.shape)

#     def spectral_norm(self, u):
#         """Return the L2 norm of a spectral array.
#         """
#         return numpy.sum(
#             (u[...,0]*numpy.conjugate(u[...,0])).real
#             +2*numpy.sum(
#                (u[...,1:]*numpy.conjugate(u[...,1:])).real,
#                axis=-1
#                )
#             )

#     def physical_norm(self, u):
#         """Return the L2 norm of a physical array.
#         """
#         return numpy.sum(u*u)/self.N**3
