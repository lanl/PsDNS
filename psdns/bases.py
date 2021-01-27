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
    #   3. Support for 1, 2, or 3-d.
    N = numpy.broadcast_to(numpy.atleast_1d(N), (3,))
    padding = numpy.broadcast_to(numpy.atleast_1d(padding), (3,))
    xdims = (N*padding).astype(int)
    x = (2*numpy.pi/xdims[:,numpy.newaxis,numpy.newaxis,numpy.newaxis,]) \
      *numpy.mgrid[:xdims[0],:xdims[1],:xdims[2]]
    k = numpy.mgrid[:N[0],:N[1],:N[2]//2+1]
    # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
    fftfreq0 = numpy.fft.fftfreq(xdims[0], 1/xdims[0])[[*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)]]
    fftfreq1 = numpy.fft.fftfreq(xdims[1], 1/xdims[1])[[*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)]]
    rfftfreq = numpy.fft.rfftfreq(xdims[2], 1/xdims[2])[:N[2]//2+1]
    #: The spectral wave number coordinates of the local array
    k = numpy.array( [
        fftfreq0[k[0]],
        fftfreq1[k[1]],
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
        # Index array which picks out retained modes in a complex transform
        N = self.k.shape[1:]
        i0 = numpy.array([*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)])
        i1 = numpy.array([*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)])
        return SpectralArray(
            numpy.fft.rfftn(
                self,
                s=self.x.shape[1:],
                )[...,i0[:,numpy.newaxis],i1,:N[2]],
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
        N = self.k.shape[1:]
        i0 = numpy.array([*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)])
        i1 = numpy.array([*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)])
        s = numpy.zeros(
            shape = list(self.shape[:-3]) + list(self.x.shape[1:]),
            dtype = complex
            )
        s[...,i0[:,numpy.newaxis],i1,:N[2]] = self
        return PhysicalArray(
            numpy.fft.irfftn(
                s,
                s=self.x.shape[1:],
                ),
            self.k,
            self.x
            )*self.x[0].size

    def grad(self):
        return 1j*self[...,numpy.newaxis,:,:,:]*self.k
    
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
