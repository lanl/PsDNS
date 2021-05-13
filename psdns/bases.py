"""Spectral bases for solvers.

This module defines the various spectral bases that can be used to
write solvers.
"""
import warnings

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
class SpectralGrid(object):
    def __init__(self, N, padding=1, aliasing_strategy=''):
        """Return a new :class:`SpectralGrid` object.

        Pre-text.

        :param N: The size of the grid
        :type N: int or tuple(int)
        :param padding:
        :type padding: float or tuple(float)
        :param aliasing_strategy: When truncating either of the first
          two axes to an even number of points, it is necessary to use a
          special treatment for the ``N//2`` mode.  When this flag is
          set to ``'mpi4py'``, the method used is the same as in the
          `mpi4py-fft <https://mpi4py-fft.readthedocs.io>`_ package.  Note
          that this method has some inconsistencies.  See :ref:`mpi4py-fft
          Compatability`.
        :type aliasing_strategy: 'mpi4py'


        Post text.

        :param more: more paramters
        """
        # To do:
        #   1. Arguments for domain size
        #   3. Support for 1, 2, or 3-d.
        self._aliasing_strategy = aliasing_strategy
        N = numpy.broadcast_to(numpy.atleast_1d(N), (3,))
        padding = numpy.broadcast_to(numpy.atleast_1d(padding), (3,))
        xdims = (N*padding).astype(int)
        self.x = (2*numpy.pi/xdims[:,numpy.newaxis,numpy.newaxis,numpy.newaxis,]) \
          *numpy.mgrid[:xdims[0],:xdims[1],:xdims[2]]
        k = numpy.mgrid[:N[0],:N[1],:N[2]//2+1]
        # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
        fftfreq0 = numpy.fft.fftfreq(xdims[0], 1/xdims[0])[[*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)]]
        fftfreq1 = numpy.fft.fftfreq(xdims[1], 1/xdims[1])[[*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)]]
        rfftfreq = numpy.fft.rfftfreq(xdims[2], 1/xdims[2])[:N[2]//2+1]
        self.k = numpy.array( [
            fftfreq0[k[0]],
            fftfreq1[k[1]],
            rfftfreq[k[2]]
            ] )
        if self.k.shape[1] > self.x.shape[1] and self.k.shape[1] % 2 == 0:
            warnings.warn("Using even number of modes in x: see the manual for why you don't want to do this")
        if self.k.shape[2] > self.x.shape[2] and self.k.shape[2] % 2 == 0:
            warnings.warn("Using even number of modes in y: see the manual for why you don't want to do this")
        

class PhysicalArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, shape_or_data, grid, dtype=float):
        try:
            if shape_or_data.shape[-3:] != grid.x.shape[1:]:
                raise ValueError(
                    "data array shape {} does not match grid shape {}".format(
                        shape_or_data.shape[-3:], grid.x.shape[1:]
                        )
                    )
            self._data = numpy.asarray(shape_or_data)
        except AttributeError:
            self._data = numpy.zeros(
                shape=list(shape_or_data)+list(grid.x.shape[1:]),
                dtype=dtype,
                )
        self.grid = grid
        self.shape = self._data.shape

    def __array__(self, dtype=None):
        return numpy.array(self._data, dtype, copy=False)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"PhysicalArray({str(self._data)})"

    def __getitem__(self, key):
        # Need to add checks for valid extents.  Specifically, for
        # slices check whether the return type should be a Physical
        # array or a regular numpy.ndarray.
        try:
            ret = PhysicalArray(self._data[key], self.grid)
        except ValueError:
            ret = self._data[key]
        return ret
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [ i._data if isinstance(i, PhysicalArray) else i
                   for i in inputs ]
        if 'out' in kwargs:
            kwargs['out'] = tuple(
                i._data if isinstance(i, SpectralArray)
                else i
                for i in kwargs['out']
                )
        ret = getattr(ufunc, method)(*inputs, **kwargs)
        if method == '__call__':
            # Need additional type and shape checking to see when to
            # return a PhysicalArray, when an ndarray, and when the
            # operation is invalid.
            return PhysicalArray(ret, self.grid)
        else:
            return ret

    def transpose(self, *indicies):
        return PhysicalArray(self._data.transpose(*indicies), self.grid)

    def clip(self, min=None, max=None):
        return PhysicalArray(self._data.clip(min, max), self.grid)
    
    def to_spectral(self):
        # Index array which picks out retained modes in a complex transform
        N = self.grid.k.shape[1:]
        M = self.grid.x.shape[1:]
        i0 = numpy.array([*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)])
        i1 = numpy.array([*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)])
        s = numpy.fft.rfftn(
            self._data,
            axes = (-3,-2,-1),
            )
        if self.grid._aliasing_strategy == 'mpi4py':
            if M[0] > N[0] and N[0] % 2 == 0:
                s[...,-(N[0]//2),:,:] = s[...,N[0]//2,:,:]+s[...,-(N[0]//2),:,:]
            if M[1] > N[1] and N[1] % 2 == 0:
                s[...,:,-(N[1]//2),:] = s[...,:,N[1]//2,:]+s[...,:,-(N[1]//2),:]
        return SpectralArray(
            s[...,i0[:,numpy.newaxis],i1,:N[2]]/self.grid.x[0].size,
            self.grid,
            )

    def norm(self):
        return numpy.average(self*self)


class SpectralArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, shape_or_data, grid, dtype=complex):
        try:
            if shape_or_data.shape[-3:] != grid.k.shape[1:]:
                raise ValueError(
                    "data array shape {} does not match grid shape {}".format(
                        shape_or_data.shape[-3:], grid.k.shape[1:]
                        )
                    )
            self._data = numpy.asarray(shape_or_data)
        except AttributeError:
            self._data = numpy.zeros(
                shape=list(shape_or_data)+list(grid.k.shape[1:]),
                dtype=dtype
                )
        self.grid = grid
        self.shape = self._data.shape

    def __array__(self, dtype=None):
        return numpy.array(self._data, dtype, copy=False)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"SpectralArray({str(self._data)})"

    def __getitem__(self, key):
        try:
            ret = SpectralArray(self._data[key], self.grid)
        except ValueError:
            ret = self._data[key]
        return ret

    def __setitem__(self, key, value):
        self._data[key] = value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            # Need additional type and shape checking to see when to
            # return a PhysicalArray, when an ndarray, and when the
            # operation is invalid.
            # There should be a cleaner way to do the following:
            if 'out' in kwargs:
                kwargs['out'] = tuple(
                    i._data if isinstance(i, SpectralArray)
                    else i
                    for i in kwargs['out']
                    )
            return SpectralArray(
                ufunc(
                    *[ i._data if isinstance(i, SpectralArray) else i for i in inputs ],
                    **kwargs
                    ),
                self.grid,
                )
        else:
            return NotImplemented

    def copy(self):
        return SpectralArray(self._data.copy(), self.grid)
        
    def conjugate(self):
        return SpectralArray(self._data.conjugate(), self.grid)

    def real(self):
        return SpectralArray(self._data.real, self.grid)

    def to_physical(self):
        N = self.grid.k.shape[1:]
        M = self.grid.x.shape[1:]
        i0 = numpy.array([*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)])
        i1 = numpy.array([*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)])
        s = numpy.zeros(
            shape = list(self.shape[:-3])
            + [ M[0], M[1], M[2]//2+1 ],
            dtype = complex
            )
        s[...,i0[:,numpy.newaxis],i1,:N[2]] = self
        if self.grid._aliasing_strategy == 'mpi4py':
            if M[0] > N[0] and N[0] % 2 == 0:
                s[...,-(N[0]//2),:,:] *= 0.5
                s[...,N[0]//2,:,:] = s[...,-(N[0]//2),:,:]
            if M[1] > N[1] and N[1] % 2 == 0:
                s[...,:,-(N[1]//2),:] *= 0.5
                s[...,:,N[1]//2,:] = s[...,:,-(N[1]//2),:]
        else:
            if M[0] > N[0] and N[0] % 2 == 0:
                s[...,N[0]//2,1:,0] = numpy.conjugate(s[...,-(N[0]//2),-1:0:-1,0])
                s[...,N[0]//2,0,0] = numpy.conjugate(s[...,-(N[0]//2),0,0])
                if M[2] == 2*(N[2]-1):
                    s[...,N[0]//2,1:,-1] = numpy.conjugate(s[...,-(N[0]//2),-1:0:-1,-1])
                    s[...,N[0]//2,0,-1] = numpy.conjugate(s[...,-(N[0]//2),0,-1])
            if M[1] > N[1] and N[1] % 2 == 0:
                s[...,1:,N[1]//2,0] = numpy.conjugate(s[...,-1:0:-1,-(N[0]//2),0])
                s[...,0,N[1]//2,0] = numpy.conjugate(s[...,0,-(N[0]//2),0])
                if M[2] == 2*(N[2]-1):
                    s[...,1:,N[1]//2,-1] = numpy.conjugate(s[...,-1:0:-1,-(N[0]//2),-1])
                    s[...,0,N[1]//2,-1] = numpy.conjugate(s[...,0,-(N[0]//2),-1])
        return PhysicalArray(
            numpy.fft.irfftn(
                s,
                s=self.grid.x.shape[1:],
                ),
            self.grid
            )*self.grid.x[0].size

    def grad(self):
        return 1j*self[...,numpy.newaxis,:,:,:]*self.grid.k
    
    def curl(self):
        """Curl of a spectral variable, in physical space.
        """
        return SpectralArray(
            1j*numpy.cross(self.grid.k, self, axis=0),
            self.grid
            )

    def norm(self):
        """Return the L2 norm of a spectral array.
        """
        if (self.grid.x.shape[3] % 2 == 0
            and self.grid.k.shape[3] == self.grid.x.shape[3]/2+1):
            return numpy.sum(
                (self[...,0]*numpy.conjugate(self[...,0])).real
                +2*numpy.sum(
                    (self[...,1:-1]*numpy.conjugate(self[...,1:-1])).real,
                    axis=-1
                    )
                +(self[...,-1]*numpy.conjugate(self[...,-1])).real
                )
        else:
            return numpy.sum(
                (self[...,0]*numpy.conjugate(self[...,0])).real
                +2*numpy.sum(
                    (self[...,1:]*numpy.conjugate(self[...,1:])).real,
                    axis=-1
                    )
                )
