"""Spectral bases for solvers.

This module defines the various spectral bases that can be used to
write solvers.
"""
from functools import cached_property
import warnings

import numpy

from mpi4py import MPI

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
    def __init__(self, sdims, pdims=None, aliasing_strategy='', comm=MPI.COMM_WORLD):
        """Return a new :class:`SpectralGrid` object.

        :param pdims: The size of the grid
        :type pdims: int or tuple(int)
        :param sdims: 
        :type sdims: int or tuple(int)
        :param aliasing_strategy: When truncating either of the first
          two axes to an even number of points, it is necessary to use a
          special treatment for the ``N//2`` mode.  When this flag is
          set to ``'mpi4py'``, the method used is the same as in the
          `mpi4py-fft <https://mpi4py-fft.readthedocs.io>`_ package.  Note
          that this method has some inconsistencies.  See :ref:`mpi4py-fft
          Compatability`.
        :type aliasing_strategy: '' (default) or 'mpi4py'
        """
        self._aliasing_strategy = aliasing_strategy
        self.comm = comm
        self.sdims = numpy.broadcast_to(numpy.atleast_1d(sdims), (3,))
        self.pdims = numpy.broadcast_to(numpy.atleast_1d(pdims), (3,)) \
          if pdims else self.sdims
        if self.sdims[0] > self.pdims[0] and self.sdims[0] % 2 == 0:
            warnings.warn("Using even number of modes in x: see the manual for why you don't want to do this")
        if self.sdims[1] > self.pdims[1] and self.sdims[1] % 2 == 0:
            warnings.warn("Using even number of modes in x: see the manual for why you don't want to do this")
        #: The slice of the physical space global array stored by this
        #: process
        self.physical_slices = [
            ( slice(i*self.pdims[0]//self.comm.size,
                    (i+1)*self.pdims[0]//self.comm.size),
              slice(0, self.pdims[1]),
              slice(0, self.pdims[2]) )
            for i in range(self.comm.size)
            ]
        self.local_physical_slice = self.physical_slices[self.comm.rank]
        #: The slice of the spectral space global array stored by this
        #: process
        self.spectral_slices = [
            ( slice(0, self.sdims[0]),
              slice(i*self.sdims[1]//self.comm.size,
                    (i+1)*self.sdims[1]//self.comm.size),
              slice(0, self.sdims[2]//2+1)
            )
            for i in range(self.comm.size)
            ]
        self.local_spectral_slice = self.spectral_slices[self.comm.rank]
        self.dx = 2*numpy.pi/self.pdims
        self.x = self.dx[:,numpy.newaxis,numpy.newaxis,numpy.newaxis] \
          *numpy.mgrid[self.local_physical_slice]
        self.k = self.kgrid()
        self.k2 =  numpy.sum(self.k*self.k, axis=0)
        self.slice1 = [
            MPI.DOUBLE_COMPLEX.Create_subarray(
                [ self.local_physical_slice[0].stop - self.local_physical_slice[0].start,
                  self.sdims[1],
                  self.sdims[2]//2+1
                ],
                [ self.local_physical_slice[0].stop - self.local_physical_slice[0].start,
                  s[1].stop - s[1].start,
                  self.sdims[2]//2+1
                ],
                [ 0, s[1].start, 0],
                ).Commit()
            for s in self.spectral_slices
            ]
        self.slice2 = [
            MPI.DOUBLE_COMPLEX.Create_subarray(
                [ self.pdims[0],
                  self.local_spectral_slice[1].stop - self.local_spectral_slice[1].start,
                  self.sdims[2]//2+1
                ],
                [ p[0].stop - p[0].start,
                  self.local_spectral_slice[1].stop - self.local_spectral_slice[1].start,
                  self.sdims[2]//2+1
                ],
                [ p[0].start, 0, 0 ],
                ).Commit()
            for p in self.physical_slices
            ]

    @cached_property
    def P(self):
        """Navier-Stokes pressure projection operator.

        Since this is only used by Navier-Stokes operators, it is
        maintained as a cached property.
        """
        return (numpy.eye(3)[:,:,None,None,None]
                -self.k[None,...]*self.k[:,None,...]
                /numpy.where(self.k2==0, 1, self.k2))
        
    def kgrid(self):
        k = numpy.mgrid[self.local_spectral_slice]
        # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
        fftfreq0 = numpy.fft.fftfreq(self.pdims[0], 1/self.pdims[0])[[*range(0, (self.sdims[0]+1)//2), *range(-(self.sdims[0]//2), 0)]]
        fftfreq1 = numpy.fft.fftfreq(self.pdims[1], 1/self.pdims[1])[[*range(0, (self.sdims[1]+1)//2), *range(-(self.sdims[1]//2), 0)]]
        rfftfreq = numpy.fft.rfftfreq(self.pdims[2], 1/self.pdims[2])[:self.sdims[2]//2+1]
        return numpy.array( [
            fftfreq0[k[0]],
            fftfreq1[k[1]],
            rfftfreq[k[2]]
            ] )
        

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
        N = self.grid.sdims
        i0 = numpy.array([*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)])
        i1 = numpy.array([*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)])
        # It would be more efficient to design MPI slices that fit
        # the non-contiguous array returned by the slicing here.
        t1 = numpy.ascontiguousarray(
            numpy.fft.rfft2(
                self._data,
                )[...,i1,:N[2]//2+1]
            )
        t2 = numpy.zeros(
            t1.shape[:-3]
            + ( self.grid.pdims[0],
                self.grid.local_spectral_slice[1].stop - self.grid.local_spectral_slice[1].start,
                self.grid.sdims[2]//2+1 ),
            dtype=complex
            )
        count = numpy.prod(self.shape[:-3], dtype=int)
        t1a = t1.reshape(count, *t1.shape[-3:])
        t2a = t2.reshape(count, *t2.shape[-3:])
        counts = [1] * self.grid.comm.size
        displs = [0] * self.grid.comm.size
        for t1b, t2b in zip(t1a, t2a):
            self.grid.comm.Alltoallw(
                [ t1b, counts, displs, self.grid.slice1 ],
                [ t2b, counts, displs, self.grid.slice2 ],
                )
        t3 = numpy.fft.fft(
            t2,
            axis=-3,
            )
        t3 = t3[...,i0,:,:]/numpy.prod(self.grid.pdims)
        # if self.grid._aliasing_strategy == 'mpi4py':
        #     if M[0] > N[0] and N[0] % 2 == 0:
        #         s[...,-(N[0]//2),:,:] = s[...,N[0]//2,:,:]+s[...,-(N[0]//2),:,:]
        #     if M[1] > N[1] and N[1] % 2 == 0:
        #         s[...,:,-(N[1]//2),:] = s[...,:,N[1]//2,:]+s[...,:,-(N[1]//2),:]
        return SpectralArray(t3, self.grid)

    def norm(self):
        n = self.grid.comm.reduce(numpy.average(self*self))
        if self.grid.comm.rank == 0:
            n /= self.grid.comm.size
        return n


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

    def get_view(self):
        extents = list(self.grid.sdims)
        subextents = list(self.grid.sdims)
        extents[2] = subextents[2] = self.grid.sdims[2]//2+1
        subextents[1] = self.grid.local_spectral_slice[1].stop - self.grid.local_spectral_slice[1].start
        starts = len(extents) * [ 0 ]
        starts[1] = self.grid.local_spectral_slice[1].start
        return MPI.DOUBLE_COMPLEX.Create_subarray(
            extents, subextents, starts
            ).Commit()
        
    def checkpoint(self, filename):
        view = self.get_view()
        fh = MPI.File.Open(self.grid.comm, filename, MPI.MODE_WRONLY|MPI.MODE_CREATE)
        fh.Set_view(0, filetype=view)
        fh.Write_all(self._data)
        fh.Close()
        view.Free()

    def read_checkpoint(self, filename):
        view = self.get_view()
        fh = MPI.File.Open(self.grid.comm, filename, MPI.MODE_RDONLY)
        fh.Set_view(0, filetype=view)
        fh.Read_all(self._data)
        fh.Close()        
        view.Free()
        
    def copy(self):
        return SpectralArray(self._data.copy(), self.grid)
        
    def conjugate(self):
        return SpectralArray(self._data.conjugate(), self.grid)

    def real(self):
        return SpectralArray(self._data.real, self.grid)

    def to_physical(self):
        N = self.grid.sdims
        i0 = numpy.array([*range(0, (N[0]+1)//2), *range(-(N[0]//2), 0)])
        i1 = numpy.array([*range(0, (N[1]+1)//2), *range(-(N[1]//2), 0)])
        s = numpy.zeros(
            self.shape[:-3]
            + ( self.grid.pdims[0],
                self.grid.local_spectral_slice[1].stop - self.grid.local_spectral_slice[1].start,
                self.grid.sdims[2]//2+1 ),
            dtype = complex
            )
        s[...,i0,:,:] = self
        t1 = numpy.ascontiguousarray(numpy.fft.ifft(s, axis=-3))
        t2 = numpy.zeros(
            self.shape[:-3]
            + ( self.grid.local_physical_slice[0].stop - self.grid.local_physical_slice[0].start,
                self.grid.sdims[1],
                self.grid.sdims[2]//2+1 ),
            dtype=complex
            )
        count = numpy.prod(self.shape[:-3], dtype=int)
        t1a = t1.reshape(count, *t1.shape[-3:])
        t2a = t2.reshape(count, *t2.shape[-3:])
        counts = [1] * self.grid.comm.size
        displs = [0] * self.grid.comm.size
        for t1b, t2b in zip(t1a, t2a):
            self.grid.comm.Alltoallw(
                [ t1b, counts, displs, self.grid.slice2 ],
                [ t2b, counts, displs, self.grid.slice1 ],
                )
        t25 = numpy.zeros(
            self.shape[:-3]
            + ( self.grid.local_physical_slice[0].stop - self.grid.local_physical_slice[0].start,
                self.grid.pdims[1],
                self.grid.pdims[2]//2+1 ),
            dtype=complex
            )
        t25[...,i1,:self.grid.sdims[2]//2+1] = t2
        t3 = numpy.fft.irfft2(
            t25,
            s=self.grid.x.shape[2:],
            axes=(-2, -1)
            )*numpy.prod(self.grid.pdims)
        # if self.grid._aliasing_strategy == 'mpi4py':
        #     if M[0] > N[0] and N[0] % 2 == 0:
        #         s[...,-(N[0]//2),:,:] *= 0.5
        #         s[...,N[0]//2,:,:] = s[...,-(N[0]//2),:,:]
        #     if M[1] > N[1] and N[1] % 2 == 0:
        #         s[...,:,-(N[1]//2),:] *= 0.5
        #         s[...,:,N[1]//2,:] = s[...,:,-(N[1]//2),:]
        # else:
        #     if M[0] > N[0] and N[0] % 2 == 0:
        #         s[...,N[0]//2,1:,0] = numpy.conjugate(s[...,-(N[0]//2),-1:0:-1,0])
        #         s[...,N[0]//2,0,0] = numpy.conjugate(s[...,-(N[0]//2),0,0])
        #         if M[2] == 2*(N[2]-1):
        #             s[...,N[0]//2,1:,-1] = numpy.conjugate(s[...,-(N[0]//2),-1:0:-1,-1])
        #             s[...,N[0]//2,0,-1] = numpy.conjugate(s[...,-(N[0]//2),0,-1])
        #     if M[1] > N[1] and N[1] % 2 == 0:
        #         s[...,1:,N[1]//2,0] = numpy.conjugate(s[...,-1:0:-1,-(N[0]//2),0])
        #         s[...,0,N[1]//2,0] = numpy.conjugate(s[...,0,-(N[0]//2),0])
        #         if M[2] == 2*(N[2]-1):
        #             s[...,1:,N[1]//2,-1] = numpy.conjugate(s[...,-1:0:-1,-(N[0]//2),-1])
        #             s[...,0,N[1]//2,-1] = numpy.conjugate(s[...,0,-(N[0]//2),-1])
        return PhysicalArray(t3, self.grid)

    def grad(self):
        return 1j*self[...,numpy.newaxis,:,:,:]*self.grid.k

    def div(self):
        return SpectralArray(
            1j*numpy.einsum("i...,i...->...", self.grid.k, self),
            self.grid
            )
    
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
            n = self.grid.comm.reduce(
                numpy.sum(
                    (self[...,0]*numpy.conjugate(self[...,0])).real
                    +2*numpy.sum(
                        (self[...,1:-1]*numpy.conjugate(self[...,1:-1])).real,
                        axis=-1
                        )
                    +(self[...,-1]*numpy.conjugate(self[...,-1])).real
                    )
                )
        else:
            n = self.grid.comm.reduce(
                numpy.sum(
                    (self[...,0]*numpy.conjugate(self[...,0])).real
                    +2*numpy.sum(
                        (self[...,1:]*numpy.conjugate(self[...,1:])).real,
                        axis=-1
                        )
                    )
                )
        return n

    def set_mode(self, mode, val):
        """Set a mode based on global array indicies."""
        mode[1] %= self.grid.sdims[1]
        if (mode[1]>=self.grid.local_spectral_slice[1].start and
            mode[1]<self.grid.local_spectral_slice[1].stop):
            self._data[mode[0], mode[1]-self.grid.local_spectral_slice[1].start, mode[2]] = val

    def get_mode(self, mode):
        # The following is a hack, but it makes communication
        # simpler.  A cleaner approach would to be to have the
        # processor that finds the mode send the data to the root,
        # instead of a global reduce.
        mode[1] %= self.grid.sdims[1]
        if (mode[1]>=self.grid.local_spectral_slice[1].start and
             mode[1]<self.grid.local_spectral_slice[1].stop):
             val = self._data[mode[0], mode[1]-self.grid.local_spectral_slice[1].start, mode[2]]
        else:
            val = 0
        return self.grid.comm.reduce(val)
