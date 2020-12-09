import mpi4py
import mpi4py_fft
import numpy


class PSFourier(object):
    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        # The PFFT for this transform
        self.fft = mpi4py_fft.PFFT(
            comm=mpi4py.MPI.COMM_WORLD,
            shape=[N, N, N],
            #padding=[1.5, 1.5, 1.5],
            dtype=numpy.double,
        )
        # The physical space grid coordinates of the local array
        self.x = (2*numpy.pi/N)*numpy.mgrid[self.fft.local_slice(False)]
        k = numpy.mgrid[self.fft.local_slice(True)]
        # Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
        fftfreq = numpy.fft.fftfreq(N, 1/N)
        rfftfreq = numpy.fft.rfftfreq(N, 1/N)
        # The spectral wave number coordinates of the local array
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
        return self.fft.forward(*args)

    def to_physical(self, *args):
        return self.fft.backward(*args)

    def spectral_array(self, *args, **kwargs):
        return mpi4py_fft.newDistArray(self.fft, True, *args, **kwargs)
        
    def physical_array(self, *args, **kwargs):
        return mpi4py_fft.newDistArray(self.fft, False, *args, **kwargs)
