import unittest

import numpy
from numpy import testing as nptest

from psdns.bases import SpectralArray, PhysicalArray, spectral_grid
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray


def random_spectral_array(k, x):
    return PhysicalArray(numpy.random.random(x.shape[1:]), k, x).to_spectral()


def random_physical_array(k, x):
    return random_spectral_array(k, x).to_physical()


class TestSymmetries(unittest.TestCase):
    r"""Test that spectral transforms have the correct symmetries.
    """
    domains = [
        (8, 1),
        (9, 1),
        (8, 1.5),
        (9, 1.5),
        (8, 2),
        (9, 2),
        ]

    def test_z_zero(self):
        r"""Test Hermitian symmetry when z=0"""
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                s = random_spectral_array(*spectral_grid(N, padding))
                for k in range(1, N//2):
                    for l in range(1, N//2):
                        with self.subTest(k=k, l=l):
                            self.assertAlmostEqual(
                                s[k, l, 0],
                                s[-k, -l, 0].conjugate()
                                )

    def test_z_max(self):
        r"""Test Hermitian symmetry when z=Nz/2"""
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                if N % 2 == 0:
                    s = random_spectral_array(*spectral_grid(N, padding))
                    for k in range(1, N//2):
                        for l in range(1, N//2):
                            with self.subTest(k=k, l=l):
                                self.assertAlmostEqual(
                                    s[k, l, 0],
                                    s[-k, -l, 0].conjugate()
                                    )

    def test_real_corners(self):
        r"""Test for real values in corners"""
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                s = random_spectral_array(*spectral_grid(N, padding))
                if padding == 1 and N % 2 == 0:
                    r = [ 0, N // 2 ]
                else:
                    r = [ 0 ]
                for k in r:
                    for l in r:
                        for m in r:
                            with self.subTest(k=k, l=l, m=m):
                                self.assertAlmostEqual(s[k, l, m].imag, 0)


class TestProperties(unittest.TestCase):
    """Test that various mathematical properties of the FFT obeyed.
    """
    domains = [
        ( (9, 9, 9), 1),
        ( (8, 9, 9), 1),
        ( (9, 8, 9), 1),
        ( (8, 8, 9), 1),
        ( (9, 9, 8), 1),
        ( (8, 9, 8), 1),
        ( (9, 8, 8), 1),
        ( (8, 8, 8), 1),
        ( (9, 9, 9), 1.5),
        ( (8, 9, 9), 1.5),
        ( (9, 8, 9), 1.5),
        ( (8, 8, 9), 1.5),
        ( (9, 9, 8), 1.5),
        ( (8, 9, 8), 1.5),
        ( (9, 8, 8), 1.5),
        ( (8, 8, 8), 1.5),
        ]

    def test_round_trip1(self):
        """Transforming to spectral and back to physical returns the original value.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                k, x = spectral_grid(N, padding)
                p = random_physical_array(k, x)
                nptest.assert_allclose(
                    numpy.asarray(p.to_spectral().to_physical()),
                    numpy.asarray(p)
                    )

    def test_round_trip2(self):
        """Transforming to physical and back to spectral returns the original value.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                k, x = spectral_grid(N, padding)
                s = random_spectral_array(k, x)
                nptest.assert_allclose(
                    numpy.asarray(s.to_physical().to_spectral()),
                    numpy.asarray(s)
                    )

    def test_linear1(self):
        """Physical-to-spectral are linear.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                grid = spectral_grid(N, padding)
                p1 = random_physical_array(*grid)
                p2 = random_physical_array(*grid)
                a = numpy.random.rand()
                nptest.assert_allclose(
                    numpy.asarray((p1+a*p2).to_spectral()),
                    numpy.asarray(p1.to_spectral()+a*p2.to_spectral())
                    )

    def test_linear2(self):
        """Spectral-to-physical are linear.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                grid = spectral_grid(N, padding)
                s1 = random_spectral_array(*grid)
                s2 = random_spectral_array(*grid)
                a = numpy.random.rand()
                nptest.assert_allclose(
                    numpy.asarray((s1+a*s2).to_physical()),
                    numpy.asarray(s1.to_physical()+a*s2.to_physical())
                    )

    def test_norm(self):
        """Spectral norm should match physical space norm.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                s = random_spectral_array(*spectral_grid(N, padding))
                self.assertAlmostEqual(
                    s.norm(),
                    s.to_physical().norm()
                    )


class TestSingleMode(unittest.TestCase):
    """One-Dimensional Transforms

    --------------------------

    For a FFT of real data, the result is Hermitian symmetric:

    .. math::

    \hat{u}(k) = \hat{u}^*(-k)

    For the 1-d case, it is sufficient to retain half the modes.  If
    the transform length, N, is odd, then the modes are

    k = 0, 1, ... (N-1)/2

    Each of the Fourier coefficients are complex, so the N pieces of
    real data transform to (N+1)/2 complex numbers, which can be
    written as N+1 real values (each complex number has a real and
    imaginary part).  However, Hermitian symmetry implies the zero
    mode is purely real, so there are still only N independent pieces
    of data.

    If N is even, the modes are

    k = 0, 1, ... N/2

    which is N/2+1 complex numbers, or N+2 real numbers, but both the
    zero and N/2 modes are real, so there are still only N pieces of
    data.

    For anit-aliasing purposes we may wish to pad the data.  In that
    case, we start with M real points, do an M point transform, to get
    M//2+1 complex coefficients, but we truncate the results array to
    N<M//2+1 modes.  When we back transfrom to physical space, we
    zero-pad the spectral array, and do an M point to recover the
    original M points in physical space.

    Notionally this truncated data could correspond to a physical grid
    of 2N-1 or 2N-2.  In the latter case, the highest mode would need
    to be purely real.  Some implementations specify the length of the
    truncated transform in this way, and impose this condition,
    however, there is no need to do so.

    Higher Dimensional Transforms
    -----------------------------

    In higher number of dimensions, the layout of arrays means it is
    no longer practical to discard all the Hermitian symmetric
    redundant modes.  This makes for some additional bookkeeping when
    doing these transforms.

    Without padding, we have three types of modes.  Here we assume there
    are [Nx, Ny, Nz] modes.  A given mode, \hat{u}[k,l,m], can be:

      1. Interior points are points where the Hermitian symmetric mode 
      is not retained in the array.

      2. Edge points are points where the Hermitian symmetric mode are
      part of the array.  These are modes for which

        m == 0 or 2*m == Nz 

      (and not a corner point).

      3. Corner points are points where the mode is Hermitian symmetric
      to itself, and therefore must be real.  These are modes for which

        k == 0  or 2*k == -Nx and
        l == 0  or 2*k == -Ny and
        m == 0  or 2*m == Nz
      







    s.shape = ( Nx, Ny, Nz//2+1 )
    p.shape = ( Mx, My, Mz )

    Note the truncated array has indicies that run:

    k = -(Nx-1)//2  ... (Nx-1)//2 or range(-(Nx-1)//2, (Nx+1)//2)
    l = -(Ny-1)//2  ... (Ny-1)//2 or range(-(Ny-1)//2, (Ny+1)//2)
    m = 0 ... Nz//2               or range(0, Nz//2+1)

    Essentially there are three possible scenarios.

    When we introduce padding, we have two additional scenarios.

    4. Truncated points are represented on the physical grid, but are
    filtered out on the spectral grid.

    k < -(Nx-1)//2 or k > (Nx-1)//2
    l < -(Ny-1)//2 or l > (Ny-1)//2
    m > Nz//2

    For these modes we only need to test the forward transform to make
    sure content in the physical field in these modes is filtered
    (truncated).  There is not backward transform test, because these
    modes don't exist in the spectral representation.

    5. As noted above, the spectral representation retains slightly
    more than half the modes, which means that certain pairs of
    Hermitian symmetric modes exist in the array.  If one member of
    any of these pairs are truncated, it must be recreated when the
    mode is restored during a back transform.  This is not actually a
    separate case to be tested, rather, these points are treated as
    "interior points" (I believe) and the test needs to make sure they
    also transform properly.

    Truncating in the z-direction is simple, just remove the extra
    modes.

    Consider N=(8, 9, 8) and padding=1.5.  So M=(12, 13, 12).  Consider
    the point:
      k, l, m = 4, 2, 2
    The Hermitian symmetric point is
      k, l, m = -4, -2, -2

    Starting with a physical space field of

      u(x,y,z) = 2 \cos ( - 4 x - 2 y -2 z + \theta )

    will result in a transform with

      \hat{u}[4, 2, 2] = \exp ( i \theta )

      \hat{u}[-4, -2, -2] = \exp ( - i \theta )

    When this is truncated, the 4, 2, 2 mode is lost, and the -4, -2, -2
    mode becomes

      \hat{u}[-4, -2, -2] = 2 \cos \theta



    """
    domains = [
        #( (9, 9, 9), 1),
        #( (9, 9, 8), 1),
        #( (8, 9, 8), 1),
        #( (9, 8, 8), 1),
        #( (8, 8, 8), 1),
        #( (9, 9, 9), 1.5),
        #( (9, 9, 8), 1.5),
        #( (9, 8, 8), 1.5),
        #( (8, 9, 8), 1.5),
        ( (8, 8, 8), 1.5),
        ]

    def forward_backward(self, p, s):
        with self.subTest(dir="forward"):
            nptest.assert_almost_equal(
                numpy.asarray(p.to_spectral()),
                numpy.asarray(s)
                )
        with self.subTest(dir="backward"):
            nptest.assert_almost_equal(
                numpy.asarray(s.to_physical()),
                numpy.asarray(p)
                )

    def test_single_mode(self):
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                grid = spectral_grid(N, padding)
                Nx, Ny, Nz = grid[0].shape[1:]
                Mx, My, Mz = grid[1].shape[1:]
                print(grid[0].shape)
                print(grid[1].shape)
                for k in range(-(Mx//2), (Mx+1)//2):
                    for l in range(-(My//2), (My+1)//2):
                        for m in range(0, (My+1)//2):
                            if k != 4 or l != 2 or m != 0:
                                continue
                            print(k, l, m)
                            with self.subTest(k=k, l=l, m=m):
                                s = SpectralArray((), *grid)
                                p = PhysicalArray((), *grid)
                                theta = 2*numpy.pi*(numpy.random.random()-0.5)
                                if (k > (Nx-1)//2 or k < -(Nx//2) or
                                    l > (Ny-1)//2 or l < -(Ny//2) or
                                    m >= Nz):
                                    p[...] = 2*numpy.cos(k*grid[1][0]+l*grid[1][1]+m*grid[1][2]+theta)
                                    nptest.assert_almost_equal(
                                        numpy.asarray(p.to_spectral()),
                                        numpy.asarray(s)
                                        )
                                else:
                                    print("missed point")
                                #    continue
                                # if (m == 0 or 2*m == Mz):
                                #     if ((k == 0 or 2*k == -Mx) and
                                #         (l == 0 or 2*l == -My)):
                                #         with self.subTest(type="corner"):
                                #             s[k, l, m] = 1.0
                                #             p[...] = numpy.cos(k*grid[1][0]+l*grid[1][1]+m*grid[1][2])
                                #             self.forward_backward(p, s)
                                #             continue
                                #     elif (2*k != -Nx or 2*l != -Ny):
                                #         with self.subTest(type="edge"):
                                #             s[k, l, m] = numpy.exp(1j*theta)
                                #             s[-k, -l, m] = numpy.exp(-1j*theta)
                                #             p[...] = 2*numpy.cos(k*grid[1][0]+l*grid[1][1]+m*grid[1][2]+theta)
                                #             self.forward_backward(p, s)
                                #             continue
                                # with self.subTest(type="interior"):
                                #     s[k, l, m] = numpy.exp(1j*theta)
                                #     p[...] = 2*numpy.cos(k*grid[1][0]+l*grid[1][1]+m*grid[1][2]+theta)
                                #     self.forward_backward(p, s)
                    
    
# class TestSingleMode(unittest.TestCase):
#     """Test that the transform of a single mode is correct.

#     Transform of physical size N1, N2, N3, no padding.

#     N3 is odd.  Then 

#     """
#     domains = [
#         (8, 1),
#         (9, 1),
#         (8, 1.5),
#         (9, 1.5),
#         #(8, 2),
#         (9, 2),
#         ]

#     def test_interior(self):
#         for N, padding in self.domains:
#             with self.subTest(N=N, padding=padding):
#                 for m in range(1, (N+1)//2):
#                     for l in range(-(N//2), (N+1)//2):
#                         for k in range(-(N//2), (N+1)//2):
#                             with self.subTest(k=k, l=l, m=m):
#                                 theta = 2*numpy.pi*(numpy.random.random()-0.5)
#                                 s = SpectralArray((), *spectral_grid(N, padding))
#                                 s[k,l,m] = numpy.exp(1j*theta)
#                                 p = PhysicalArray((), *spectral_grid(N, padding))
#                                 p[...] = 2*numpy.cos(
#                                     k*p.x[0]+l*p.x[1]+m*p.x[2]+theta
#                                     )
#                                 with self.subTest(dir="forward"):
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(p.to_spectral()),
#                                         numpy.asarray(s)
#                                         )
#                                 with self.subTest(dir="backward"):
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(s.to_physical()),
#                                         numpy.asarray(p)
#                                         )

#     def test_z_zero(self):
#         for N, padding in self.domains:
#             with self.subTest(N=N, padding=padding):
#                 for l in range(1, N//2):
#                     for k in range(1, N//2):
#                         with self.subTest(k=k, l=l):
#                             theta = 2*numpy.pi*(numpy.random.random()-0.5)
#                             s = SpectralArray((), *spectral_grid(N, padding))
#                             s[k,l,0] = numpy.exp(1j*theta)
#                             s[-k,-l,0] = s[k,l,0].conjugate()
#                             p = PhysicalArray((), *spectral_grid(N, padding))
#                             p[...] = 2*numpy.cos(
#                                 k*p.x[0]+l*p.x[1]+theta
#                                 )
#                             with self.subTest(dir="forward"):
#                                 nptest.assert_almost_equal(
#                                     numpy.asarray(p.to_spectral()),
#                                     numpy.asarray(s)
#                                     )
#                             with self.subTest(dir="backward"):
#                                 nptest.assert_almost_equal(
#                                     numpy.asarray(s.to_physical()),
#                                     numpy.asarray(p)
#                                     )

#     def test_z_max(self):
#         for N, padding in self.domains:
#             with self.subTest(N=N, padding=padding):
#                 if N % 2 == 0 and padding == 0:
#                     for l in range(1, N//2):
#                         for k in range(1, N//2):
#                             with self.subTest(k=k, l=l):
#                                 theta = 2*numpy.pi*(numpy.random.random()-0.5)
#                                 s = SpectralArray((), *spectral_grid(N, padding))
#                                 s[k,l,-(N//2)] = numpy.exp(1j*theta)
#                                 s[-k,-l,-(N//2)] = s[k,l,N//2].conjugate()
#                                 p = PhysicalArray((), *spectral_grid(N, padding))
#                                 p[...] = 2*numpy.cos(
#                                     k*p.x[0]+l*p.x[1]-(N//2)*p.x[2]+theta
#                                     )
#                                 with self.subTest(dir="forward"):
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(p.to_spectral()),
#                                         numpy.asarray(s)
#                                         )
#                                 with self.subTest(dir="backward"):
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(s.to_physical()),
#                                         numpy.asarray(p)
#                                         )

#     def test_corners(self):
#         for N, padding in self.domains:
#             with self.subTest(N=N, padding=padding):
#                 if padding == 1 and N % 2 == 0:
#                     r = [ 0, N // 2 ]
#                 else:
#                     r = [ 0 ]
#                 for k in r:
#                     for l in r:
#                         for m in r:
#                             with self.subTest(k=-k, l=-l, m=m):
#                                 s = SpectralArray((), *spectral_grid(N, padding))
#                                 s[-k,-l,m] = 1.0
#                                 p = PhysicalArray((), *spectral_grid(N, padding))
#                                 p[...] = numpy.cos(
#                                     -k*p.x[0]-l*p.x[1]+m*p.x[2]
#                                     )
#                                 with self.subTest(dir="forward"):
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(p.to_spectral()),
#                                         numpy.asarray(s)
#                                         )
#                                 with self.subTest(dir="backward"):
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(s.to_physical()),
#                                         numpy.asarray(p)
#                                         )

#     def test_truncated(self):
#         for N, padding in self.domains:
#             with self.subTest(N=N, padding=padding):
#                 s = SpectralArray((), *spectral_grid(N, padding))                
#                 M = int(N*padding)
#                 for m in range(0, M//2+1):
#                     for l in range(-(M-1)//2, (M-1)//2+1):
#                         for k in range(-(M-1)//2, (M-1)//2+1):
#                             if ((l < -(N//2) or l > (N-1)//2) or #and #OR!
#                                 (k < -(N//2) or k > (N-1)//2)):
#                                 with self.subTest(k=k, l=l, m=m):
#                                     theta = 2*numpy.pi*(numpy.random.random()-0.5)
#                                     p = PhysicalArray((), *spectral_grid(N, padding))
#                                     p[...] = 2*numpy.cos(
#                                         k*p.x[0]+l*p.x[1]+m*p.x[2]+theta
#                                         )
#                                     #print("CHECK", k, l, m, N, padding,
#                                      #         numpy.nonzero(numpy.around(p.to_spectral(),2)))
#                                     nptest.assert_almost_equal(
#                                         numpy.asarray(p.to_spectral()),
#                                         numpy.asarray(s)
#                                         )


class TestMPI4PyFFT(unittest.TestCase):
    """Test that our transforms return the same results as mpi4py-fft.
    """
    domains = [
        ( (9, 9, 9), 1),
        ( (8, 9, 9), 1),
        ( (9, 8, 9), 1),
        ( (8, 8, 9), 1),
        ( (9, 9, 8), 1),
        ( (8, 9, 8), 1),
        ( (9, 8, 8), 1),
        ( (8, 8, 8), 1),
        ( (9, 9, 9), 1.5),
        ( (8, 9, 9), 1.5),
        ( (9, 8, 9), 1.5),
        ( (8, 8, 9), 1.5),
        ( (9, 9, 8), 1.5),
        ( (8, 9, 8), 1.5),
        ( (9, 8, 8), 1.5),
        ( (8, 8, 8), 1.5),
        ]

    def test_mpi4py_fft_forward(self):
        """Physical-to-spectral transforms match mpi4py-fft.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                fft = PFFT(
                    MPI.COMM_WORLD,
                    N,
                    padding=3*[padding,],
                    axes=(0, 1, 2),
                    dtype=numpy.float,
                    grid=(-1,)
                    )
                p = random_physical_array(*spectral_grid(N, padding))
                u = newDistArray(fft, False)
                u[...] = p
                u_hat = fft.forward(u, normalize=True)
                s = p.to_spectral()
                nptest.assert_allclose(
                    u_hat,
                    numpy.asarray(s),
                    )
    
    def test_mpi4py_fft_backward(self):
        """Spectral-to-physical transforms match mpi4py-fft.
        """
        for N, padding in self.domains:
            with self.subTest(N=N, padding=padding):
                fft = PFFT(
                    MPI.COMM_WORLD,
                    N,
                    padding=3*[padding,],
                    axes=(0, 1, 2),
                    dtype=numpy.float,
                    grid=(-1,)
                    )
                s = random_spectral_array(*spectral_grid(N, padding))
                u_hat = newDistArray(fft, True)
                u_hat[...] = s
                u = fft.backward(u_hat, normalize=False)
                nptest.assert_allclose(
                    u,
                    numpy.asarray(s.to_physical())
                    )
