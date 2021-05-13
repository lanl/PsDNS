import unittest

import numpy
from numpy import testing as nptest

from psdns.bases import SpectralGrid, SpectralArray, PhysicalArray
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray


# With mpi4py, Symmetries passes, but norms fail.

_domains = [
    #: A list of domains on which to run individual tests.  Each domain
    #: is a 2-tuple which is passed as the arguments to the
    #: :class:`~psdns.bases.SpectralGrid` constructor.
    ( (9, 9, 9), (9, 9, 9) ),
    ( (8, 9, 9), (8, 9, 9) ),
    ( (9, 8, 9), (9, 8, 9) ),
    ( (8, 8, 9), (8, 8, 9) ),
    ( (9, 9, 8), (9, 9, 8) ),
    ( (8, 9, 8), (8, 9, 8) ),
    ( (9, 8, 8), (9, 8, 8) ),
    ( (8, 8, 8), (8, 8, 8) ),
    ( (9, 9, 9), (12, 12, 12) ),
    ( (8, 9, 9), (12, 12, 12) ),
    ( (9, 8, 9), (12, 12, 12) ),
    ( (8, 8, 9), (12, 12, 12) ),
    ( (9, 9, 8), (12, 12, 12) ),
    ( (8, 9, 8), (12, 12, 12) ),
    ( (9, 8, 8), (12, 12, 12) ),
    ( (8, 8, 8), (12, 12, 12) ),
    ( (8, 8, 8), (12, 12, 8) ),
    ]


def random_spectral_array(grid, **kwargs):
    """Return a random :class:`~psdns.bases.SpectralArray`."""
    # In order to assure that this spectral array has the approriate
    # symmetries, start with a physical space array, and transform to
    # spectral space.
    return PhysicalArray(numpy.random.random(grid.x.shape[1:]), grid, **kwargs).to_spectral()


def random_physical_array(grid, **kwargs):
    """Return a random :class:`~psdns.bases.PhysicalArray`."""
    # In order to assure that this physical array has no spectral
    # content in wave numbers which are truncated on this grid, start
    # with a spectral space array, and transform to physical space. 
    return random_spectral_array(grid, **kwargs).to_physical()


class TestSymmetries(unittest.TestCase):
    r"""Test that spectral transforms have the correct symmetries.
    """
    def test_z_zero(self):
        r"""Test Hermitian symmetry when z=0"""
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                for k in range(1, sdims[0]//2):
                    for l in range(1, sdims[1]//2):
                        with self.subTest(k=k, l=l):
                            self.assertAlmostEqual(
                                s[k, l, 0],
                                s[-k, -l, 0].conjugate()
                                )

    def test_z_max(self):
        r"""Test Hermitian symmetry when z=Nz/2"""
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                if sdims[2] % 2 == 0:
                    s = random_spectral_array(SpectralGrid(sdims, pdims))
                    for k in range(1, sdims[0]//2):
                        for l in range(1, sdims[1]//2):
                            with self.subTest(k=k, l=l):
                                self.assertAlmostEqual(
                                    s[k, l, 0],
                                    s[-k, -l, 0].conjugate()
                                    )

    def test_real_corners(self):
        r"""Test for real values in corners"""
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                for k in [ 0, -sdims[0]//2 ] if sdims[0] == pdims[0] and sdims[0] % 2 == 0 else [ 0 ]:
                    for l in [ 0, -sdims[1]//2 ] if sdims[1] == pdims[1] and sdims[1] % 2 == 0 else [ 0 ]:
                        for m in [ 0, sdims[2]//2 ] if sdims[2] == pdims[2] and sdims[2] % 2 == 0 else [ 0 ]:
                            with self.subTest(k=k, l=l, m=m):
                                self.assertAlmostEqual(s[k, l, m].imag, 0)


class TestProperties(unittest.TestCase):
    """Test that various mathematical properties of the FFT obeyed.
    """
    def test_round_trip1(self):
        """Transforming to spectral and back to physical returns the original value.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                p = random_physical_array(SpectralGrid(sdims, pdims))
                nptest.assert_allclose(
                    numpy.asarray(p.to_spectral().to_physical()),
                    numpy.asarray(p)
                    )

    def test_round_trip2(self):
        """Transforming to physical and back to spectral returns the original value.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                nptest.assert_allclose(
                    numpy.asarray(s.to_physical().to_spectral()),
                    numpy.asarray(s)
                    )

    def test_linear1(self):
        """Physical-to-spectral transforms are linear.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                grid = SpectralGrid(sdims, pdims)
                p1 = random_physical_array(grid)
                p2 = random_physical_array(grid)
                a = numpy.random.rand()
                nptest.assert_allclose(
                    numpy.asarray((p1+a*p2).to_spectral()),
                    numpy.asarray(p1.to_spectral()+a*p2.to_spectral())
                    )

    def test_linear2(self):
        """Spectral-to-physical transforms are linear.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                grid = SpectralGrid(sdims, pdims)
                s1 = random_spectral_array(grid)
                s2 = random_spectral_array(grid)
                a = numpy.random.rand()
                nptest.assert_allclose(
                    numpy.asarray((s1+a*s2).to_physical()),
                    numpy.asarray(s1.to_physical()+a*s2.to_physical())
                    )

    def test_norm(self):
        """Spectral norm should match physical space norm.

        .. note::

            When the :data:`use_mpi4py_fft_scaling` flag is set, certain
            domains are not tested, because of an inconsistency in the
            treatment of certain modes, see :meth:`test_norm_pointwise`.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                if (s.grid._aliasing_strategy in [ '', 'mpi4py' ] and
                    ((s.grid.x.shape[1] > s.grid.k.shape[1] and s.grid.k.shape[1] % 2 == 0) or
                     (s.grid.x.shape[2] > s.grid.k.shape[2] and s.grid.k.shape[2] % 2 == 0))):
                    print(f"skipping test_norm with sdims={sdims}, pdims={pdims}: expected failure")
                    continue
                self.assertAlmostEqual(
                    s.norm(),
                    s.to_physical().norm()
                    )

    def test_norm2(self):
        """Physical norm should match spectral space norm.

        .. note::

            When the :data:`use_mpi4py_fft_scaling` flag is set, certain
            domains are not tested, because of an inconsistency in the
            treatment of certain modes, see :meth:`test_norm_pointwise`.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                p = random_physical_array(SpectralGrid(sdims, pdims))
                if (p.grid._aliasing_strategy in [ '', 'mpi4py' ] and
                    ((p.grid.x.shape[1] > p.grid.k.shape[1] and p.grid.k.shape[1] % 2 == 0) or
                     (p.grid.x.shape[2] > p.grid.k.shape[2] and p.grid.k.shape[2] % 2 == 0))):
                    print(f"skipping test_norm with sdims={sdims}, pdims={pdims}: expected failure")
                    continue
                self.assertAlmostEqual(
                    p.norm(),
                    p.to_spectral().norm()
                    )

    # def test_norm_pointwise(self):
    #     """Spectral norm should match physical space norm for a single mode.

    #     This test is specifically intended to help catch errors due to
    #     scaling of particular modes.  It takes compares the norm of a
    #     physical space array containing a single cosine mode to the norm
    #     of its transform in spectral.  These should be equal.  However,
    #     when the :data:`~psdns.bases.use_mpi4py_fft_scaling` flag is
    #     set, this is not the case for certain modes, as in the following
    #     example.

    #     Consider a case for which the first index is padded, and the
    #     spectral arrays are truncated to an even number of modes.  (The
    #     same argument can be applied to the second index.)  Before
    #     truncating, the transform contains both modes ``N[0]//2`` and
    #     ``-(N[0]//2)``, however, after truncating, only the latter mode
    #     is retained.

    #     Consider the physical space array initialized by::

    #     p = 2*cos(-(N[0]//2)*x)

    #     When transformed, the resulting array has two non-zero modes::

    #     s[-4,0,0] == s[4,0,0] == 1

    #     The truncation algorithm sets::

    #     s[-4,0,0] == 2

    #     and the other non-zero mode is not kept.

    #     In this case we have::

    #     p.norm() == 2

    #     and::

    #     s.norm() == 4

    #     This is not a bug in the implementation, it is an inconsistency
    #     in the algorithm used by `mpi4py-fft
    #     <https://mpi4py-fft.readthedocs.io>`_.
          

    #     1. Initialize p=2*cos(-4x).
    #     2. Transform has only non-zero modes of u[-4,0,0]=u[4,0,0]=1.
    #     3. Truncation sets u[-4,0,0]=2.  u[4,:,:4] is not retained.
    #     4. |p| = 2, |s| = 4.
    #     5. Back transforming, the padding algorithm sets
    #     u[-4,0,0]=u[4,0,0]=1.
    #     6. This is actually correct, and the transform is p=2*cos(-4x).



    #     CLEAN THIS UP!!!

    #     Consider the case N=(8,9,9), padding=1.5.

    #     1. Initialize p=2*cos(-4x+y).
    #     2. Transform has only non-zero modes of u[-4,1,0]=u[4,-1,0]=1.
    #     3. Truncation sets u[-4,1,0]=u[-4,-1,0]=1.  u[4,:,:] is not
    #     retained.
    #     4. |p| = |s| = 2.
    #     5. Back transforming, the padding algorithm sets
    #     u[-4,1,0]=u[4,1,0]=1/2.
    #     6. This is bad, because u[4,-1,0] and u[-4,-1,0], which are also
    #     in the padded array, are zero, so the array passed for transform
    #     is not Hermitian symmetric!  And, in fact, the result is not the
    #     original physical space array, i.e., this fails the round-trip
    #     test.

    #     However,
    #     """
    #     for N, padding in _domains:
    #         with self.subTest(sdims=sdims, pdims=pdims):
    #             skips = 0
    #             for k in range(-(N[0]//2), (N[0]+1)//2):
    #                 for l in range(-(N[1]//2), (N[1]+1)//2):
    #                     for m in range(N[2]//2+1):
    #                         # It might seem easier to set this test up
    #                         # in spectral space, but it is not, because
    #                         # that requires special case handling for
    #                         # edge and corner points.
    #                         theta = numpy.random.rand()
    #                         p = PhysicalArray((), SpectralGrid(N, padding))
    #                         p[...] = 2*numpy.cos(k*p.x[0]+l*p.x[1]+m*p.x[2]+theta)
    #                         if (use_mpi4py_fft_scaling and
    #                             ((p.x.shape[1] > p.k.shape[1]
    #                               and -2*k == p.k.shape[1] or k == 0) and
    #                              (p.x.shape[2] > p.k.shape[2]
    #                               and -2*l == p.k.shape[2] or l == 0))):
    #                             skips += 1
    #                             continue
    #                         with self.subTest(k=k, l=l, m=m):
    #                             self.assertAlmostEqual(
    #                                 p.norm(),
    #                                 p.to_spectral().norm()
    #                                 )
    #             if skips:
    #                 print(f"test_norm_pointwise with N={N}, skipped {skips} tests, expected failure")


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



    """

    In 1-d, truncation of a complex transform to an even number of
    modes leaves a mode with a symmetry issue.  EXPLICATE

    In 3-d, in general truncation of interior or corner modes is not a
    problem.  Howerver, if Nx, Ny even, there is a strange type of
    filtering.  Furthermore, when back transforming, the edge modes
    are an issue...

    Consider M>N physical and spectral modes respectively, and N is
    even.  Before truncation we have two interior modes,

      \hat{u}[-(N//2), l, m] <===> \hat{u}[N//2, -l, -m]
      \hat{u}[N//2, l, m] <===> \hat{u}[-(N//2), -l, -m]
    
    These have the physical space representation

      \cos -(N//2) x + l y + m z
      \cos N//2 x + l y + m z
    
    So truncating gets rid of two modes with the same wavenumber
    magnitude.

    More importantly, consider the special case of (the same would
    apply for l=M//2 if there is no truncation in the second index).

      \hat{u}[-(N//2), 0, 0]

    The H.S. of this mode exists on the untruncated grid, [(N//2), 0,
    0], but not on the truncated grid.  If the reverse transform is
    performed with this H.S. mode zero-padded, then the results are
    undefined, because the array is not H.S., so the inverse FFT is
    not real.

    Therefore, some modification must be made to these modes.
    """
    def assert_forward_backward(self, p, s):
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
        """Forward and backward transforms of a single mode are correct.

        Note that the various cases could be broken out into different
        tests, however, this implementation, that iterates over all
        modes in one routine was chosen to reduce the likelihood that
        tests of certain modes were inadvertantly omitted.

        Three kinds of points:
        1. Interiour
        2. Edge (m==0 or m==N/2)
        3. Corner (k==l==m==0 or N/2).
        With anti-aliasing:
        4. Truncated modes only need to be tested for forward
        transform.
        5. If we truncate in either of the first two dimensions to an
        even number of point, then we will have H.S. pairs of edge
        points where only one point is retained.  The algorithm needs to
        fix this before back transforming, otherwise the padded array
        returned to the inverse real-FFT routine will not, in general,
        be H.S., and therefore the results will be undefined.

        The simplest solution is to set the member of the pair which is
        kept to zero.  This is equivalent to always setting the spectral
        size of the array to an odd number.

        The next option would be to pad the H.S. mode with the complex
        conjugate of the retained mode.

        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                skips = 0
                for k in range(-(pdims[0]//2), (pdims[0]+1)//2):
                    for l in range(-(pdims[1]//2), (pdims[1]+1)//2):
                        for m in range(pdims[2]//2+1):
                            with self.subTest(k=k, l=l, m=m):
                                grid = SpectralGrid(sdims, pdims)
                                s = SpectralArray((), grid)
                                p = PhysicalArray((), grid)
                                if (grid._aliasing_strategy in [ '', 'mpi4py' ] and
                                    ((p.grid.x.shape[1] > p.grid.k.shape[1] and abs(2*k) == p.grid.k.shape[1]) or
                                     (p.grid.x.shape[2] > p.grid.k.shape[2] and abs(2*l) == p.grid.k.shape[2]))):
                                    skips += 1
                                    continue
                                theta = numpy.random.rand()
                                if ((-2*k > sdims[0] or 2*k > sdims[0]-1) or
                                    (-2*l > sdims[1] or 2*l > sdims[1]-1) or
                                    2*m > sdims[2]):
                                    # Truncated modes
                                    p[...] = 2*numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta)
                                    # If the truncated mode is k==4, and m==0
                                    # then the spectral result is not
                                    # actually zero, because there is
                                    # content in the k==-4 mode, which
                                    # is retained.
                                    if (((2*k==sdims[0] and 2*abs(l)<=sdims[1]) or 
                                         (2*l==sdims[1] and 2*abs(k)<=sdims[0])) and
                                        (m==0 or 2*m == pdims[2] == sdims[2])):
                                        s[-k,-l,m] = numpy.exp(-1j*theta)
                                        with self.subTest(type='HS pos'):
                                            self.assert_forward_backward(p, s)
                                        continue
                                    with self.subTest(dir='forward', type='trunc'):
                                        nptest.assert_almost_equal(
                                            numpy.asarray(p.to_spectral()),
                                            numpy.asarray(s)
                                            )
                                    continue
                                elif (((pdims[0]>sdims[0] and -2*k==sdims[0] and 2*abs(l)<=sdims[1]) or
                                       (pdims[1]>sdims[1] and -2*l==sdims[1] and 2*abs(k)<=sdims[0])) and
                                      (m==0 or 2*m == sdims[2])):
                                    s[k,l,m] = numpy.exp(1j*theta)
                                    p[...] = 2*numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta)
                                    type = 'HS neg'
                                elif ((k == 0 or -2*k == pdims[0]) and
                                      (l == 0 or -2*l == pdims[1]) and
                                      (m == 0 or 2*m == pdims[2] == sdims[2])):
                                    # Corner points must be real
                                    s[k, l, m] = 1.0
                                    p[...] = numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2])
                                    type = 'corner'
                                elif m == 0 or 2*m == pdims[2]:
                                    # Edge points must have a
                                    # Hermitian symmetric mode in the
                                    # spectral array.
                                    #
                                    # The only time the indicies -k
                                    # and k or -l and l point to the
                                    # same entry are if

                                    #
                                    # k >=  N[0] - k
                                    #
                                    # 2 k >= N[0]
                                    #
                                    # I believe this should always be
                                    # caught by option 1
                                    # (truncation).
                                    #
                                    s[k, l, m] = numpy.exp(1j*theta)
                                    s[-k, -l, m] = numpy.exp(-1j*theta)
                                    p[...] = 2*numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta)
                                    type = 'edge'
                                else:
                                    # Interior points have a Hermitian
                                    # symmetric ghost point that is
                                    # not carried.
                                    s[k, l, m] = numpy.exp(1j*theta)
                                    p[...] = 2*numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta)
                                    type = 'interior'
                                with self.subTest(type=type):
                                    self.assert_forward_backward(p, s)
                if skips:
                    print(f"test_single_mode skipping {skips} subtests with sdims={sdims}, pdims={pdims}: expected failure")


class TestMPI4PyFFT(unittest.TestCase):
    """Test that our transforms return the same results as mpi4py-fft.
    """
    def test_mpi4py_fft_forward(self):
        """Physical-to-spectral transforms match mpi4py-fft.
        """
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                fft = PFFT(
                    MPI.COMM_WORLD,
                    sdims,
                    padding=numpy.asarray(pdims)/numpy.asarray(sdims),
                    axes=(0, 1, 2),
                    dtype=numpy.float,
                    grid=(-1,)
                    )
                p = random_physical_array(SpectralGrid(sdims, pdims, aliasing_strategy='mpi4py'))
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
        for sdims, pdims in _domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                fft = PFFT(
                    MPI.COMM_WORLD,
                    sdims,
                    padding=numpy.asarray(pdims)/numpy.asarray(sdims),
                    axes=(0, 1, 2),
                    dtype=numpy.float,
                    grid=(-1,)
                    )
                s = random_spectral_array(SpectralGrid(sdims, pdims, aliasing_strategy='mpi4py'))
                u_hat = newDistArray(fft, True)
                u_hat[...] = s
                u = fft.backward(u_hat, normalize=False)
                nptest.assert_allclose(
                    u,
                    numpy.asarray(s.to_physical())
                    )
