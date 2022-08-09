"""Tests of the FFT classes.

The fundemental building block of PsDNS are the FFT classes defined in
:mod:`~psdns.bases`.  These are tested in this module.  Because the
space of potential inputs to the FFT functions is so large, it is
necessary to identify specific tests which will be sufficiently
comprehensive to provide confidence for all possible FFTs.  Currently
there are three main tests:

  1. The :class:`TestSingleMode` class tests forward and backward
     transforms of a single Fourier mode.
  2. The classes :class:`TestSymmetries` and :class:`TestProperties` test
     whether the transforms obey all the properties they should.
  3. :class:`TestMPI4PyFFT` performs a code-to-code comparison between
     PsDNS and the `mpi4py <https://mpi4py-fft.readthedocs.io>`_
     package.

Each of these tests is performed using :attr:`domains` of several
different sizes, intended to exercise the various different
scenarios for domain truncation.

In principle, testing each mode individually, along with linearity,
should be enough to guarantee all possible FFTs, however, additional
tests are provided to provide redundant testing.

Although it is usually preferrable to make unit tests purely
deterministic, in order to avoid missing errors due to particular
choice of array values, several tests make use of randomized arrays
provided by the helper routines :func:`random_spectral_array` and
:func:`random_physical_array`.

The tests in this module do not automatically run on different numbers
of MPI ranks.  The tests all use the default communicator, that is,
they perform the FFTs on all the available MPI ranks.  Testing for
different numbers of MPI ranks must be managed manually by the user by
running these tests with different arguments to :program:`mpirun` (as
described in :mod:`psdns.tests`).  Note that the scaling tests
(:mod:`psdns.tests.test_scaling`) do not cover this, since they test
only run-times, not whether the results are correct.
"""
import unittest

import numpy
from numpy import testing as nptest

from mpi4py import MPI
try:
    from mpi4py_fft import PFFT, newDistArray
    mpi4py_fft_loaded = True
except:
    mpi4py_fft_loaded = False
    
from psdns import *

#: :meta hide-value:
#:
#: A list of domains on which to run individual tests.  Each domain
#: is a 2-tuple which is passed as the arguments to the
#: :class:`~psdns.bases.SpectralGrid` constructor.  The domains sizes
#: are choses to be small enough to keep test times short, and to
#: inlude all the important combinations of truncanted (anti-aliased)
#: and non-truncated, as well as odd and even length, transforms in each
#: direction.
domains = [
    ((9, 9, 9), (9, 9, 9)),
    ((8, 9, 9), (8, 9, 9)),
    ((9, 8, 9), (9, 8, 9)),
    ((8, 8, 9), (8, 8, 9)),
    ((9, 9, 8), (9, 9, 8)),
    ((8, 9, 8), (8, 9, 8)),
    ((9, 8, 8), (9, 8, 8)),
    ((8, 8, 8), (8, 8, 8)),
    ((9, 9, 9), (12, 12, 12)),
    ((8, 9, 9), (12, 12, 12)),
    ((9, 8, 9), (12, 12, 12)),
    ((8, 8, 9), (12, 12, 12)),
    ((9, 9, 8), (12, 12, 12)),
    ((8, 9, 8), (12, 12, 12)),
    ((9, 8, 8), (12, 12, 12)),
    ((8, 8, 8), (12, 12, 12)),
    ((8, 8, 8), (12, 12, 8)),
    ]


def random_spectral_array(grid, shape=()):
    """Return a random :class:`~psdns.bases.SpectralArray`.

    Return a :class:`~psdns.bases.SpectralArray` on the specified
    *grid* with the specified *shape*, filled with random data.

    In order to assure that this spectral array has the approriate
    symmetries, it is obtained by first creating a random physical
    array, and then transforming to spectral space.
    """
    return random_physical_array(grid, shape).to_spectral()


def random_physical_array(grid, shape=()):
    """Return a random :class:`~psdns.bases.PhysicalArray`.

    Return a :class:`~psdns.bases.PhysicalArray` on the specified
    *grid* with the specified *shape*, filled with random data.

    Note, if the specified *grid* is anti-aliased, then this physical
    array will contain higher mode data that will be lost when
    transformed.  In particular, transforming this array to spectral
    space, and then back to physical space, will not recover the
    original array.  For tests that require a physical array that does
    not contain truncated spectral modes, the data can be filtered
    using a transform to spectral and back to physical, i.e.::

        p = random_physical_array(grid).to_spectral().to_physical()
    """
    return PhysicalArray(grid, numpy.random.random(shape+grid.x.shape[1:]))


class TestSingleMode(tests.TestCase):
    """Test transform of a single spectral mode
    """
    def initialize_transform_pair(self, klm, sdims, pdims):
        r"""Return a physical and spectral representation of a single mode

        In order to test transforms of a single mode, this method
        returns a tuple containing a
        :class:`~psdns.bases.PhysicalArray`, and
        :class:`~psdns.bases.SpectralArray`, and a string describing the
        type of mode (see below for descriptions of the mode types).
        The amplitude is always set to one, and the phase is random.
        The test method can then use these fields to test that one
        transforms to the other.  The index of the spectral mode is
        given by the tuple *klm*; *sdims* and *pdims* are the size of the
        array to create.

        As described in :ref:`Real Transforms` the
        :class:`~psdns.bases.SpectralArray` class retains slightly over
        half the Hermitian symmetric modes.  These can be divided into
        different groups, which transform differently.  Note that the
        various cases could be broken out into different tests methods.
        However, this implementation, that iterates over all modes in
        one routine, was chosen to reduce the likelihood that tests of
        certain modes were inadvertantly omitted.

        #. **Interior modes.** For most modes, their is also a Hermitian
           symmetric mode which is not stored.  The physical
           representation is given by equation
           :eq:`3d-transform`.

        #. **Edge modes.**  These are modes for which both of the
           Hermitian symmetric mode are part of the array, which happens
           when :math:`m=0` or :math:`2 m = N_z`
           (eqs. :eq:`edge-zero`-:eq:`edge-nz2`) .  In this case, both
           modes must be initialized consistently.

        #. **Corner modes.**  Hermitian symmetry requires that certain
           modes (see eq. :eq:`corner`)  be purely real, and the
           physical space representation is given by equation
           :eq:`corner-transform`.

        #. **Truncated modes.** When the physical dimensions are larger
           than the spectral dimensions, certain modes are truncated.
           For these modes, the physical-to-spectral transform will
           return zero.  There is no test for the spectral-to-physical
           transform, since these modes are not representable in
           spectral space.

           Also, with ``aliasing_strategy=truncate`` (see :ref:`Keeping
           it real`), which is the setting tested in this test, for
           truncation to an even number of modes, the extra negative
           mode, :math:`-N_x/2` or :math:`-N_y/2`, is zeroed out, and
           therefore is tested identically to the truncated mode.

           .. note::

               This test does not check the result of a
               spectral-to-physical transform of a zeroed out mode set
               to a non-zero value by the user, since this behavior is
               not defined, and should be avoided.
        """
        k, l, m = klm
        grid = SpectralGrid(sdims, pdims)
        s = SpectralArray(grid)
        p = PhysicalArray(grid)
        theta = grid.comm.bcast(numpy.random.rand())
        if ((grid.pdims[0] > grid.sdims[0] and abs(2*k) >= grid.sdims[0]) or
            (grid.pdims[1] > grid.sdims[1] and abs(2*l) >= grid.sdims[1]) or
            2*m > sdims[2]):
            p[...] = 2*numpy.cos(
                k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta
                )
            typ = 'trunc'
        elif ((k == 0 or -2*k == pdims[0]) and
              (l == 0 or -2*l == pdims[1]) and
              (m == 0 or 2*m == pdims[2] == sdims[2])):
            s.set_mode([k, l, m], 1)
            p[...] = numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2])
            typ = 'corner'
        elif m == 0 or 2*m == pdims[2]:
            s.set_mode([k, l, m], numpy.exp(1j*theta))
            s.set_mode([-k, -l, m], numpy.exp(-1j*theta))
            p[...] = 2*numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta)
            typ = 'edge'
        else:
            s.set_mode([k, l, m], numpy.exp(1j*theta))
            p[...] = 2*numpy.cos(k*p.grid.x[0]+l*p.grid.x[1]+m*p.grid.x[2]+theta)
            typ = 'interior'
        return p, s, typ

    def test_single_mode(self):
        r"""Forward and backward transforms of a single mode are correct.

        Test that a single spectral mode transforms correctly, both
        physical-to-spectral and spectral-to-physical.  This test loops
        over all modes supported on the physical space grid, and uses
        the method :meth:`initialize_transform_pair` to generate the
        physical and spectral space representations.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                for k in range(-(pdims[0]//2), (pdims[0]+1)//2):
                    for l in range(-(pdims[1]//2), (pdims[1]+1)//2):
                        for m in range(pdims[2]//2+1):
                            with self.subTest(k=k, l=l, m=m):
                                p, s, typ = self.initialize_transform_pair(
                                    (k, l, m), sdims, pdims
                                    )
                                with self.subTest(type=typ, dir="forward"):
                                    nptest.assert_almost_equal(
                                        numpy.asarray(p.to_spectral()),
                                        numpy.asarray(s)
                                        )
                                if typ == 'trunc':
                                    continue
                                with self.subTest(type=typ, dir="backward"):
                                    nptest.assert_almost_equal(
                                        numpy.asarray(s.to_physical()),
                                        numpy.asarray(p)
                                        )


class TestSymmetries(tests.TestCase):
    r"""Test that spectral transforms have the correct symmetries.

    Hermitian symmetry imposes certain contraints on spectral arrays
    (see :ref:`Three-dimensional transforms`).  This test confirms that,
    for a random array, these symmetries occur.
    """
    def test_z_zero(self):
        r"""Test Hermitian symmetry when z=0"""
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                for k in range(1, sdims[0]//2):
                    for l in range(1, sdims[1]//2):
                        with self.subTest(k=k, l=l):
                            with self.rank_zero(s.grid.comm):
                                self.assertAlmostEqual(
                                    s.get_mode([k, l, 0]),
                                    s.get_mode([-k, -l, 0]).conjugate()
                                    )

    def test_z_max(self):
        r"""Test Hermitian symmetry when z=Nz/2"""
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                if sdims[2] % 2 == 0:
                    s = random_spectral_array(SpectralGrid(sdims, pdims))
                    for k in range(1, sdims[0]//2):
                        for l in range(1, sdims[1]//2):
                            with self.subTest(k=k, l=l):
                                with self.rank_zero(s.grid.comm):
                                    self.assertAlmostEqual(
                                        s.get_mode([k, l, 0]),
                                        s.get_mode([-k, -l, 0]).conjugate()
                                        )

    def test_real_corners(self):
        r"""Test for real values in corners"""
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                krange = [0, -sdims[0]//2] \
                    if sdims[0] == pdims[0] and sdims[0] % 2 == 0 \
                    else [0]
                lrange = [0, -sdims[1]//2] \
                    if sdims[1] == pdims[1] and sdims[1] % 2 == 0 \
                    else [0]
                mrange = [0, sdims[2]//2] \
                    if sdims[2] == pdims[2] and sdims[2] % 2 == 0 \
                    else [0]
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                for k in krange:
                    for l in lrange:
                        for m in mrange:
                            with self.subTest(k=k, l=l, m=m):
                                with self.rank_zero(s.grid.comm):
                                    self.assertAlmostEqual(
                                        s.get_mode([k, l, m]).imag,
                                        0
                                        )


class TestProperties(tests.TestCase):
    """Test that various mathematical properties of the FFT obeyed.
    """
    def test_round_trip1(self):
        """Transforming to spectral and back to physical returns the original value.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                p = random_physical_array(SpectralGrid(sdims, pdims))
                # Filter out unsupported spectral modes
                p = p.to_spectral().to_physical()
                nptest.assert_allclose(
                    numpy.asarray(p.to_spectral().to_physical()),
                    numpy.asarray(p)
                    )

    def test_round_trip2(self):
        """Transforming to physical and back to spectral returns the original value.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                nptest.assert_allclose(
                    numpy.asarray(s.to_physical().to_spectral()),
                    numpy.asarray(s)
                    )

    def test_linear1(self):
        """Physical-to-spectral transforms are linear.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                grid = SpectralGrid(sdims, pdims)
                p1 = random_physical_array(grid)
                p2 = random_physical_array(grid)
                a = grid.comm.bcast(numpy.random.rand(), root=0)
                nptest.assert_allclose(
                    numpy.asarray((p1+a*p2).to_spectral()),
                    numpy.asarray(p1.to_spectral()+a*p2.to_spectral())
                    )

    def test_linear2(self):
        """Spectral-to-physical transforms are linear.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                grid = SpectralGrid(sdims, pdims)
                s1 = random_spectral_array(grid)
                s2 = random_spectral_array(grid)
                a = grid.comm.bcast(numpy.random.rand(), root=0)
                nptest.assert_allclose(
                    numpy.asarray((s1+a*s2).to_physical()),
                    numpy.asarray(s1.to_physical()+a*s2.to_physical())
                    )

    @unittest.skip("Something is funny about norms.")
    def test_norm(self):
        """Spectral norm should match physical space norm.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(SpectralGrid(sdims, pdims))
                with self.rank_zero(s.grid.comm):
                    self.assertAlmostEqual(
                        s.norm(),
                        s.to_physical().norm()
                        )

    @unittest.skip("Something is funny about norms.")
    def test_norm2(self):
        """Physical norm should match spectral space norm.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                p = random_physical_array(SpectralGrid(sdims, pdims))
                # Filter out unsupported spectral modes
                p = p.to_spectral().to_physical()
                with self.rank_zero(p.grid.comm):
                    self.assertAlmostEqual(
                        p.norm(),
                        p.to_spectral().norm()
                        )

    def test_norm3(self):
        """Check magnitude (scaling) of physical norm"""
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                p = random_physical_array(SpectralGrid(sdims, pdims))
                p[...] = numpy.cos(p.grid.x[0])
                with self.rank_zero(p.grid.comm):
                    self.assertAlmostEqual(p.norm(), 0.5)

    def test_vector_to_spectral(self):
        """Vectors transform to spectral elementwise
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                p = random_physical_array(
                    SpectralGrid(sdims, pdims),
                    shape=(3,)
                    )
                s = p.to_spectral()
                with self.subTest("check shape"):
                    self.assertEqual(
                        p.shape[:-3],
                        s.shape[:-3]
                        )
                for i in range(3):
                    with self.subTest(vector_element=i):
                        nptest.assert_almost_equal(
                            numpy.asarray(p[i].to_spectral()),
                            numpy.asarray(s[i])
                            )

    def test_vector_to_physical(self):
        """Vectors transform to physical elementwise
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                s = random_spectral_array(
                    SpectralGrid(sdims, pdims),
                    shape=(3,)
                    )
                p = s.to_physical()
                with self.subTest("check shape"):
                    self.assertEqual(
                        s.shape[:-3],
                        p.shape[:-3]
                        )
                for i in range(3):
                    with self.subTest(vector_element=i):
                        nptest.assert_almost_equal(
                            numpy.asarray(s[i].to_physical()),
                            numpy.asarray(p[i])
                            )


@unittest.skipIf(
    MPI.COMM_WORLD.size != 1,
    "Test may fail if array subsizes don't match (see documentation)"
    )
@unittest.skipIf(
    not mpi4py_fft_loaded,
    "MPI4PyFFT library is not available"
    )
class TestMPI4PyFFT(tests.TestCase):
    """Test that our transforms return the same results as mpi4py-fft.

    Compare the results of PsDNS transforms to those provided by the
    mpi4py-fft library, with ``aliasing_stragegy=mpi4py`` for the
    :class:`~psdns.bases.SpectralGrid`.  Note that, since the two
    codes may decompse domains differently, this test may fail if run
    on multiple MPI ranks.
    """
    def test_mpi4py_fft_forward(self):
        """Physical-to-spectral transforms match mpi4py-fft.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                fft = PFFT(
                    MPI.COMM_WORLD,
                    sdims,
                    padding=numpy.asarray(pdims)/numpy.asarray(sdims),
                    axes=(0, 1, 2),
                    dtype=float,
                    grid=(-1,)
                    )
                p = random_physical_array(
                    SpectralGrid(sdims, pdims, aliasing_strategy='mpi4py')
                    )
                u = newDistArray(fft, False)
                u[...] = p
                u_hat = fft.forward(u, normalize=True)
                s = p.to_spectral()
                numpy.set_printoptions(linewidth=120, precision=2)
                nptest.assert_allclose(
                    u_hat,
                    numpy.asarray(s),
                    )

    def test_mpi4py_fft_backward(self):
        """Spectral-to-physical transforms match mpi4py-fft.
        """
        for sdims, pdims in domains:
            with self.subTest(sdims=sdims, pdims=pdims):
                fft = PFFT(
                    MPI.COMM_WORLD,
                    sdims,
                    padding=numpy.asarray(pdims)/numpy.asarray(sdims),
                    axes=(0, 1, 2),
                    dtype=float,
                    grid=(-1,)
                    )
                s = random_spectral_array(
                    SpectralGrid(sdims, pdims, aliasing_strategy='mpi4py')
                    )
                u_hat = newDistArray(fft, True)
                u_hat[...] = s
                u = fft.backward(u_hat, normalize=False)
                nptest.assert_allclose(
                    u,
                    numpy.asarray(s.to_physical())
                    )
