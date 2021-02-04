import unittest


import numpy
from numpy import testing as nptest


from psdns.bases import SpectralArray, PhysicalArray, spectral_grid


class TestTransforms(unittest.TestCase):
    def test_forward_fft(self):
        p = PhysicalArray((), *spectral_grid(8))
        p[...] = numpy.cos(p.x[0])*numpy.sin(2*p.x[2])
        s = SpectralArray((), p.k, p.x)
        s[...] = 0
        s[1,0,2] = -0.25j
        s[-1,0,2] = -0.25j
        nptest.assert_almost_equal(p.to_spectral(), s)

    def test_round_trip(self):
        p = PhysicalArray((), *spectral_grid(8))
        p[...] = numpy.random.rand(*p.shape)
        nptest.assert_almost_equal(
            p,
            p.to_spectral().to_physical(),
            )
