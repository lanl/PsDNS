import unittest


import numpy
from numpy import testing as nptest


from psdns.bases import SpectralArray, PhysicalArray, spectral_grid


class TestTransforms(unittest.TestCase):
    r"""

    For this implementation, given the choice of signs and
    normalizations, the transform of a single Fourier mode,
    :math:`\hat{u}(k_i)` is given by

    .. math::

        u(x_i) = \hat{u}(k_i) \exp 2 \pi i x_i k_i

    Since real transform is Hermetian symmetric, :math:`\hat{u}(k_i) =
    \hat{u}^*(-k_i)`, only half of the spectral data is stored.  Setting
    a single element of the spectral array to a non-zero value activates
    two modes, :math:`k_i` and :math:`-k_i`.  If the coefficient written
    in polar form is :math:`\hat{u} = U e^{i \theta}` then

    .. math::
    
        u(x_i) =   2 U \cos ( 2 \pi x_i k_i + \theta )

    In numpy notation, if::

        uhat[p,q,r] = U*exp(1j*theta)

    then::

        u[k,l,m] = 2*U*numpy.cos(2*numpy.pi*(k*i+l*j+m*k)+theta)

    Edge Cases (literally)
    ----------------------
    
    
    """
    def test_forward_fft(self):
        """Forward transforms with a single mode"""
        N = 9
        p = PhysicalArray((), *spectral_grid(N))
        s = SpectralArray((), p.k, p.x)
        for params in [
                (0, 0, 0, 0),
                (1, 1, 1, 0),
                (1, 1, 1, 0.1),
                (1, 1, 1, -0.1),
                (6, 6, 3, 0.1),
                (7, 1, 1, 0.1),
                (1, 7, 1, 0.1),
                (7, 7, 1, 0.1),
                (1, 1, 4, 0),
                ]:
            with self.subTest(k=params[:3], theta = params[3]):
                k, l, m, theta = params
                p[...] = 2*numpy.cos(k*p.x[0]+l*p.x[1]+m*p.x[2]+theta)
                s[...] = 0
                s[k,l,m] = numpy.exp(1j*theta)
                if k==l==m==0:
                    s[k,l,m] *= 2
                if 2*m==N:
                    s[k,l,m] *= 2
                nptest.assert_almost_equal(
                    numpy.asarray(p.to_spectral()),
                    numpy.asarray(s)
                    )

    def test_linear(self):
        p1 = PhysicalArray((), *spectral_grid(8))
        p2 = PhysicalArray((), *spectral_grid(8))
        p1[...] = numpy.random.random(p1.shape)
        p2[...] = numpy.random.random(p2.shape)
        nptest.assert_almost_equal(
            numpy.asarray(p1+1.7*p2),
            numpy.asarray((p1.to_spectral()+1.7*p2.to_spectral()).to_physical())
            )

    def test_nonlinear(self):
        """Products of modes

        This test uses the fact that

        .. math::

            \cos \alpha \cos \beta = \cos \alpha + \beta + \cos \alpha - \beta
        """
        s1 = SpectralArray((), *spectral_grid(8))
        s2 = SpectralArray((), *spectral_grid(8))
        s1[1,1,1] = 1
        s2[0,0,0] = 2
        s2[2,2,2] = 1  # ???
        nptest.assert_almost_equal(
            numpy.asarray((s1.to_physical()*s1.to_physical()).to_spectral()),
            numpy.asarray(s2)
            )

    def test_dealiasing(self):
        """Products of modes

        This test uses the fact that

        .. math::

            \cos \alpha \cos \beta = \cos \alpha + \beta + \cos \alpha - \beta
        """
        s1 = SpectralArray((), *spectral_grid(8, padding=2)) # This
                                        # doesn't work with padding =
                                        # 1.5.  Why?
        s2 = SpectralArray((), *spectral_grid(8, padding=2))
        s1[4,4,4] = 1
        s2[0,0,0] = 2
        nptest.assert_almost_equal(
            numpy.asarray((s1.to_physical()*s1.to_physical()).to_spectral()),
            numpy.asarray(s2)
            )

    def test_round_trip(self):
        p = PhysicalArray((), *spectral_grid(8))
        p[...] = numpy.random.rand(*p.shape)
        nptest.assert_almost_equal(
            numpy.asarray(p),
            numpy.asarray(p.to_spectral().to_physical())
            )
