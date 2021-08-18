Fast-Fourier Transforms
=======================

Details on how to use the FFT classes can be found in the reference
for the :mod:`~psdns.bases` module.  This section reviews the theory
and some of the implementation considerations for the FFTs.

Definitions
-----------

Consider a continuous, periodic function in the domain :math:`x \in
[0,L]`, represented on a discrete set of N mesh points,
:math:`x_{n}=nL/N`. Using Fourier analysis, we can consider this
function to be a combination of sine and cosine (Fourier) modes. The
complex, discrete Fourier transform (DFT) is defined by

.. math::
   :label:
      
    \hat{u}_{k}
    = \frac{1}{N} \sum_{n=0}^{N-1} u_n
      \exp \left( - \frac{2\pi}{L} i x_{n} k \right)

and

.. math::
    :label:
       
    u_{n}
    = \sum_{k=0}^{N-1} \hat{u}_k
      \exp \left( \frac{2\pi}{L} i x_n k \right)

Here :math:`u_n = u(x_n)` are the discrete spatial values on the
grid, and :math:`\hat{u}_k` are the discrete Fourier modes. The
normalization is chosen so that a spectral component with unit
amplitude has amplitude one in physical space.

Due to the finite mesh spacing, there is a maximum frequency which can
be represented on the mesh. Higher frequency modes are aliased, that
is, they take on identical values at the grid points to lower
frequency modes, which they are therefore indistinguishable
from. Specifically,

.. math::
    :label: aliasing

    \hat{u}_{k+mN}
    & = \frac{1}{N} \sum_{n=0}^{N-1} u_n
        \exp \left( - \frac{2\pi}{L} i x_n \left( k + m N \right) \right) \\
    & = \frac{1}{N} \sum_{n=0}^{N-1} u_n
        \exp \left( - \frac{2\pi}{L} i x_n k \right)
        \exp \left( - \frac{2\pi}{L} i x_n m N \right) \\
    & = \hat{u}_k

for any integer :math:`m`, since :math:`\exp \left( - \frac{2\pi}{L} i
x_n m N \right) = \exp \left( - 2 \pi i n m \right) = 1`.  Therefor,
it is common to interpret the modes from :math:`k=0, \ldots, N` as
modes from :math:`k = -N/2, \dots, N/2`. Because of the vagaries of
integer arithmetic, and the way most FFT libraries lay out these modes
in memory, the modes are stored in the order

.. math::
    :label:

    k =
    \begin{cases}
    0, 1, 2, \ldots, (N-1)/2, - (N-1)/2, \ldots, -1
    & N\text{ is odd} \\
    0, 1, 2, \ldots, N/2-1, -N/2, \ldots, -1
    & N\text{ is even}
    \end{cases}

Note that, when :math:`N` is even, the :math:`N/2` mode can
equivalently be interpreted as the :math:`-N/2` mode, since both are
equal.

In three dimensions we have

.. math::
    :label:

    \hat{u}[k,l,m]
    = \frac{1}{N}
      \sum_{p=0}^{N_z-1} \sum_{q=0}^{N_y-1} \sum_{r=0}^{N_z-1}
        u[p,q,r]
        \exp \left(
          - \frac{2\pi}{L} i \left( x_p k + y_q l + z_r m \right)
        \right)

where :math:`N = N_x N_y N_z`, and

.. math::
    :label:

    u[p,q,r]
    = \sum_{k=0}^{N_x-1} \sum_{l=0}^{N_y-1} \sum_{m=0}^{N_z-1}
        \hat{u}[k,l,m]
        \exp \left(
          \frac{2\pi}{L} i \left( x_p k + y_q l + z_r m \right)
        \right)

Real Transforms
---------------

One-dimesional transforms
^^^^^^^^^^^^^^^^^^^^^^^^^

By inspection, using the fact that :math:`\left( e^{i\theta} \right)^*
= e^{-i \theta}`, the modes :math:`\hat{u}[k,l,m]` are Hermitian
symmetric if :math:`u[p,q,r]` are real, that is,
:math:`\hat{u}[-k,-l,-m] = \hat{u}^*[k,l,m]` This means that we only
need to retain half of the modes in order to have a complete
description. First consider a one-dimensional transform. Hermitian
symmetry implies that the zero mode, :math:`\hat{u}_0 = \hat{u}_0^*`
is real, and if :math:`\hat{u}_0` is the only non-zero mode, then
:math:`u_n = \hat{u}_0` for all :math:`n`.  Each positive mode implies
two modes, since it has a corresponding Hermitian symmetric mode.  So
if the only non-zero mode is :math:`\hat{u}_k`, then then
corresponding physical space representation is

.. math::
    :label: 1d-transform

    u_n
    & = \hat{u}_k \exp \left( \frac{2\pi}{L} i x_n k \right)
      + \hat{u}_k^* \exp \left( - \frac{2\pi}{L} i x_n k \right) \\
    & = 2 \left| \hat{u}_k \right |
        \cos \left( \frac{2\pi k}{L} x_n +\operatorname{Arg} u_k \right)

If :math:`N` is even, then the highest mode is :math:`N/2` and since
:math:`\hat{u}_{N/2} = \hat{u}_{-N/2} = \hat{u}_{N/2}^*` it is also
real.  However, it does not imply two modes, since the :math:`-N/2`
mode would not be present in the full, complex-to-complex
transform. If it was the only non-zero mode, then

.. math::

    u_n
    & = \hat{u}_{N/2} \exp \left( \frac{N\pi}{L} i x_n \right) \\
    & = \hat{u}_{N/2} \cos \left( \frac{N\pi}{L} x_n \right)

since :math:`\sin \left( \frac{N\pi}{L} x_n \right) = \sin \left( \pi
n \right) = 0`.  In other words, the :math:`N/2` mode only includes a
cosine component, since the sine is identically zero on the grid
points.

Three-dimensional transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In higher dimensions, in order to keep a contiguous array, we actually
retain all the non-negative modes in :math:`z`, :math:`m \ge 0`, which
is slightly more than half the modes. This description implies some
constraints on the :math:`\hat{u}` array.  Equation :eq:`1d-transform`
generalizes to

.. math::
    :label: 3d-transform

    u[p,q,r]
    = 2 \left| \hat{u}[k,l,m] \right |
        \cos \left( \frac{2\pi}{L} \left( x_p k + y_q l + z_r m \right)
                  + \operatorname{Arg} u_k \right)

However, for certain modes there are specific additional constraints.
Hermitian symmetry requires

.. math::
    :label: edge-zero

    \hat{u}[-k,-l,0] = \hat{u}^*[k,l,0]

which also implies :math:`\hat{u}[0,0,0]` is real. If :math:`N_z` is
even, the fact that :math:`\hat{u}[k,l,N/2] = \hat{u}[k,l,-N/2]` leads
to the additional constraint

.. math::
    :label: edge-nz2

    \hat{u}[-k,-l,-N_z/2] = \hat{u}^*[k,l,-N_z/2]

In these cases, for :eq:`3d-transform` to apply, both Hermitian
symmetric modes must be set consistently.

Furthermore, the three-dimension extension of equation :eq:`aliasing`,
along with the Hermitian symmetry, imply that certain “corner” values
are real.  Specifically

.. math::
    :label: corner

    \hat{u}[
      0\text{ or }-N_x/2,
      0\text{ or }-N_y/2,
      0\text{ or }-N_z/2]
    \text{ is real}

where the :math:`N_i/2` condition only applies when :math:`N_i` is
even.  For corner modes, since there is only a single mode implied,
the :eq:`3d-transform` reduces to

.. math::
    :label: corner-transform

    u[p,q,r]
    = \hat{u}[k,l,m]
      \cos \left( \frac{2\pi}{L} \left( x_p k + y_q l + z_r m \right) \right)


Dealiasing
----------

The nice thing about a spectral representation is that derivatives can
be taken exactly, simply, and locally (without using neighbor points),

.. math::

    \widehat{\frac{\partial f}{\partial x_i}}
    = - i k_i \hat{f}(\boldsymbol{k})

The disadvantage is that non-linear operations, such as products, are
now convolutions, which are both expensive in operation count, and
require global data.  The psuedo-spectal approach works in spectral
space except when computing non-linear terms, which are done in
physical space.  This requires fast Fourier transforms (FFT) for
constructing the non-linear terms.

However, non-linear operations generate higher-order modes.  If we
attempt to transform a function that contains a higher mode component
than is supported on the grid, the result is aliasing.  This is a
consequence of the result above (here stated in three-dimensions),
that :math:`\hat{u}[k + p N_x, l + q N_y, m + r N_z] = \hat{u}[k,l,m]`
for any integer :math:`p, q, r`.  In a typical application we are
concerned with the non-linear interaction between two modes :math:`k`
and :math:`l` (working this example in one-dimension, for simplicity).
These will generate a new mode :math:`k+l`. If :math:`k + l > N` is
not resolved in the spectrum, the results will alias to the :math:`k +
l - N` mode.  In order to avoid this, typically some form of
dealiasing is employed. Here we consider dealiasing by padding.

The idea is that we pad out our transform to include additional modes
that are set equal to zero. The non-linear operation is performed in
physical space, and then the back transform is truncated so any modes
containing aliased energy are removed. If we have :math:`N` modes
(including positive and negative modes), the highest mode is
:math:`N/2` for :math:`N` even, and :math:`(N-1)/2` for :math:`N` odd.

For operations involving the product of two terms, dealiasing
typically uses the 3/2-rule, that is, the number of physical grid
points, :math:`M`, is 3/2 the number of retained spectral modes,
:math:`N`.

For :math:`N` even, the largest spectral mode is :math:`N/2`, so the
largest non-linear mode is :math:`N`, which aliases to
:math:`N-M<0`. If we want this to be truncated, then

.. math::

    N-M & < -N/2 + 1 \\
    M & > 3N/2 - 1 \\
    M & \geq 3N/2

or

.. math::

    2M/3 \geq N

For :math:`N` odd, the largest spectral mode is :math:`(N-1)/2`, so
the largest non-linear mode is :math:`N-1`, which aliases to
:math:`N-1-M < 0`.  If we want this to be truncated, then

.. math::

    N-1-M & < - (N-1)/2 \\
    M & > 3 (N-1)/2

or

.. math::

    2M/3+1 > N

Note that for all of these constraints, although :math:`N` is the
number of spectral modes retained for calculation, the actual
transform length is :math:`M`.

FFT routines typically work only, or, at least most efficiently, with
lengths that are the product of small primes.  The most efficient is
generally when :math:`M = 2^n`.  However, the most efficient choice of
:math:`M`, must be a multiple of 3.  Although historically many codes
used FFT sizes which are powers of 2, with modern FFT libraries, there
is no good reason to do this.

Keeping It Real
---------------

Anti-aliasing can be accomplished by a simple truncation of the higher
modes when performing a physical-to-spectral transform, and
zero-padding before the corresponding spectral-to-physical transform.
For the case where either of the first two axes are truncated to an
even number of modes, this creates two problems.  The first is the
introduction of a directional bias, and the second is a violation of
Hermetian symmetry.

For illustration, consider truncating the first axis to length ``N``,
where ``N`` is even.  In this case, the mode ``N//2`` is truncated,
whereas the mode ``-(N//2)`` is retained.  More generally, this means
that given a pair of modes

.. math::

   \hat{u}[k,l,m] \cos ( k x + l y + m z )
   + \hat{u}[-k,l,m] \cos ( -k x + l y + m z )

where :math:`k = N/2`, only the second mode will be retained.  It is
self-evident that this introduces a directional bias, preferring modes
that face "left" but not "right."

Furthermore, consider a mode for which :math:`k = N/2` and :math:`m =
0`, an "edge" mode.  In this case, in the untruncated transform there
are two Hermitian symmetric modes,

.. math::

   \hat{u}[-N/2, -l, 0] = \hat{u}^*[N/2, l, 0]

of which only the first is retained.  If we simply zero-pad the
:math:`\hat{u}` array, the resulting array is not Hermitian symmetric.
Passing this to an inverse real-to-complex FFT routine will produce
implemtation dependent results.

There are several ways we might address this issue.  Which one is used
is controlled by setting the parameter ``aliasing_strategy`` when
creating a :class:`~psdns.bases.SpectralGrid`.  The simplest approach
is to set the problematic modes to zero, i.e., set

.. math::

   \hat{u}[k, l, m] = 0, 2 k = N_x, 2 l = N_y

Setting ``aliasing_strategy=truncate`` will do this, however, it is
equivalent to, and more efficiently implemented by, restricting
:math:`N_x, N_y` to odd numbers.  Currently PsDNS issues a warning if
users attempt to create a :class:`~psdns.bases.SpectralGrid` instance
with even spectral size in the first two dimensions.

The `mpi4py-fft <https://mpi4py-fft.readthedocs.io>`_ package uses a
different approach.  On truncation, if ``N`` is even, it sets

.. math::

   \hat{u}[-N/2, l, 0] = \hat{u}[-N/2, l, 0] + \hat{u}[N/2, l, 0]

and similarly for the second axis.  This same scaling is implemented
here, as an available option, primarily for testing purposes.  It is
activated by passing ``aliasing_strategy=mpi4py``.  The test case
:class:`~psdns.tests.test_fft.TestMPI4PyFFT` is provided to confirm
that both codes produce the same results with this setting.  Note that
this approach is effectively a filter, which means that transforming
from spectral to physical and back to spectral will not return the
original array with this setting.
