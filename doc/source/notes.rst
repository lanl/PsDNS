Implementation Notes
====================

Anti-aliasing can be accomplished by a simple truncation of the higher modes when performing a physical-to-spectral transform, and zero-padding before the corresponding spectral-to-physical transform.  For the case where either of the first two axes are truncated to an even number of modes, this creates two problems.  The first is the introduction of a directional bias, and the second is a violation of Hermetian symmetry.

For illustration, consider truncating the first axis to length ``N``, where ``N`` is even.  In this case, the mode ``N//2`` is truncated, whereas the mode ``-(N//2)`` is retained.  More generally, this means that given a pair of modes

.. math::

   \hat{u}[k,l,m] \cos ( k x + l y + m z ) + \hat{u}[-k,l,m] \cos ( -k x + l y + m z )

where :math:`k = N/2`, only the second mode will be retained.  It is self-evident that this introduces a directional bias, preferring modes that face "left" but not "right."

Furthermore, consider a mode for which :math:`k = N/2` and :math:`m = 0`, an "edge" mode.  In this case, in the untruncated transform there are two Hermitian symmetric modes,

.. math::

   \hat{u}[-N/2, -l, 0] = \hat{u}^*[N/2, l, 0]
   
of which only the first is retained.  If we simply zero-pad the :math:`\hat{u}` array, the resulting array is not Hermitian symmetric.  Passing this to an inverse real-to-complex FFT routine will produce implemtation dependent results.

There are several ways we might address this issue.  The simplest is to set the problematic modes to zero, i.e., set

.. math::

   \hat{u}[k, l, m] = 0, 2 k = N_x, 2 l = N_y 

This is equivalent to, and more efficiently implemented by, restricting :math:`N_x, N_y` to odd numbers.

mpi4py-fft Compatability
------------------------

The `mpi4py-fft <https://mpi4py-fft.readthedocs.io>`_ package uses a different approach.  On truncation, if ``N`` is even, it sets

.. math::

   \hat{u}[-N/2, l, 0] = \hat{u}[-N/2, l, 0] + \hat{u}[N/2, l, 0]

and similarly for the second axis.  This same scaling is implemented here, as an available option, primarily for testing purposes.  It is activated by setting :data:`psdns.bases.use_mpi4py_fft_scaling` to ``True``.  The test case :class:`~psdns.tests.test_fft.TestMPI4PyFFT` is provided to confirm that both codes produce the same results.
