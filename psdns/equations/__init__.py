r"""Psuedo-spectral implementations of some useful equations

In PsDNS, equations are represented as class objects that implement a
:meth:`rhs` method.  The :meth:`rhs` method takes one argument,
*uhat*, which is the solution vector, normally in spectral space
expressed as a :class:`~psdns.bases.SpectralArray`, and it returns the
right-hand side vector.  That is, for the PDE

.. math::

   \frac{\partial}{\partial t} \boldsymbol{U}(t)
   = \boldsymbol{F}[\boldsymbol{U}]

the :meth:`rhs` method takes :math:`\boldsymbol{U}` and returns
:math:`\boldsymbol{F}[\boldsymbol{U}]`.

Since normally only a single equation is needed in a given script,
no equations are imported by default when importing :mod:`psdns`.

Users can also implement their own equations.  There is no need to
subclass from any specific base class, any class that implements a
:meth:`rhs` method can be used as an equation.

The :mod:`~psdns.equations` sub-module also includes some functions
that return initial conditions for certain canonical problems, either
in the form of stand-alone functions, or class methods.
"""
