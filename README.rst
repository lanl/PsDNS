Pseudo-Spectral Fluid Dynamics in Python
========================================

This package contains some simple tools for writing fluid-dynamics solvers using pseudo-spectral methods.  The basic structure of the package is closely patterned on the spectral DNS code of Mortensen and Langtangen, which can be found `here <https://github.com/spectralDNS/spectralDNS>`_ and is described `here <http://arxiv.org/pdf/1602.03638v1.pdf>`_.  That package requires a substantial software stack, and is built on a generic spectral Galerkin package, which makes it more difficult to tinker with.  This package is designed to be a transparent and easy to modify as possible, for use as a research code.

Documentation
-------------

PsDNS has a `Sphinx <https://www.sphinx-doc.org>`_ manual, including
detailed documentation of the API.  In order to build the manual,
Sphinx needs access to the internet (to link to dependency packages)
and to be able to load the ``psdns`` source code (to access Python
docstrings).

To build the documentation from a fresh checkout of the code, first
make sure your Python environment is set to one that includes all the
dependencies for PsDNS (i.e., one from which you can successully run
the examples) and then type::
  
  cd doc
  PYTHONPATH=../psdns make html

for HTML documentation, or for PDF::

  PYTHONPATH=../psdns make latexpdf

Examples
--------

Example scripts can be found in the ``examples`` directory.

License
-------

This code was approved for open source release (C21086) under the
BSD-3 license (see :file:`LICENSE.txt`)

©2021. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others to
do so.

