Pseudo-Spectral Fluid Dynamics in Python
========================================

This package contains some simple tools for writing fluid-dynamics solvers using pseudo-spectral methods.  The basic structure of the package is closely patterned on the spectral DNS code of Mortensen and Langtangen, which can be found `here <https://github.com/spectralDNS/spectralDNS>`_ and is described `here <http://arxiv.org/pdf/1602.03638v1.pdf>`_.  That package requires a substantial software stack, and is built on a generic spectral Galerkin package, which makes it more difficult to tinker with.  This package is designed to be as transparent and easy to modify as possible, for use as a research code.

Dependencies
------------

PsDNS depends on several other Python packages, most importantly
`mpi4py <https://mpi4py.readthedocs.io>`_ for parallelization.  It
also uses `numpy <https://numpy.org>`_ and `scipy <https://scipy.org`_
for numerics, `matplotlib <https://matplotlib.org>`_ for plotting, and
`sphinx <https://www.sphinx-doc.org>`_ for documentation.  The easist
way to install these dependencies is using `conda
<https://anaconda.org>`_.  If you have conda installed, you can just
create a new environment using::

  conda create -n <environment name> mpi4py scipy matplotlib sphinx

Note that this has been tested and works well on OS X.  For
high-performance computer systems, good MPI performance relies on
efficient use of the custom interconnects that are specific to your
system.  Therefor it is important to compile the ``mpi4py`` package to
use your system-specifica tuned MPI libraries, not the off-the-shelf
versions provided by conda.  Please consult the documentation for the
specific HPC system you are using to figure out how to do that.

Examples
--------

Example scripts can be found in the ``examples`` directory.  The
source code for these example are straightforward and can be used as a
template to for other problems.  To run one of these examples without
installing PsDNS::

  PYTHONPATH=. python examples/tgv.py

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

Tests
-----

PsDNS comes with an extensive test suite.  The full test suite can be
run using::

  PYTHONPATH=. python -m unittest -v

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
