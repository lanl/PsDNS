.. psdns documentation master file, created by
   sphinx-quickstart on Wed Dec 16 22:02:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pseudospectral Direct Numerical Simulation (PsDNS)
==================================================

PsDNS is a Python package for simulating partial differential
equations using psuedo-spectral methods.  The primary purpose is
direct numerical simulation of turbulence (hence the name of the
package).  The implementation is similar in spirit to the `spectralDNS
<https://github.com/spectralDNS/spectralDNS>`_ package of Mortensen
and Darian [Mortensen2016]_, however, it aims for ease of use for
turbulence researchers.  To that end, PsDNS is designed to make adding
diagnostics and new equations sets as quick and intuitive as possibe.


.. toctree::
   :maxdepth: 4

   usage
   fft 
   Reference Guide <psdns>
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`bib`
