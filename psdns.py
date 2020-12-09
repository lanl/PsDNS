"""Psuedo-spectral DNS

A simple psuedo-spectral DNS, designed for easy modification of the
governing equation or diagnostics.  The code uses mpi4py-fft to
parallelize the Fourier transforms.

The basic structure of the code is closely patterned on the spectral
DNS code of Mortensen and Langtangen, which can be found `here
<https://github.com/spectralDNS/spectralDNS>`_ and is described `here
<http://arxiv.org/pdf/1602.03638v1.pdf>`_.
"""
import sys

from psdns.tgv import TGV


solver = TGV(
    dt=0.01,
    tfinal=1.0,
    tdump=1.0,
    Re=400,
    N=2**6,
    )
solver.run()
solver.print_statistics()
