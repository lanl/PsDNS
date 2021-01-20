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

from psdns.solvers import HomogeneousDecay, TaylorGreenIC
from psdns.integrators import RungeKutta


class Solver(RungeKutta, TaylorGreenIC, HomogeneousDecay):
    pass

solver = Solver(
    dt=0.01,
    tfinal=10.0,
    tdump=0.1,
    Re=400,
    N=2**6,
    )
solver.run()
solver.print_statistics()
