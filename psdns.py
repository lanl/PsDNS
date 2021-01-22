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

from psdns.diagnostics import StandardDiagnostics
from psdns.integrators import RungeKutta
from psdns.solvers import HomogeneousDecay, TaylorGreenIC


class Equations(TaylorGreenIC, HomogeneousDecay, StandardDiagnostics):
    pass

solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=Equations(
        Re=400,
        N=2**6,
        padding=1.5,
        tdump=0.1,
        ),
    )
solver.run()
solver.print_statistics()
