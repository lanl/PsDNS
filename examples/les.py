"""Large-eddy simulation

A simple psuedo-spectral LES for the TGV.
"""
import numpy

from psdns.diagnostics import StandardDiagnostics, Spectra
from psdns.integrators import RungeKutta
from psdns.solvers import Smagorinsky, TaylorGreenIC


class Equations(Smagorinsky, TaylorGreenIC):
    pass


solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=Equations(
        Re=400,
        N=2**5,
        padding=1.5,
        ),
    diagnostics=[
        StandardDiagnostics(tdump=0.1, outfile="tgv.dat"),
        Spectra(tdump=1.0, outfile="spectra.dat"),
        ],
    )
solver.run()
solver.print_statistics()
