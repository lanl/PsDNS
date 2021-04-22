"""DNS

A simple psuedo-spectral DNS for the TGV, corresponding to the results
of Brachet, et al. (1983). 
"""
import pickle

from psdns.diagnostics import StandardDiagnostics, Spectra, FieldDump
from psdns.integrators import RungeKutta
from psdns.solvers import NavierStokes, TaylorGreenIC


class Equations(NavierStokes, TaylorGreenIC):
    pass


solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=Equations(
        Re=400,
        N=2**6,
        padding=1.5,
        ),
    diagnostics=[
        StandardDiagnostics(tdump=0.1, outfile="tgv.dat"),
        Spectra(tdump=1.0, outfile="spectra.dat"),
        FieldDump(tdump=1.0),
        ],
    )
solver.run()
solver.print_statistics()
