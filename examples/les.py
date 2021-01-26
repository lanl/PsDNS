"""Large-eddy simulation

A simple psuedo-spectral LES for the TGV.
"""
from psdns.diagnostics import StandardDiagnostics
from psdns.integrators import RungeKutta
from psdns.solvers import Smagorinsky, TaylorGreenIC


class Equations(Smagorinsky, TaylorGreenIC, StandardDiagnostics):
    pass


solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=Equations(
        Re=400,
        N=2**5,
        padding=1.5,
        tdump=0.1,
        ),
    )
solver.run()
solver.print_statistics()
