"""RANS

The Taylor-Green vortex, corresponding to the results of Brachet, et
al. (1983), except with a :math:`k-\varepsilon` RANS model.

"""
from psdns.diagnostics import StandardDiagnostics, Spectra, FieldDump
from psdns.integrators import RungeKutta, ImplicitEuler
from psdns.solvers import KEpsilon, TaylorGreenIC


class Equations(KEpsilon, TaylorGreenIC):
    pass


solver = RungeKutta(
    dt=0.01,
    tfinal=4,
    equations=Equations(
        Re=400,
        sdims=2**4-1,
        pdims=3*2**3,
        ),
    diagnostics=[
        StandardDiagnostics(tdump=0.1, outfile="rans.dat"),
        ],
    )
solver.run()
solver.print_statistics()
