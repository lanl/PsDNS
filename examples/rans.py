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
        N=2**4-1,
        padding=1.6,
        ),
    diagnostics=[
        StandardDiagnostics(tdump=0.1, outfile="rans.dat"),
        ],
    )
print(solver.equations.uhat.k.shape)
print(solver.equations.uhat.x.shape)
solver.run()
solver.print_statistics()
