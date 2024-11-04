"""DNS of a shear-layer
"""
import numpy

from psdns import *
from psdns.equations.navier_stokes import NavierStokes


L = 4*numpy.pi
N = 6

grid = SpectralGrid(
    sdims=[2**N-1, 2**N-1, 4*2**N-1],
    pdims=[3*2**(N-1), 3*2**(N-1), 12*2**(N-1)],
    box_size=[L, L, 4*L]
    ).checkpoint("data.grid")
equations = NavierStokes(Re=500)

solver = RungeKutta(
    dt=0.05,
    tfinal=50.0,
    equations=equations,
    ic=equations.shear(grid, 0.5),
    diagnostics=[
        FieldDump(tdump=1.0, grid=grid, filename="data{:04g}"),
        StandardDiagnostics(tdump=0.1, grid=grid, fields=['divU'], outfile="std.dat"),
        Profiles(tdump=1.0, grid=grid, outfile="profiles.dat"),
        ],
    )
solver.run()
solver.print_statistics()
