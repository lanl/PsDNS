"""DNS of a shear-layer
"""
import numpy

from psdns import *
from psdns.equations.navier_stokes import NavierStokes


L = 2*numpy.pi
N = 6

def shear_ic(grid):
    u = PhysicalArray(grid, (3,))
    u[0] = numpy.where((grid.x[2]>L) & (grid.x[2]<=3*L), -0.5, 0.5)
    for A, a, b, c in [ ( 1e-4, 1, 4, 0 ), ( 1e-4, 1, 4, 1 ) ]:
        u[0] = u[0] - 2*a/b*A*(grid.x[2]-3*L)*numpy.exp(-a*(grid.x[2]-3*L)**2) \
          *numpy.cos(b*grid.x[0]) \
          *numpy.cos(c*grid.x[1])
        u[2] = u[2] + A*numpy.exp(-a*(grid.x[2]-3*L)**2) \
          *numpy.sin(b*grid.x[0]) \
          *numpy.cos(c*grid.x[1])
    s = u.to_spectral()
    s._data = numpy.ascontiguousarray(s._data)
    return s


grid = SpectralGrid(
    sdims=[2**N-1, 2**N-1, 4*2**N-1],
    pdims=[3*2**(N-1), 3*2**(N-1), 12*2**(N-1)],
    box_size=[L, L, 4*L]
    )
equations = NavierStokes(Re=1000)

solver = RungeKutta(
    dt=0.05,
    tfinal=100.0,
    equations=equations,
    ic=shear_ic(grid),
    diagnostics=[
        FieldDump(tdump=1.0, grid=grid, filename="data{:04g}"),
        StandardDiagnostics(tdump=0.1, grid=grid, fields=['divU'], outfile="std.dat"),
        Profiles(tdump=1.0, grid=grid, outfile="profiles.dat")
        ],
    )
solver.run()
solver.print_statistics()
