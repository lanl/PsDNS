"""Single-mode Rayleigh-Taylor

Navier-Stokes equations with Boussinesq buoyancy terms set up for a
single mode Rayleigh-Taylor simulation.  The initial disturbance is a
velocity disturbance that roughly corresponds to the linear
instability eigenfunction.
"""
import numpy
from psdns import *
from psdns.equations.navier_stokes import Boussinesq


grid = SpectralGrid(
    sdims=[2**5-1, 2**5-1, 2**7-1],
    pdims=[3*2**4, 3*2**4, 3*2**6],
    box_size=[2*numpy.pi, 2*numpy.pi, 8*numpy.pi]
    )
equations = Boussinesq(Re=400)

solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=equations,
    ic=equations.perturbed_interface(
        grid,
        0.1*equations.band(grid, 8, 12),
        0.1,
        0.1
        ),
    diagnostics=[
        FieldDump(tdump=1.0, grid=grid, filename="data{:04g}"),
        StandardDiagnostics(tdump=0.1, grid=grid, fields=['tke', 'dissipation', 'divU'], outfile="std.dat"),
        Profiles(tdump=0.1, grid=grid, outfile='profiles.dat'),
        ],
    )
solver.run()
solver.print_statistics()
