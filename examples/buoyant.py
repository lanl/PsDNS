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
    box_size=[4*numpy.pi, 4*numpy.pi, 16*numpy.pi]
    )
equations = Boussinesq(Re=100)

x = grid.x[:2,:,:,0]
solver = RungeKutta(
    dt=0.01,
    tfinal=20.0,
    equations=equations,
    ic=equations.perturbed_interface(
        grid,
        1e-4*equations.band(grid, 1, 4)
        + 0.01*numpy.cos(x[0]) * numpy.cos(x[1]),
        numpy.sqrt(2*numpy.pi),
        0.1,
        ),
    diagnostics=[
        FieldDump(tdump=1.0, grid=grid, filename="data{:04g}"),
        StandardDiagnostics(
            tdump=0.1, grid=grid,
            fields=['tke', 'dissipation', 'cavg', 'divU'],
            outfile="std.dat"
            ),
        Profiles(tdump=0.1, grid=grid, outfile='profiles.dat'),
        DissipationProfiles(tdump=0.1, grid=grid, outfile='dissipation.dat'),
        PressureProfiles(tdump=0.1, grid=grid, outfile='pressure.dat'),
        RapidPressProfiles(tdump=0.1, grid=grid, outfile='press_r.dat'),
        SlowPressProfiles(tdump=0.1, grid=grid, outfile='press_s.dat'),
        BuoyantPressProfiles(tdump=0.1, grid=grid, outfile='press_b.dat'),
        ],
    )
solver.run()
solver.print_statistics()
