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
    sdims=[21, 21, 683],
    pdims=[32, 32, 1024],
    box_size=[4*numpy.pi, 4*numpy.pi, 128*numpy.pi]
    )
grid.checkpoint("data.grid")
equations = Boussinesq(Re=1)

x = grid.x[:2,:,:,0]
solver = RungeKutta(
    dt=0.01,
    tfinal=200.0,
    equations=equations,
    ic=equations.perturbed_interface(
        grid,
        1.0e-3*equations.band(grid, 1, 4)
        + 1.0e-1*numpy.cos(x[0]/2) * numpy.cos(x[1]/2),
        0.1,
        0.1,
        ),
    diagnostics=[
        FieldDump(tdump=1.0, grid=grid, filename="data{:04g}"),
        StandardDiagnostics(
            tdump=0.1, grid=grid,
            fields=['tke', 'dissipation', 'cavg', 'divU'],
            outfile="std.dat"
            ),
        ProfilesConcentration(tdump=0.1, grid=grid, outfile='profiles.dat'),
        DissipationProfiles(tdump=0.1, grid=grid, outfile='dissipation.dat'),
        PressureProfiles(tdump=0.1, grid=grid, outfile='pressure.dat'),
        RapidPressProfiles(tdump=0.1, grid=grid, outfile='press_r.dat'),
        SlowPressProfiles(tdump=0.1, grid=grid, outfile='press_s.dat'),
        BuoyantPressProfiles(tdump=0.1, grid=grid, outfile='press_b.dat'),
        ],
    )
solver.run()
solver.print_statistics()
