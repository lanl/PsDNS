"""DNS

A simple psuedo-spectral DNS which recreates one of the cases from
[Mansour1994]_.
"""
from psdns import *
from psdns.equations.navier_stokes import NavierStokes

grid = SpectralGrid(sdims=171, pdims=256)
equations = NavierStokes(Re=1000)

solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=equations,
    ic=equations.rogallo(
        grid,
        params={'q2':3, 'sigma':2, 'kp':25},
        ),
    diagnostics=[
        StandardDiagnostics(
            tdump=0.01, grid=grid, outfile="dns.dat",
            fields=['tke', 'dissipation', 'S'],
            ),
        Spectra(tdump=1.0, grid=grid, outfile="spectra.dat"),
        ],
    )
solver.run()
solver.print_statistics()
solver.json()

