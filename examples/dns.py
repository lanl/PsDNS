"""DNS

A simple psuedo-spectral DNS for the TGV, corresponding to the results
of Brachet, et al. (1983). 
"""
from psdns import *
from psdns.equations.navier_stokes import NavierStokes


equations = NavierStokes(Re=400)

solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=equations,
    ic=equations.taylor_green_vortex(
        SpectralGrid(sdims=2**6-1, pdims=3*2**5)
        ),
    diagnostics=[
        StandardDiagnostics(tdump=0.1, outfile="tgv.dat"),
        Spectra(tdump=1.0, outfile="spectra.dat"),
        FieldDump(tdump=1.0),
        ],
    )
solver.run()
solver.print_statistics()
