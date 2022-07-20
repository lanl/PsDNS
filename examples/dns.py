"""DNS

A simple psuedo-spectral DNS for the TGV, corresponding to the results
of Brachet, et al. (1983).
"""
from psdns import *
import cProfile, pstats, io
from psdns.equations.navier_stokes import NavierStokes
with cProfile.Profile() as pr:
  idx = 7
  grid = SpectralGrid(sdims=128)
  equations = NavierStokes(Re=400)

  solver = RungeKutta(
    dt=0.01,
    tfinal=0.1,
    equations=equations,
    ic=equations.taylor_green_vortex(
        grid
        ),
    diagnostics=[
        #StandardDiagnostics(tdump=0.1, grid=grid, outfile="tgv.dat"),
        #Spectra(tdump=1.0, grid=grid, outfile="spectra.dat"),
        #FieldDump(tdump=1.0, grid=grid),
        ],
    )
  solver.run()
  solver.print_statistics()

pr.print_stats()
#pr.dump_stats("profile")

