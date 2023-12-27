"""A script to convert dump files to VTK for visualization.
"""
import numpy

from psdns import *

grid = SpectralGrid.read_checkpoint("data.grid")

def add_vorticity(uhat):
    uhat[4:] = uhat[:3].curl()

solver = Reader(
    dt = 1,
    tfinal = 100,
    diagnostics = [
        VTKDump(
            tdump=1, grid=grid,
            filename="./phys{time:04g}",
            names=['U', 'V', 'W', 'C', 'OmegaX', 'OmegaY', 'OmegaZ' ]
            )
        ],
    equations = add_vorticity,
    ic = SpectralArray(grid, (7,)),
    filename = "data{:04g}",
    )
solver.run()
