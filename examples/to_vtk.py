"""A script to convert dump files to VTK for visualization.
"""
import numpy

from psdns import *

grid = SpectralGrid(
    sdims=[2**5-1, 2**5-1, 2**7-1],
    pdims=[3*2**4, 3*2**4, 3*2**6],
    box_size=[2*numpy.pi, 2*numpy.pi, 8*numpy.pi]
    )

def add_vorticity(uhat):
    uhat[4:] = uhat[:3].curl()

solver = Reader(
    dt = 1,
    tfinal = 10,
    diagnostics = [
        VTKDump(
            tdump=1, grid=grid,
            filename="./phys{time:04g}",
            names=['U', 'V', 'W', 'C', 'OmegaX', 'OmegaY', 'OmegaZ' ]
            )
        ],
    equations = add_vorticity,
    ic = SpectralArray(grid, (7,)),
    )
solver.run()
