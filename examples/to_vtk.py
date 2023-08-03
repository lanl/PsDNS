"""A script to convert dump files to VTK for visualization.
"""
import glob

import evtk
import numpy

from psdns import *

L = 2*numpy.pi
N = 6

grid = SpectralGrid(
    sdims=[2**N-1, 2**N-1, 4*2**N-1],
    pdims=[3*2**(N-1), 3*2**(N-1), 12*2**(N-1)],
    box_size=[L, L, 4*L]
    )

def add_vorticity(uhat):
    uhat[3:] = uhat[:3].curl()

solver = Reader(
    dt = 1,
    tfinal = 10,
    diagnostics = [
        VTKDump(
            tdump=1, grid=grid,
            names=['U', 'V', 'W', 'OmegaX', 'OmegaY', 'OmegaZ' ]
            )
        ],
    equations = add_vorticity,
    ic = SpectralArray(grid, (6,)),
    )
solver.run()
