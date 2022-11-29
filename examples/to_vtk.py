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

for fn in sorted(glob.glob("data????")):
    s = SpectralArray(grid, (3,))
    s.read_checkpoint(fn)
    p = s.to_physical()
    w = s.curl().to_physical()
    evtk.hl.gridToVTK(
        fn,
        grid.x[0], grid.x[1], grid.x[2],
        pointData = {
            "U": (numpy.asarray(p[0]), numpy.asarray(p[1]), numpy.asarray(p[2])),
            "Vorticity": (numpy.asarray(w[0]), numpy.asarray(w[1]), numpy.asarray(w[2])),
        }
    )
