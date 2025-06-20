"""A script to convert dump files to VTK for visualization.
"""
import numpy as np  # It's common practice to alias numpy as np.

from psdns import *

myfolder = "~/Desktop/LANL/PsDNS/caso_buoyant/data/"
griddata = myfolder + "data.grid"
print(griddata)
grid = SpectralGrid.read_checkpoint(griddata)  

def add_vorticity(uhat):
    uhat[4:] = uhat[:3].curl()  # Ensure that `curl` is a valid method for `uhat[:3]`.

solver = Reader(
    dt=1,
    tfinal=5,
    diagnostics=[
        VTKDump(
            tdump=0.5, 
            grid=grid,
            filename=myfolder + "phys{time:04g}",
            names=['U', 'V', 'W', 'C', 'OmegaX', 'OmegaY', 'OmegaZ']
        )
    ],
    equations=add_vorticity, 
    ic=SpectralArray(grid, (7,)),  
    filename=myfolder + "data{:04g}",  
)

solver.run()
