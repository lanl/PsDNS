import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy
import scipy.optimize

from mpi4py import MPI

from psdns import *
from psdns.equations.navier_stokes import NavierStokes


class TestStrongScaling(unittest.TestCase):
    def test_strong_scaling_tgv(self):
        """Strong scaling for the Navier-Stokes equations"""
        for ncpu in [ 1, 2, 4 ]:
            color = MPI.COMM_WORLD.rank//ncpu
            comm = MPI.COMM_WORLD.Split(color, 0)
            if color == 0:
                grid = SpectralGrid(sdims=2**5-1, pdims=3*2**4, comm=comm)
                equations = NavierStokes(Re=100)
                solver = RungeKutta(
                    dt=0.01,
                    tfinal=0.1,
                    equations=equations,
                    ic=equations.taylor_green_vortex(
                        grid
                        ),
                    diagnostics=[]
                    )
                solver.run()
                runtime = comm.reduce(solver.runtime)
                if comm.rank == 0:
                    plt.plot(ncpu, runtime, 'sk')
                    print(ncpu, runtime)
        if MPI.COMM_WORLD.rank == 0:
            plt.xlabel("Number of CPUs")
            plt.ylabel("Total CPU Time")
            plt.savefig("strong.pdf")

