from functools import partial
import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy
import scipy.optimize

from mpi4py import MPI

from psdns import *
from psdns.equations.navier_stokes import NavierStokes


class ScalingTest(unittest.TestCase):
    ncpus = []
    # Note, use a generator expression, not a list, to avoid creating
    # grids until they are needed.
    grids = ()
    #: A callable which takes a single :class:`SpectralGrid` as an
    #: argument, and returns a :class:`SpectralArray` for the initial
    #: condition.
    equations = NavierStokes(Re=100)
    ic = equations.taylor_green_vortex
    integrator = partial(
        RungeKutta,
        dt=0.01,
        tfinal=0.1,
        equations=equations,
        diagnostics=[]
        )
    
    def plot_wall_time(self, runtimes):
        plt.plot(self.ncpus, plt.array(runtimes)/plt.array(self.ncpus), 'sk')
        plt.xlabel("Number of CPUs")
        plt.ylabel("Wall time (s)")
        plt.xscale('log')
        plt.yscale('log')
        
    def plot_speedup(self, runtimes):
        plt.plot(
            self.ncpus,
            runtimes[0]*plt.array(self.ncpus)/plt.array(runtimes),
            'sk'
            )
        plt.plot(
            [ 1, self.ncpus[-1] ],
            [ 1, self.ncpus[-1] ],
            '--k'
            )
        plt.xlabel("Number of CPUs")
        plt.ylabel("Speedup")

    def plot_efficiency(self, runtimes):
        plt.plot(
            self.ncpus,
            runtimes[0]/plt.array(runtimes),
            'sk'
            )
        plt.plot(
            [ 1, self.ncpus[-1] ],
            [ 1, 1 ],
            '--k'
            )
        plt.xlabel("Number of CPUs")
        plt.ylabel("Parallel Efficiency")
        plt.ylim(0, 1.1)

    def run_cases(self):
        runtimes = []
        for ncpu, grid in zip(self.ncpus, self.grids):
            # Only run on `ncpu` processes
            if MPI.COMM_WORLD.rank < ncpu:
                solver = self.integrator(ic=self.ic(grid))
                solver.run()
                runtime = grid.comm.reduce(solver.runtime)
                if MPI.COMM_WORLD.rank == 0:
                    runtimes.append(runtime)
        return runtimes


class TestStrongScaling(ScalingTest):
    ncpus = [ 1, 2, 3, 4 ]
    grids = (
        SpectralGrid(32, comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0))
        for ncpu in ncpus
        )

    def test_strong_scaling_tgv(self):
        """Strong scaling for the Navier-Stokes equations"""
        runtimes = self.run_cases()
        if MPI.COMM_WORLD.rank == 0:
            plt.figure(figsize=(9, 3))
            plt.subplot(131)
            self.plot_wall_time(runtimes)
            plt.subplot(132)
            self.plot_speedup(runtimes)
            plt.subplot(133)
            self.plot_efficiency(runtimes)
            plt.tight_layout()
            plt.savefig("strong.pdf")


class TestWeakScaling(ScalingTest):
    ncpus = [ 1, 8, 64 ]
    grids = (
        SpectralGrid(n, comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0))
        for n, ncpu in zip([ 32, 64, 128 ], ncpus)
        )

    def test_weak_scaling_tgv(self):
        """Strong scaling for the Navier-Stokes equations"""
        runtimes = self.run_cases()
        if MPI.COMM_WORLD.rank == 0:
            plt.figure(figsize=(3, 3))
            self.plot_wall_time(runtimes)
            plt.tight_layout()
            plt.savefig("weak.pdf")


