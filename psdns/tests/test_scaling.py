from functools import partial
from math import log
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
    
    def plot_wall_time(self, runtimes, ncpus):
        plt.plot(ncpus, plt.array(runtimes)/plt.array(ncpus), 's')
        plt.xlabel("Number of CPUs")
        plt.ylabel("Wall time (s)")
        plt.xscale('log')
        plt.yscale('log')
        
    def plot_speedup(self, runtimes, ncpus):
        plt.plot(
            ncpus,
            runtimes[0]*plt.array(ncpus)/plt.array(runtimes),
            's'
            )
        plt.plot(
            [ 1, ncpus[-1] ],
            [ 1, ncpus[-1] ],
            '--k'
            )
        plt.xlabel("Number of CPUs")
        plt.ylabel("Speedup")

    def plot_efficiency(self, runtimes, ncpus):
        plt.plot(
            ncpus,
            runtimes[0]/plt.array(runtimes),
            's'
            )
        plt.plot(
            [ 1, ncpus[-1] ],
            [ 1, 1 ],
            '--k'
            )
        plt.xlabel("Number of CPUs")
        plt.ylabel("Parallel Efficiency")
        plt.ylim(0, 1.1)

    def log_cpus(self, start, stop):
        maxsize = int(log(MPI.COMM_WORLD.size, 2))
        return numpy.array(
            [ 2**n for n in range(start, min(stop, maxsize)+1) ]
            )
        
    def run_cases(self, ncpus, grids):
        runtimes = []
        for ncpu, grid in zip(ncpus, grids):
            # Only run on `ncpu` processes
            if MPI.COMM_WORLD.rank < ncpu:
                solver = self.integrator(ic=self.ic(grid))
                solver.run()
                runtime = grid.comm.reduce(solver.runtime)
                if MPI.COMM_WORLD.rank == 0:
                    runtimes.append(runtime)
        return numpy.array(runtimes)


@unittest.skipIf(
    MPI.COMM_WORLD.size == 1,
    "More tasks required to test parallel scaling"
    )
class TestStrongScaling(ScalingTest):
    def run_strong_scaling_tgv(self, start, stop, size):
        """Strong scaling for the Navier-Stokes equations"""
        ncpus = self.log_cpus(start, stop)
        grids = (
            SpectralGrid(
                size,
                comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0)
                )
            for ncpu in ncpus
            )
        runtimes = self.run_cases(ncpus, grids)
        if MPI.COMM_WORLD.rank == 0:
            plt.subplot(131)
            self.plot_wall_time(runtimes, ncpus)
            plt.subplot(132)
            self.plot_speedup(runtimes, ncpus)
            plt.subplot(133)
            self.plot_efficiency(runtimes, ncpus)
            plt.tight_layout()
            # Parallel efficiency should be greater than 80%
            self.assertGreaterEqual(
                (runtimes[0]/runtimes).min(),
                0.5
                )
            
    def test_strong_scaling_tgv(self):
        plt.figure(figsize=(9, 3))
        with self.subTest(N=32):
            self.run_strong_scaling_tgv(0, 6, 64)
        with self.subTest(N=256):
            self.run_strong_scaling_tgv(3, 8, 256)
        with self.subTest(N=1024):
            self.run_strong_scaling_tgv(8, 10, 1024)
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


