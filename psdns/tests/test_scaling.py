"""Tests to demonstrate the parallel scaling of the code.

The tests in this module are primarily easy ways to check the parallel
scaling of the code when changes are made, or when deployed to new
architectures.  The tests to provide a pass/fail result based on the
measured scaling, however, the primary purpose of the tests is in
producing the scaling plots they output.

Currently, the module tests both strong and weak scaling, and all the
tests are performed for the basic DNS Navier-Stokes solver without
aliasing.
"""
from functools import partial
from math import log
import unittest

import numpy
import scipy.optimize

from mpi4py import MPI

from psdns import *
from psdns.equations.navier_stokes import NavierStokes

try:
    import cpuinfo
    cpu_description = cpuinfo.get_cpu_info()['brand_raw']
except:
    import platform
    cpu_description = platform.platform()
    
class ScalingTest(tests.TestCase):
    """A base class for scaling tests.

    This base class provides the generic functionality for parallel
    scaling tests, and should be subclassed to create specific test
    cases.
    """
    #: A iterable with the number of MPI ranks for each case in the
    #: scaling test.  Subclasses should replace this with a non-empty
    #: list.
    #ncpus = []
    # Note, use a generator expression, not a list, to avoid creating
    # grids until they are needed.
    #grids = ()


    #: The equation class to use for the scaling test.
    #:
    #: The default is a Navier-Stokes at :math:`\mathrm{Re}=100`, but
    #: this can be overridden in sub-classes.
    equations = NavierStokes(Re=100)
    #: A callable which returns the initial conditions.  It should take
    #: a single :class:`SpectralGrid` as an argument, and return a
    #: :class:`SpectralArray` for the initial condition.
    #:
    #: The default is a Taylor-Green vortex.
    ic = equations.taylor_green_vortex
    #: A callable which will be used to instantiate the integrator for
    #: the test cases.
    integrator = partial(
        RungeKutta,
        dt=0.01,
        tfinal=0.1,
        equations=equations,
        diagnostics=[]
        )

    def plot_wall_time(self, ax, runtimes, ncpus, label="", scaling='strong'):
        """Plot the wall clock time.

        Plot the wall clock time as a function of MPI ranks.  The
        reported value of the wall time is actually the average of the
        run time across processors, as measured by the :func:`time.time` 
        funtion.  A line showing perfect speedup is also plotted, for
        reference.
        """
        walltimes = runtimes/ncpus
        ax.plot(ncpus, walltimes, 's', label=label)
        if scaling == 'strong':
            ax.plot(ncpus, walltimes[0]*ncpus[0]/ncpus, 'k--')
        elif scaling == 'weak':
            ax.plot(
                [ 1, ncpus[-1] ],
                [ walltimes[0], walltimes[0] ],
                '--k',
                )
        ax.set_xlabel("Number of MPI ranks")
        ax.set_ylabel("Wall time (s)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    
    def plot_speedup(self, ax, runtimes, ncpus):
        """Plot the parallel speedup.

        Plot the speedup as a function of MPI ranks.  Note that the
        speedup is computed relative to the lowest rank case run, which
        in general may not be serial, since, for large problems, running
        on a single processor may require too much memory, or too long a
        run time.
        """
        ax.plot(
            ncpus,
            runtimes[0]*ncpus/runtimes,
            's',
            )
        ax.plot(
            [ 1, ncpus[-1] ],
            [ 1, ncpus[-1] ],
            '--k',
            )
        ax.set_xlabel("Number of MPI ranks")
        ax.set_ylabel("Speedup")
        ax.set_xscale('log')
        ax.set_yscale('log')

    def plot_efficiency(self, ax, runtimes, ncpus):
        """Plot the parallel efficiency.

        Plot the parallel efficiency as a function of MPI ranks.  Note
        the caveat applies here as for :meth:`plot_speedup`.
        """
        ax.plot(
            ncpus,
            runtimes[0]/runtimes,
            's'
            )
        ax.plot(
            [ 1, ncpus[-1] ],
            [ 1, 1 ],
            '--k'
            )
        ax.set_xlabel("Number of MPI ranks")
        ax.set_ylabel("Parallel Efficiency")
        ax.set_ylim(0, 1.1)

    def run_cases(self, ncpus, grids):
        """Run a series of runs on different numbers of MPI ranks.

        Given a list *ncpus*, and an iterable of *grids*, compute the
        initial condition and run the equation on each grid using the
        number of MPI ranks from the corresponding element in *ncpus*.
        As an individual grid may be quite large, *grids* should use a
        generator to compute each grid as needed.

        The return value is a list of runtimes (total wall clock time
        used by each rank) for the corresponding runs.
        """
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
    "More ranks required to test strong scaling"
    )
class TestStrongScaling(ScalingTest):
    """Test strong scaling for the Navier-Stokes equations."""
    #: A list of cases to run.  Each case is defined by a tuple
    #: consisting of the problem size, and the minimum and maximum
    #: number of processors (specified as powers of two) to run on.
    cases = [
        ( 64, 0, 6 ),
        ( 256, 3, 8 ),
        ( 1024, 8, 10 )
        ]
        
    def test_strong_scaling_tgv(self):
        """Strong scaling for the Navier-Stokes equations"""
        with self.subplots(1, 3, figsize=(9, 3)) as (fig, axs):
            for N, nmin, nmax in self.cases:
                with self.subTest(N=N):
                    ncpus = numpy.array(
                        [ 2**n for n in range(nmin, nmax+1)
                          if 2**n <= MPI.COMM_WORLD.size ]
                        )
                    if ncpus.size == 0:
                        continue
                    grids = (
                        SpectralGrid(
                            N,
                            comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0)
                            )
                        for ncpu in ncpus
                        )
                    runtimes = self.run_cases(ncpus, grids)
                    if MPI.COMM_WORLD.rank == 0:
                        # Check that the scaling is reasonable
                        fit = lambda x, A, n: A*x**-n
                        popt, pcov = scipy.optimize.curve_fit(
                            fit,
                            ncpus,
                            runtimes/ncpus,
                            )
                        self.assertGreaterEqual(
                            popt[1],
                            0.8,
                            )
                # Plotting is outside the subtest, so if the assert fails,
                # the plot is still generated.
                if MPI.COMM_WORLD.rank == 0:
                    # Plot the results
                    self.plot_wall_time(axs[0], runtimes, ncpus, f"$N={N}^3$")
                    self.plot_speedup(axs[1], runtimes, ncpus)
                    self.plot_efficiency(axs[2], runtimes, ncpus)
                    fig.suptitle(cpu_description)
                    fig.tight_layout()
        

@unittest.skipIf(
    MPI.COMM_WORLD.size < 8,
    "More ranks required to test weak scaling"
    )
class TestWeakScaling(ScalingTest):
    """Test weak scaling for the Navier-Stokes equations.

    **A note about weak scaling:** Usually, for a weak scaling test, the
    number of ranks and the problem size are scaled together, so that
    the run time for the problem remains fixed.  The FFT algorithm
    scales as :math:`N \log N`, which make this difficult.  For the
    psuedo-spectral DNS, we assume that the performance is dominated by
    the FFTs.  Since three-dimensional FFTs consist of three sequential
    one-dimensional FFTs, and there are :math:`N^2` one-dimensional
    FFTs per direction, the overall scaling is :math:`3 N^3 \log N`.
    Consequently, we must rescale the run times by :math:`\log N` to
    obtain constant scaling.
    """
    def test_weak_scaling_tgv(self):
        """Weak scaling for the Navier-Stokes equations"""
        ncpus = [
            2**n for n in range(13)
            if 2**n <= MPI.COMM_WORLD.size
            ]
        size = [ int(2**6*ncpu**(1/3)) for ncpu in ncpus ]
        grids = (
            SpectralGrid(
                n,
                comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0)
                )
            for n, ncpu in zip(size, ncpus)
            )
        runtimes = self.run_cases(ncpus, grids)
        if MPI.COMM_WORLD.rank == 0:
            with self.subplots() as (fig, ax):
                self.plot_wall_time(
                    ax,
                    runtimes/numpy.log2(numpy.array(size)),
                    ncpus,
                    scaling='weak'
                    )
                ax.set_ylabel(r"Wall time (s) / $\log_2 N$")
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0)
                ax.get_legend().remove()
                fig.suptitle(cpu_description)
