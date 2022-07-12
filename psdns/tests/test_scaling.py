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

import  numpy
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
    cases.  The :attr:`integrator`, :attr:`equations`, and :attr:`ic`
    class attributes determine the integrator, equation set, and
    initial conditions to use for the test, and can be over-ridden in
    sub-classes.
    """
    #: The equation class to use for the scaling test.
    #: The default is a Navier-Stokes at :math:`\mathrm{Re}=100`, but
    #: this can be overridden in sub-classes.
    equations = NavierStokes(Re=100)
    #: A callable which returns the initial conditions.  It should take
    #: a single :class:`~psdns.bases.SpectralGrid` as an argument, and
    #: return a :class:`~psdns.bases.SpectralArray` for the initial
    #: condition.
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

    def plot_wall_time(
            self, ax, total_walltimes, ncpus, label="",
            scaling='strong'):
        r"""Plot the wall clock time.

        Given a sequence of *total_walltimes* and *ncpus* (really number
        of MPI ranks), plot the total wall clock time on *ax*, which is
        a :class:`matplotlib.axes.Axes`.  An optional *label* for the
        plot legend can also be provided.  If *scaling* is ``'strong'``,
        then a line showing perfect speedup
        (:math:`N_\mathrm{Ranks}^{-1}`) is shown for reference.  For
        ``'weak'`` scaling, the reference line is a constant.
        """
        walltimes = total_walltimes/ncpus
        ax.plot(ncpus, walltimes, 's', label=label)
        if scaling == 'strong':
            ax.plot(ncpus, walltimes[0]*ncpus[0]/ncpus, 'k--')
        elif scaling == 'weak':
            ax.plot(
                [1, ncpus[-1]],
                [walltimes[0], walltimes[0]],
                '--k',
                )
        ax.set_xlabel("Number of MPI ranks")
        ax.set_ylabel("Wall time (s)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

    def plot_speedup(self, ax, total_walltimes, ncpus):
        """Plot the parallel speedup.

        Given a sequence of *total_walltimes* and *ncpus* (really number
        of MPI ranks), plot the speedup on *ax*, which is a
        :class:`matplotlib.axes.Axes`.  Note that the speedup is
        computed relative to the lowest rank case run, which in general
        may not be serial, since, for large problems, running on a
        single processor may require too much memory, or too long a run
        time.
        """
        ax.plot(
            ncpus,
            total_walltimes[0]*ncpus/total_walltimes,
            's',
            )
        ax.plot(
            [1, ncpus[-1]],
            [1, ncpus[-1]],
            '--k',
            )
        ax.set_xlabel("Number of MPI ranks")
        ax.set_ylabel("Speedup")
        ax.set_xscale('log')
        ax.set_yscale('log')

    def plot_efficiency(self, ax, total_walltimes, ncpus):
        """Plot the parallel efficiency.

        Given a sequence of *total_walltimes* and *ncpus* (really number
        of MPI ranks), plot the parallel efficiency on *ax*, which is a
        :class:`matplotlib.axes.Axes`.  Note the caveat applies here as
        for :meth:`plot_speedup`.

        The parallel efficiency is defined as the inverse ratio of the
        actual total CPU time to the serial CPU time.
        """
        ax.plot(
            ncpus,
            total_walltimes[0]/total_walltimes,
            's'
            )
        ax.plot(
            [1, ncpus[-1]],
            [1, 1],
            '--k'
            )
        ax.set_xlabel("Number of MPI ranks")
        ax.set_ylabel("Parallel Efficiency")
        ax.set_xscale('log')
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
        total_walltimes = []
        for ncpu, grid in zip(ncpus, grids):
            # Only run on `ncpu` processes
            if MPI.COMM_WORLD.rank < ncpu:
                solver = self.integrator(ic=self.ic(grid))
                solver.run()
                total_walltimes.append(solver.total_walltime)
        return numpy.array(total_walltimes)


@unittest.skipIf(
    MPI.COMM_WORLD.size == 1,
    "More ranks required to test strong scaling"
    )
class TestStrongScaling(ScalingTest):
    """Test strong scaling for the Navier-Stokes equations.

    This class tests the strong scaling for the DNS solver.  Strong
    scaling shows how the problem scales if the total problem size
    remains constant as the number of ranks is increased.  This
    means the amount of work per rank goes down.  There will therefore
    be a maximum number of ranks that can be used, beyond which there
    problem cannot be decomposed into smaller chunks.  Strong scaling
    mimics a typical calculation, in which the solution is required for
    a specific problem size, and the question is how much benefit is
    acheived by running on more resources.
    """
    #: A list of cases to run.  Each case is defined by a tuple
    #: consisting of the problem size, and the minimum and maximum
    #: number of processors (specified as powers of two) to run on.
    cases = [
        (64, 0, 6),
        (256, 3, 8),
        (1024, 8, 10)
        ]

    def test_strong_scaling_tgv(self):
        r"""Strong scaling for the Navier-Stokes equations

        Run the DNS problem on a variety of grid sizes and number of
        tasks to obtain the strong scaling.  If insufficient MPI ranks
        are provided, only a subset (or possibly none) of the cases will
        be run.  The test passes if the speedup is at least
        :math:`N_\mathrm{Ranks}^{-0.8}`.  This test also plots the
        results.

        Results for up to 1024 ranks are shown in :numref:`fig-strong`.
        These results were run on sufficient nodes to allow at least one
        CPU per MPI rank.  The CPUs were Intel Xeon Broadwell E5-2695v4
        2.1GHz, and the system has Intel OmniPath interconnects.  Run
        times for the :math:`1024^3` simulations are about the same as
        for the solver of [Mortensen2016]_ running on an IBM Blue Gene cluster.

        .. _fig-strong:

        .. figure:: fig/test_scaling.TestStrongScaling.test_strong_scaling_tgv.png

            Strong scaling up to 1024 ranks.
        """
        with self.subplots(1, 3, figsize=(9, 3)) as (fig, axs):
            for N, nmin, nmax in self.cases:
                with self.subTest(N=N):
                    ncpus = numpy.array(
                        [2**n for n in range(nmin, nmax+1)
                         if 2**n <= MPI.COMM_WORLD.size]
                        )
                    if ncpus.size == 0:
                        continue
                    grids = (
                        SpectralGrid(
                            N,
                            comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0),
                            )
                        for ncpu in ncpus
                        )
                    total_walltimes = self.run_cases(ncpus, grids)
                    if MPI.COMM_WORLD.rank == 0:
                        # Check that the scaling is reasonable
                        fit = lambda x, A, n: A*x**-n
                        popt, pcov = scipy.optimize.curve_fit(
                            fit,
                            ncpus,
                            total_walltimes/ncpus,
                            )
                        self.assertGreaterEqual(
                            popt[1],
                            0.8,
                            )
                # Plotting is outside the subtest, so if the assert fails,
                # the plot is still generated.
                if MPI.COMM_WORLD.rank == 0:
                    # Plot the results
                    self.plot_wall_time(
                        axs[0], total_walltimes, ncpus, f"$N={N}^3$"
                        )
                    self.plot_speedup(axs[1], total_walltimes, ncpus)
                    self.plot_efficiency(axs[2], total_walltimes, ncpus)
                    fig.suptitle(cpu_description)
                    fig.tight_layout()


@unittest.skipIf(
    MPI.COMM_WORLD.size < 8,
    "More ranks required to test weak scaling"
    )
class TestWeakScaling(ScalingTest):
    r"""Test weak scaling for the Navier-Stokes equations.

    This class tests the weak scaling of the DNS solver.  Weak scaling
    is how the code scales when the amount of work per rank remains
    constant, so that the total problem size increases with the number
    of ranks.  In the case of a three-dimensional DNS solver on a cube
    mesh, each doubling of the problem size corresponds to a factor of
    eight increase in number of points.  Therefore, only rank counts
    which are powers of eight (1, 8, 64, 512, ...) correspond to an
    exact integral increase in problem size and also maintain FFT
    lengths which are powers of two.

    .. note::

        **Weak scaling of the FFT**

        Usually, for a weak scaling test, the number of ranks and the
        problem size are scaled together, so that the run time for the
        problem remains fixed.  The FFT algorithm scales as :math:`N
        \log N`, which make this difficult.  For the psuedo-spectral
        DNS, we assume that the performance is dominated by the FFTs.
        Since three-dimensional FFTs consist of three sequential
        one-dimensional FFTs, and there are :math:`N^2` one-dimensional
        FFTs per direction, the overall scaling is :math:`3 N^3 \log N`.
        Consequently, we must rescale the run times by :math:`\log N` to
        obtain constant scaling.

    Figure :numref:`fig-weak` shows the scaling for the same
    architechture as described in
    :meth:`~TestStrongScaling.test_strong_scaling_tgv`.

    .. _fig-weak:

    .. figure:: fig/test_scaling.TestWeakScaling.test_weak_scaling_tgv.png

        Weak scaling up to 1024 ranks.
    """
    def test_weak_scaling_tgv(self):
        """Weak scaling for the Navier-Stokes equations"""
        ncpus = [
            2**n for n in range(13)
            if 2**n <= MPI.COMM_WORLD.size
            ]
        size = [int(2**6*numpy.cbrt(ncpu)) for ncpu in ncpus]
        grids = (
            SpectralGrid(
                n,
                comm=MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank//ncpu, 0)
                )
            for n, ncpu in zip(sizes, ncpus)
            )
        total_walltimes = self.run_cases(ncpus, grids)
        if MPI.COMM_WORLD.rank == 0:
            with self.subplots() as (fig, ax):
                self.plot_wall_time(
                    ax,
                    total_walltimes/numpy.log2(numpy.array(sizes)),
                    ncpus,
                    scaling='weak'
                    )
                ax.set_ylabel(r"Wall time (s) / $\log_2 N$")
                ax.set_ylim(ymin=0)
                ax.get_legend().remove()
                fig.suptitle(cpu_description)
