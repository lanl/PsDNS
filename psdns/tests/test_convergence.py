import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy
import scipy.optimize

from psdns import *
from psdns.equations.linear import Wave, Burgers


class TestConvergence(unittest.TestCase):   
    def convergence_test(self, equations, grids, solver_args, filename):
        errs = []
        for grid in grids:
            solver = RungeKutta(
                equations=equations,
                ic=equations.exact(grid, 0).to_spectral(),
                **solver_args,
            )
            solver.run()
            errs.append(
                (solver.uhat.to_physical()
                -equations.exact(solver.uhat.grid, solver.time)).norm(),
                )
        #     plt.plot(
        #         solver.uhat.grid.x[0,:,0,0],
        #         (solver.uhat.to_physical()
        #         -equations.exact(solver.uhat.grid, solver.time))[:,0,0],
        #         label=f"n={grid.x.shape[1]}",
        #         )
        # plt.legend()
        if grids[0].comm.rank == 0:
            ns = [ grid.pdims[0] for grid in grids ]
            fit = lambda x, A, n: A*x**n
            popt, pcov = scipy.optimize.curve_fit(
                fit,
                ns,
                errs,
                method='trf',
                )
            plt.loglog(
                ns,
                errs,
                'ko',
                )
            plt.plot(
                ns,
                fit(ns, *popt),
                'r-',
                label=u"${:.2f}n^{{{:.2f}}}$".format(*popt)
                )
            plt.legend()
            plt.xlabel("Number of points")
            plt.ylabel("Error")
            plt.savefig(filename)
            plt.clf()
            self.assertLess(popt[1], -0.5)

    def test_wave(self):
        self.convergence_test(
            equations=Wave(),
            grids=[ SpectralGrid([2**n, 8, 2]) for  n in range(2, 12) ],
            solver_args={'dt': 0.001, 'tfinal': 1.0},
            filename="wave.pdf"
            )

    def test_burgers(self):
        self.convergence_test(
            equations=Burgers(),
            grids=[ SpectralGrid([2**n, 8, 2]) for  n in range(3, 7) ],
            solver_args={'dt': 0.001, 'tfinal': 1.0},
            filename="burgers.pdf"
            )
