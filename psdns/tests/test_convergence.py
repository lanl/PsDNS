import numpy
import scipy.optimize

from psdns import *
from psdns.equations.linear import Wave, Burgers


class TestConvergence(tests.TestCase):
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
            fit = numpy.poly1d(numpy.polyfit(numpy.log(ns), numpy.log(errs), 1))
            with self.subplots() as (fig, ax):
                ax.loglog(
                    ns,
                    errs,
                    'ko',
                    )
                ax.plot(
                    ns,
                    numpy.exp(fit(numpy.log(ns))),
                    'r-',
                    label=f"p={fit.coeffs[0]:0.2g}",
                    )
                ax.set_xlabel("Number of points")
                ax.set_ylabel("Error")
                ax.legend()
            self.assertLess(fit.coeffs[0], -1)

    def test_wave(self):
        self.convergence_test(
            equations=Wave(),
            grids=[ SpectralGrid([2**n, 8, 2]) for  n in range(2, 8) ],
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
