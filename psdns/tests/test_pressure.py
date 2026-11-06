"""Tests for pressure decomposition and pressure-profile correlations."""
import io

import numpy
from numpy import testing as nptest

from psdns import (
    BuoyantPressProfilesWithConcentration,
    PhysicalArray,
    RapidPressProfiles,
    SpectralGrid,
)
from psdns.equations.navier_stokes import Boussinesq, NavierStokes
from psdns import tests


class PressureTestCase(tests.TestCase):
    """Shared manufactured field and assertions for pressure tests."""

    @staticmethod
    def initial_condition(grid, with_scalar=False):
        """Return a low-mode, divergence-free shear flow."""
        nfields = 4 if with_scalar else 3
        state = PhysicalArray(grid, (nfields,))
        x, _, z = grid.x

        mean_amplitude = 0.4
        disturbance_amplitude = 0.25
        state[0] = (
            mean_amplitude*numpy.sin(z)
            + disturbance_amplitude*numpy.sin(x)*numpy.cos(z)
            )
        state[1] = 0.0
        state[2] = -disturbance_amplitude*numpy.cos(x)*numpy.sin(z)

        if with_scalar:
            state[3] = 0.3*numpy.cos(x)*numpy.sin(z)

        return state.to_spectral()

    def assert_decomposition(self, total, components):
        reconstructed = sum(components[1:], components[0])
        nptest.assert_allclose(
            numpy.asarray(total),
            numpy.asarray(reconstructed),
            rtol=0.0,
            atol=2e-14,
            )


class TestPressureDecomposition(PressureTestCase):
    """Verify the planar rapid/slow/buoyant pressure decomposition."""

    def test_rapid_plus_slow_equals_total_pressure(self):
        grid = SpectralGrid(sdims=9, pdims=12)
        equations = NavierStokes(Re=100, Sc=1)
        uhat = self.initial_condition(grid)

        total = equations.pressure(uhat).disturbance()
        rapid = equations.press_rapid(uhat)
        slow = equations.press_slow(uhat)

        self.assert_decomposition(total, (rapid, slow))
        rapid_norm = rapid.norm()
        slow_norm = slow.norm()
        with self.rank_zero(grid.comm):
            self.assertGreater(rapid_norm, 1e-10)
            self.assertGreater(slow_norm, 1e-10)
        grid.comm.Barrier()

    def test_buoyant_pressure_decomposition(self):
        grid = SpectralGrid(sdims=9, pdims=12)
        equations = Boussinesq(At=0.3, g=-0.75, Re=100, Sc=1)
        uhat = self.initial_condition(grid, with_scalar=True)

        total = equations.pressure(uhat).disturbance()
        rapid = equations.press_rapid(uhat)
        slow = equations.press_slow(uhat)
        buoyant = equations.press_buoyant(uhat)

        self.assert_decomposition(total, (rapid, slow, buoyant))
        buoyant_norm = buoyant.norm()
        with self.rank_zero(grid.comm):
            self.assertGreater(buoyant_norm, 1e-10)
        grid.comm.Barrier()


class TestPressureProfiles(PressureTestCase):
    """Verify raw pressure correlations and symmetric pressure strain."""

    def test_rapid_pressure_strain(self):
        grid = SpectralGrid(sdims=9, pdims=12)
        equations = NavierStokes(Re=100, Sc=1)
        uhat = self.initial_condition(grid)
        output = io.StringIO()
        diagnostic = RapidPressProfiles(
            tdump=1.0, grid=grid, outfile=output
            )

        correlations = diagnostic.correlations(equations, uhat)
        if grid.comm.rank == 0:
            q = correlations['pressure_gradient']
            expected = numpy.asarray([
                2*q[0, 0],
                2*q[1, 1],
                2*q[2, 2],
                q[0, 1] + q[1, 0],
                q[0, 2] + q[2, 0],
                q[1, 2] + q[2, 1],
                ])
            nptest.assert_allclose(
                correlations['pressure_strain'], expected,
                rtol=0.0, atol=1e-14,
                )
            nptest.assert_allclose(
                numpy.sum(correlations['pressure_strain'][:3], axis=0),
                0.0,
                rtol=0.0, atol=1e-14,
                )
            self.assertGreater(
                numpy.amax(numpy.abs(correlations['pressure_gradient'])),
                1e-10,
                )

        diagnostic.diagnostic(0.0, equations, uhat)
        if grid.comm.rank == 0:
            table = numpy.loadtxt(io.StringIO(output.getvalue()))
            self.assertEqual(table.shape, (grid.pdims[2], 21))
            column_header = output.getvalue().splitlines()[1]
            self.assertEqual(len(column_header.removeprefix('# ').split()), 21)
        grid.comm.Barrier()

    def test_buoyant_pressure_profile_with_concentration(self):
        grid = SpectralGrid(sdims=9, pdims=12)
        equations = Boussinesq(At=0.3, g=-0.75, Re=100, Sc=1)
        uhat = self.initial_condition(grid, with_scalar=True)
        output = io.StringIO()
        diagnostic = BuoyantPressProfilesWithConcentration(
            tdump=1.0, grid=grid, outfile=output
            )

        diagnostic.diagnostic(0.0, equations, uhat)
        if grid.comm.rank == 0:
            table = numpy.loadtxt(io.StringIO(output.getvalue()))
            self.assertEqual(table.shape, (grid.pdims[2], 25))
            column_header = output.getvalue().splitlines()[1]
            self.assertEqual(len(column_header.removeprefix('# ').split()), 25)
        grid.comm.Barrier()

