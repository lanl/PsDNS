"""Tests for supporting functionality
"""
import contextlib
import filecmp
import operator
import os
import tempfile
import unittest

from mpi4py import MPI
import cupy
import numpy as np
from numpy import testing as nptest

from psdns import *
from psdns.equations.basic import Wave


class TestArrayConversion(unittest.TestCase):
    """Access local array data while preserving field arithmetic."""

    def setUp(self):
        self.grid = SpectralGrid(8)
        for datatype in self.grid.slice1 + self.grid.slice2:
            self.addCleanup(datatype.Free)

    def test_array_view(self):
        for array_type in (PhysicalArray, SpectralArray):
            for transpose in (False, True):
                with self.subTest(array_type=array_type.__name__,
                                  transpose=transpose):
                    field = array_type(self.grid, (2, 3))
                    values = cupy.arange(np.prod(field.shape), dtype=float)
                    values = values.reshape(field.shape)
                    if array_type is SpectralArray:
                        values = values * (1 + 2j)
                    field[...] = values
                    if transpose:
                        field = field.transpose(1, 0, 2, 3, 4)
                        values = values.transpose(1, 0, 2, 3, 4)
                    view = field.asarray()
                    self.assertEqual(view.shape, values.shape)
                    self.assertEqual(view.dtype, values.dtype)
                    self.assertEqual(view.strides, values.strides)
                    cupy.testing.assert_array_equal(view, values)
                    view[...] = 3
                    cupy.testing.assert_array_equal(field.asarray(), 3)
                    field[...] = 4
                    cupy.testing.assert_array_equal(view, 4)
                    view.shape = view.shape[:2] + (-1,)
                    self.assertEqual(field.asarray().shape, field.shape)

    def test_numpy_copy(self):
        for array_type in (PhysicalArray, SpectralArray):
            with self.subTest(array_type=array_type.__name__):
                field = array_type(self.grid)
                field[...] = 2
                host = cupy.asnumpy(field.asarray())
                self.assertIsInstance(host, np.ndarray)
                self.assertEqual(host.shape, field.shape)
                self.assertEqual(host.dtype, field.asarray().dtype)
                field[...] = 3
                nptest.assert_array_equal(host, 2)

    def test_implicit_conversion(self):
        for array_type in (PhysicalArray, SpectralArray):
            with self.subTest(array_type=array_type.__name__):
                field = array_type(self.grid)
                for asarray in (np.asarray, cupy.asarray):
                    with self.subTest(asarray=asarray):
                        with self.assertRaises(TypeError):
                            asarray(field)
                for copy in (False, True):
                    with self.subTest(copy=copy):
                        with self.assertRaises(TypeError):
                            np.array(field, dtype=np.complex64, copy=copy)

    def test_mixed_arithmetic(self):
        for array_type in (PhysicalArray, SpectralArray):
            field = array_type(self.grid)
            field[...] = 2
            device = cupy.full(field.shape, 3)
            for operation in (operator.add, operator.sub, operator.mul):
                for left, right, expected in (
                        (field, device, operation(2, 3)),
                        (device, field, operation(3, 2))):
                    with self.subTest(array_type=array_type.__name__,
                                      operation=operation.__name__,
                                      field_first=left is field):
                        result = operation(left, right)
                        self.assertIsInstance(result, array_type)
                        self.assertIs(result.grid, self.grid)
                        self.assertEqual(result.shape, field.shape)
                        cupy.testing.assert_array_equal(
                            result.asarray(), expected
                            )

    def test_wave_rhs(self):
        field = PhysicalArray(self.grid, cupy.sin(self.grid.x[0]))
        result = Wave().rhs(field.to_spectral())
        self.assertIsInstance(result, SpectralArray)
        self.assertIs(result.grid, self.grid)
        cupy.testing.assert_allclose(
            result.to_physical().asarray(), -cupy.cos(self.grid.x[0]),
            rtol=1e-12, atol=1e-12
            )


class MPITemporaryDirectory(object):
    """Create a temporary directory for MPI processes

    This is a wrapper to the Python standard library
    :class:`tempfile.TemporaryDirectory` which creates a single
    temporary directory and makes it accessible across all MPI ranks.
    """
    def __init__(self):
        if MPI.COMM_WORLD.rank == 0:
            self.mgr = tempfile.TemporaryDirectory()

    def __enter__(self):
        if MPI.COMM_WORLD.rank == 0:
            name = self.mgr.name
        else:
            name = ''
        return MPI.COMM_WORLD.bcast(name)

    def __exit__(self, exc_type, exc_value, exc_tb):
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank == 0:
            self.mgr.__exit__(exc_type, exc_value, exc_tb)


class TestIO(unittest.TestCase):
    """Test the SpectralArray.checkpoint() method

    Test the :meth:`psdns.bases.SpectralArray.checkpoint` and
    :meth:`psdns.bases.SpectralArray.read_checkpoint` methods.
    """
    def test_mpi_serial_parallel_identical(self):
        """Serial and parallel writing produce the same file.
        """
        with MPITemporaryDirectory() as d:
            serial = os.path.join(d, "serial")
            parallel = os.path.join(d, "parallel")
            if MPI.COMM_WORLD.rank == 0:
                s = SpectralArray(SpectralGrid(2**4, comm=MPI.COMM_SELF), (3,))
                s[...] = s.grid.k
                s.checkpoint(serial)
            s = SpectralArray(SpectralGrid(2**4), (3,))
            s[...] = s.grid.k
            s.checkpoint(parallel)
            self.assertTrue(filecmp.cmp(serial, parallel))

    def test_mpi_read(self):
        """Test parallel reading
        """
        with MPITemporaryDirectory() as d:
            checkpoint = os.path.join(d, "checkpoint")
            if MPI.COMM_WORLD.rank == 0:
                s = SpectralArray(SpectralGrid(2**4, comm=MPI.COMM_SELF), (3,))
                s[...] = s.grid.k
                s.checkpoint(checkpoint)
            s = SpectralArray(SpectralGrid(2**4), (3,))
            s.read_checkpoint(checkpoint)
            cupy.testing.assert_allclose(s.asarray(), s.grid.k)

    def test_mpi_write(self):
        """Test parallel writing
        """
        with MPITemporaryDirectory() as d:
            checkpoint = os.path.join(d, "checkpoint")
            s = SpectralArray(SpectralGrid(2**4), (3,))
            s[...] = s.grid.k
            s.checkpoint(checkpoint)
            if MPI.COMM_WORLD.rank == 0:
                s = SpectralArray(SpectralGrid(2**4, comm=MPI.COMM_SELF), (3,))
                s.read_checkpoint(checkpoint)
            cupy.testing.assert_allclose(s.asarray(), s.grid.k)
