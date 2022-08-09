"""Tests for supporting functionality
"""
import contextlib
import filecmp
import os
import tempfile
import unittest

from mpi4py import MPI
import numpy
from numpy import testing as nptest

from psdns import *


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


@unittest.skip("MPI I/O Datatypes need fixing for pencils.")
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
            nptest.assert_allclose(
                numpy.asarray(s),
                numpy.asarray(s.grid.k)
                )

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
            nptest.assert_allclose(
                numpy.asarray(s),
                numpy.asarray(s.grid.k)
                )
