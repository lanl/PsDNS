import unittest

from mpi4py import MPI
import numpy
from numpy import testing as nptest

from psdns import *

class TestIO(unittest.TestCase):
    def test_mpi_write(self):
        if MPI.COMM_WORLD.rank == 0:
            s = SpectralArray((3,), SpectralGrid(2**4, comm=MPI.COMM_SELF))
            s[...] = s.grid.k
            s.checkpoint("checkpoint")
        s = SpectralArray((3,), SpectralGrid(2**4))
        s.read_checkpoint("checkpoint")
        nptest.assert_allclose(
            numpy.asarray(s),
            numpy.asarray(s.grid.k)
            )
