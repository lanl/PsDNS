import contextlib
import os
import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt

from mpi4py import MPI


class TestCase(unittest.TestCase):
    """A customized unittest class to provide plotting support.
    """
    testdir = "test_results"

    @contextlib.contextmanager
    def subplots(self, *args, **kwargs):
        try:
            os.mkdir(self.testdir)
        except FileExistsError:
            pass
        try:
            yield plt.subplots(*args, **kwargs)
        finally:
            if MPI.COMM_WORLD.rank == 0:
                plt.savefig(os.path.join(self.testdir, self.id()[12:]+".pdf"))
            plt.clf()
