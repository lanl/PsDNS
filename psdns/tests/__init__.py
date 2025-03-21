"""Unit testing for PsDNS

PsDNS uses Python :mod:`unittest` for several purposes:

  1. Conventional unit testing.
  2. Parallel scaling tests.
  3. Formal code verification.

For this reason, some of the tests are longer than typical for pure
unit testing.  Users may prefer to run only subsets of the complete
test suites, depending on what they want to test.

For code verification and scaling tests, the unit test code may output
plots.  These are put in the directory specified by
:attr:`TestCase.testdir`.

To run the full suite directly from the source directory::

  PYTHONPATH=. python -m unittest -v

To run a specific set of tests, provide the name of the tests, e.g.::

  PYTHONPATH=. python -m unittest -v psdns.tests.test_fft

The scaling tests will only run if run with MPI and sufficient tasks
are allocated.  On most systems the syntax is::

  PYTHONPATH=. mpirun -np 8 python -m unittest -v psdns.tests.test_scaling

The :mod:`~psdns.tests` module includes a customized :class:`TestCase`
class for use in PsDNS tests.
"""
import contextlib
import os
import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt

from mpi4py import MPI


class TestCase(unittest.TestCase):
    """A specialized test case for PsDNS.

    This is a specialization of the standard Python
    :class:`unittest.TestCase` which provides certain utilities useful
    for tests in PsDNS.
    """
    #: Directory in which to place unit test results.  If this
    #: directory does not exist, it will be created.
    testdir = "test_results"

    @contextlib.contextmanager
    def subplots(self, *args, **kwargs):
        """Manage plotting for test output

        For some tests, it is desirable to output plots of the
        results, in addition to the pass/fail result.  This context
        manager takes care of setting up plots, and writing the
        results to a standard location.

        The calling arguments are all passed to the
        :func:`matplotlib.pyplot.subplots` function, and the context
        manager returns the *fig* and *ax* objects returned by that
        function.  The user can then use *fig* and *ax* to do
        :mod:`matplotlib` plotting.  On exit, the plots are written to
        a file in the :attr:`testdir` directory, with the name
        ``testname.pdf``, where ``testname`` is replaced with the
        canoncial name of the unittest.
        """
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

    @contextlib.contextmanager
    def rank_zero(self, comm):
        """Catch failed unittest assertions, except on MPI rank 0

        When writing unittests which will run under MPI, it is sometimes
        necessary to test something only on one rank, typically rank 0,
        rather than on all ranks.  This context manager is intended to
        wrap assert methods so that only failures on rank zero are
        considered.

        The code

        .. code-block:: python

          with self.rank_zero(comm):
            self.assertEqual(a, b)

        is roughly equivalent to

        .. code-block:: python

          if comm.rank == 0:
            self.assertEqual(a, b)
        """
        try:
            yield
        except Exception as e:
            if comm.rank == 0:
                raise e
