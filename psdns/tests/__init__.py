import os
import unittest

import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt


class PlotTestMixin(object):
    """A mixin class to provide uniform saving of matplotlib plots.

    This class should only be used as a mixin when subclassing
    :class:`~unittest.TestCase`, as it relies on certain methods of that
    class.
    """
    testdir = "test_results"

    def savefig(self):
        try:
            os.mkdir(self.testdir)
        except FileExistsError:
            pass
        finally:
            plt.savefig(os.path.join(self.testdir, self.id()[12:]+".pdf"))

    
