Using PsDNS
===========

Running Examples
----------------

PsDNS includes implementations for several PDEs, along with standard
integrators and diagnostics.  To perform a simulation, you must create
a script which instantiates the solver and runs it.  Simple example
scripts are in the :file:`examples` directory.

To run an example script without installing the pacakge::

  PYTHONPATH=. python examples/dns.py

Writing a PsDNS script is pretty self-explanatory, using one of the
examples as a template.  Refer to :ref:`psdns package` for details on
all the available classes and options.

Adding Diagnostics
------------------

PsDNS provides a number of diagnostics.  One of the arguments to the
:class:`~psdns.integrators.Integrator` class is a list of the
diagnostics.  Each diagnostic has its own output file, and its own
dump interval.  You can add as many instances of each diagnostic as
you want to the list.  Documentation for the available diagnostics are
in the :mod:`~psdns.diagnostics` module.

You can also create new diagnostics classes, by subclassing
:class:`~psdns.diagnostics.Diagnostic`.  Look at some of the provided
diagnostics to see how to do that.

Implementing Equations
----------------------

The :mod:`~psdns.equations` module includes implementation of several
important equations, including the incompressible Navier-Stokes
equations, and the simple Smagorinsky LES model.  Note that, since
only one equation is used in a given simulation, the
:mod:`~psdns.equations` module does not import anything by default,
and users should import the specific equation they want.

Equations are primarily written in `Numpy <https://numpy.org>`_.
Implementing new equations should be simple by following the pattern
of some of those distributed with PsDNS. 
