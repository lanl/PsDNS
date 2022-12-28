"""Single-mode Rayleigh-Taylor

Navier-Stokes equations with Boussinesq buoyancy terms set up for a
single mode Rayleigh-Taylor simulation.  The initial disturbance is a
velocity disturbance that roughly corresponds to the linear
instability eigenfunction.
"""
from psdns import *
from psdns.equations.navier_stokes import NavierStokes
import scipy

class Boussinesq(NavierStokes):
    """The Navier-Stokes equation with Boussinesq approximation for buoyancy.

    The incompressible Navier-Stokes equations, with a scalar
    transport equation and the Boussinesq approximation for buoyancy
    are:

    .. math::

        u_{i,i} & = 0 \\
        u_{i,t} + u_{j} u_{i,j} 
        & = -p_{,i} + \frac{1}{\text{Re}} u_{i,jj} +cg_{i} \\
        c_{,t} + u_{j}c_{,j} & = \frac{\text{Sc}}{\text{Re}} c_{,jj}
    """
    def __init__(self, Sc=1, **kwargs):
        """Return a Boussinesq equation object.

        Intantiating returns a Boussinsesq equation object with
        Reynolds number *Re* and Schmidt number *Sc*.
        """
        super().__init__(**kwargs)

        self.Sc = Sc

    def rhs(self, uhat):
        """Compute the Boussinesq right hand side.

        The numerical implementation is the same as for
        :meth:`psdns.equations.navier_stokes.NavierStokes.rhs` execpt
        for the additional bouyancy term.  As a result the momentum
        equation becomes

        .. math::

            \left(
              \frac{\partial}{\partial t} + \nu k^{2}
            \right) \hat{u}_{i}
            & = \left( \frac{k_{i}k_{j}}{k^{2}} - \delta_{ij} \right)
            \left( \widehat{u_{k}u_{j,k}} - \hat{c} g_{j} \right)
        """
        u = uhat[:3].to_physical()
        vorticity = uhat[:3].curl().to_physical()
        nl = numpy.cross(u, vorticity, axis=0)
        nl = PhysicalArray(uhat[:3].grid, nl).to_spectral()
        nl[2] += uhat[3]
        du = numpy.einsum("ij...,j...->i...", uhat[:3].grid.P, nl)
        du -= self.nu*uhat.grid.k2*uhat[:3]

        gradc = uhat[3].grad().to_physical()
        dc = numpy.einsum("i...,i...", u, gradc)
        dc = PhysicalArray(uhat.grid, dc).to_spectral()
        dc -= self.Sc*self.nu*uhat.grid.k2*uhat[3]
        
        return numpy.concatenate((du, dc[numpy.newaxis, ...]))

    def ic(self, grid):
        u = PhysicalArray(grid, (4,))
        x = u.grid.x
        eta = x[2] - 4*numpy.pi
        A = 0.1
        k = 1
        kx = ky = 1
        u[0] = -A/2*k/kx*numpy.sign(eta)*numpy.exp(-numpy.abs(k*eta))*numpy.cos(kx*x[0])*numpy.sin(ky*x[1])
        u[1] = -A/2*k/ky*numpy.sign(eta)*numpy.exp(-numpy.abs(k*eta))*numpy.sin(kx*x[0])*numpy.cos(ky*x[1])
        u[2] = A*numpy.exp(-numpy.abs(k*eta))*numpy.sin(kx*x[0])*numpy.sin(ky*x[1])
        u[3] = scipy.special.erf(eta/0.1)
        s = u.to_spectral()
        s._data = numpy.ascontiguousarray(s._data)
        return s


grid = SpectralGrid(
    sdims=[2**5-1, 2**5-1, 2**6-1],
    pdims=[3*2**4, 3*2**4, 3*2**5],
    box_size=[2*numpy.pi, 2*numpy.pi, 8*numpy.pi]
    )
equations = Boussinesq(Re=400)

solver = RungeKutta(
    dt=0.01,
    tfinal=10.0,
    equations=equations,
    ic=equations.ic(grid),
    diagnostics=[
        StandardDiagnostics(tdump=0.1, grid=grid, fields=['tke', 'dissipation', 'divU'], outfile="tgv.dat"),
        VTKDump(tdump=0.1, names=['U', 'V', 'W', 'C'], grid=grid)
        ],
    )
solver.run()
solver.print_statistics()
