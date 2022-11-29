"""DNS of a shear-layer
"""
import numpy

from psdns import *
from psdns.equations.navier_stokes import NavierStokes


L = 2*numpy.pi
N = 6


class MeanProfile(Diagnostic):
    @staticmethod
    def planaravg(f, grid, axis=()):
        favg = grid.comm.reduce(numpy.sum(f, axis=axis+(-3, -2)))
        if grid.comm.rank == 0:
            favg /= grid.pdims[0]*grid.pdims[1]
        return favg
    
    def diagnostic(self, time, equations, uhat):
        # This diagnostic relies on the fact that the physical space data
        # has each task containing the entire span in z.
        u = uhat.to_physical()
        ubar = self.planaravg(u, uhat.grid)
        
        # From here, we need u' fluctuation.  We redefine uhat and u to be
        # fluctuations.  There are two ways to do that:
        if False:
            ubar = uhat.grid.comm.bcast(ubar)
            u = u - ubar[:,numpy.newaxis,numpy.newaxis,:]
            uhat = u.to_spectral()
        else:
            uhat = uhat.copy()
            # In spectral space, u' is uhat with the zero modes in x & y set to zero.
            if uhat.grid.local_spectral_slice[0].start == 0:
                uhat[:,0,:,:] = 0
            if uhat.grid.local_spectral_slice[1].start == 0:
                uhat[:,:,0,:] = 0
            u = uhat.to_physical()
            
        Rij = [
            self.planaravg(u[i]*u[j], uhat.grid)
            for i, j in [ (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2) ]
            ]
        gradu = uhat.grad()
        grad2u = gradu.grad().to_physical()
        gradu = gradu.to_physical()
        eps = self.planaravg(gradu**2, uhat.grid, axis=(0, 1))
        S =  [
            self.planaravg(gradu[i, k]*gradu[j, k]*gradu[i, j], uhat.grid)
            for i in range(3) for j in range(3) for k in range(3)
            ]
        G = self.planaravg(grad2u**2, uhat.grid, axis=(0, 1, 2))
        
        if uhat.grid.comm.rank == 0:
            numpy.savetxt(
                self.outfile,
                (numpy.vstack([ uhat.grid.x[2,0,0,:], ubar ] + Rij + [ eps, sum(S), G ])).T,
                header="t = {}\nt u v w Rxx Ryy Rzz Rxy Rxz Ryz epsilon S G".format(time)
                )
            self.outfile.write("\n\n")
            self.outfile.flush()


def shear_ic(grid):
    u = PhysicalArray(grid, (3,))
    u[0] = numpy.where((grid.x[2]>L) & (grid.x[2]<=3*L), -0.5, 0.5)
    for A, a, b, c in [ ( 1e-4, 1, 4, 0 ), ( 1e-4, 1, 4, 1 ) ]:
        u[0] = u[0] - 2*a/b*A*(grid.x[2]-3*L)*numpy.exp(-a*(grid.x[2]-3*L)**2) \
          *numpy.cos(b*grid.x[0]) \
          *numpy.cos(c*grid.x[1])
        u[2] = u[2] + A*numpy.exp(-a*(grid.x[2]-3*L)**2) \
          *numpy.sin(b*grid.x[0]) \
          *numpy.cos(c*grid.x[1])
    s = u.to_spectral()
    s._data = numpy.ascontiguousarray(s._data)
    return s


grid = SpectralGrid(
    sdims=[2**N-1, 2**N-1, 4*2**N-1],
    pdims=[3*2**(N-1), 3*2**(N-1), 12*2**(N-1)],
    box_size=[L, L, 4*L]
    )
equations = NavierStokes(Re=1000)

solver = RungeKutta(
    dt=0.05,
    tfinal=100.0,
    equations=equations,
    ic=shear_ic(grid),
    diagnostics=[
        #FieldDump(tdump=1.0, grid=grid),
        #StandardDiagnostics(tdump=1.0, grid=grid, fields=['divU'], outfile="std.dat"),
        MeanProfile(tdump=1.0, grid=grid, outfile="velocity2.dat")
        ],
    )
solver.run()
solver.print_statistics()
