"""Large-eddy simulation

A simple psuedo-spectral :math:`k-\varepsilon` for the TGV.
"""
import numpy

from psdns.bases import PhysicalArray, SpectralArray, spectral_grid
from psdns.diagnostics import StandardDiagnostics
from psdns.integrators import RungeKutta, Euler
from psdns.solvers import KEpsilon


class Equations(KEpsilon):
    def __init__(self, N, padding, tdump, t0=0, **kwargs):
        self.tdump = tdump
        self.lastdump = -1e9
        
        k, x = spectral_grid(N, padding)
        self.uhat = SpectralArray([3,], k, x)

        kwargs['Pr_k'] = 1.0
        kwargs['Pr_e'] = 1.0
        super().__init__(**kwargs)

        self.dU = 1
        self.dhdt = self.dU*(self.Ce2-self.Ce1)/2*numpy.sqrt(self.Cmu/(self.Ce1*self.Ce2))
        self.kstar = (self.Ce2-self.Ce1)/(8*self.Ce2)
        self.estar = (self.Ce2-self.Ce1)/(16*self.Ce2)*numpy.sqrt(self.Cmu*self.Ce1/self.Ce2)
        self.t0 = t0

        u, K, e = self.exact(0)
        U = PhysicalArray([5,], k, x)
        U[0] = u
        #print(U.shape)
        #print(U[0])
        #print(U[0].to_spectral().to_physical())
        U[1] = 0
        U[2] = 0
        U[3] = K
        U[4] = e
        self.uhat = U.to_spectral()
        
    def exact(self, time):
        h = self.dhdt*(time+self.t0)
        eta = numpy.arcsin(numpy.sin(self.uhat.x[1]))/h
        u = self.dU/2*eta.clip(min=-1, max=1)
        K = (self.kstar*self.dU**2*(1-eta**2)).clip(min=1e-12)
        e = (self.estar*self.dU**3/h*(1-eta**2)).clip(min=1e-12)
        return u, K, e
        
    def diagnostics(self, time, uhat):
        if time-self.lastdump<self.tdump-1e-8:
            return

        u = uhat.to_physical()
        for xi, ui in zip(uhat.x[1,0,:,0], u[:,0,:,0].T):
           print(xi, *ui)
        #for ki, ui in zip(uhat.k[1,0,:,0], abs(uhat[:,0,:,0]).T):
        #    print(ki, *ui)
        print("\n\n", flush=True)

        self.lastdump = time


solver = RungeKutta(
    dt=0.01,
    tfinal=2.0,
    equations=Equations(
        Re=1e6,
        N=[1, 2**6, 1],
        padding=[1, 1, 1],
        tdump=1.0,
        t0=20.0,
        ),
    )
solver.run()
solver.print_statistics()
