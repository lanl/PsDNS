import mpi4py
import numpy


class Diagnostics:
    def __init__(self, tdump, **kwargs):
        super().__init__(**kwargs)
        self.tdump = tdump
        self.lastdump = -1e9
        self.tke0 = 0
        
    def diagnostics(self, time, u, uhat):
        if time-self.lastdump<self.tdump-1e-8:
            if time+self.dt-self.lastdump<self.tdump-1e-8:
                self.tke0 = self.mpiavg(u*u)
            return
        tke = self.mpiavg(u*u)
        eps = [ self.mpiavg(self.to_physical(1j*self.k[i]*uhat[j])**2)
                for i in range(3) for j in range(3) ]
        G = [ self.mpiavg(self.to_physical(-self.k[j]*self.k[l]*uhat[i])**2)
              for i in range(3) for j in range(3)
              for l in range(3) ]
        G3 = [ self.mpiavg(self.to_physical(-1j*self.k[j]*self.k[l]*self.k[m]*uhat[i])**2)
               for i in range(3) for j in range(3)
               for l in range(3) for m in range(3) ]
        G4 = [ self.mpiavg(self.to_physical(self.k[j]*self.k[l]*self.k[m]*self.k[n]*uhat[i])**2)
               for i in range(3) for j in range(3)
               for l in range(3) for m in range(3)
               for n in range(3) ]
        G5 = [ self.mpiavg(self.to_physical(1j*self.k[j]*self.k[l]*self.k[m]*self.k[n]*self.k[o]*uhat[i])**2)
               for i in range(3) for j in range(3)
               for l in range(3) for m in range(3)
               for n in range(3) for o in range(3) ]

        if self.rank == 0:
            print(
                time, 0.5*tke, 0.5*(self.tke0-tke)/self.dt if self.tke0 else 0, 2*self.nu*sum(eps),
                #sum(G), sum(G3), sum(G4), sum(G5),
                flush=True
                )

        self.lastdump = time

    def mpiavg(self, x):
        return mpi4py.MPI.COMM_WORLD.reduce(numpy.sum(x)/self.N**3)
