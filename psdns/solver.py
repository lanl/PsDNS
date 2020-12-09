import sys
from time import time as walltime

import mpi4py


class Euler(object):
    def __init__(self, dt, tfinal, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.tfinal = tfinal
        self.time = 0
        
    def run(self):
        self.diagnostics(self.time, self.u, self.uhat)

        time0 = walltime()
        
        while self.time<self.tfinal-1e-8:
            self.time += self.dt
            self.uhat += self.dt*self.rhs()

            for i in range(3):
                self.to_physical(self.uhat[i], self.u[i])

            self.diagnostics(self.time, self.u, self.uhat)

        self.runtime = mpi4py.MPI.COMM_WORLD.reduce(walltime()-time0)

    def print_statistics(self):
        if self.rank == 0:
            print("Total compute time = ", self.runtime, file=sys.stderr)
