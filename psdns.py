import sys
from time import time as walltime

import mpi4py
import mpi4py_fft
import numpy

Re = 200
nu = 1/Re
N = 2**6
dt = 0.01
tfinal = 10.0
tdump = 0.1

rank = mpi4py.MPI.COMM_WORLD.Get_rank()
fft = mpi4py_fft.PFFT(
    comm=mpi4py.MPI.COMM_WORLD,
    shape=[N, N, N],
    #padding=[1.5, 1.5, 1.5],
    dtype=numpy.double,
    )

x = (2*numpy.pi/N)*numpy.mgrid[fft.local_slice(False)]
k = numpy.mgrid[fft.local_slice(True)]
# Note, use sample spacing/2pi to get radial frequencies, rather than circular frequencies.
fftfreq = numpy.fft.fftfreq(N, 1/N)
rfftfreq = numpy.fft.rfftfreq(N, 1/N)
k = numpy.array( [
    fftfreq[k[0]],
    fftfreq[k[1]],
    rfftfreq[k[2]]
    ] )

u = mpi4py_fft.newDistArray(fft, False, rank=1)
vorticity = mpi4py_fft.newDistArray(fft, False, rank=1)
uhat = mpi4py_fft.newDistArray(fft, True, rank=1)
nl = mpi4py_fft.newDistArray(fft, True, rank=1)
du = mpi4py_fft.newDistArray(fft, True, rank=1)
k2 = numpy.sum(k*k, axis=0)
P = numpy.eye(3)[:,:,None,None,None]-k[None,...]*k[:,None,...]/numpy.where(k2==0, 1, k2)

u[0] = numpy.cos(x[0])*numpy.sin(x[1])*numpy.sin(x[2])
u[1] = -numpy.sin(x[0])*numpy.cos(x[1])*numpy.sin(x[2])
u[2] = 0

for i in range(3):
    fft.forward(u[i], uhat[i])

time = 0

def mpiavg(x):
    return mpi4py.MPI.COMM_WORLD.reduce(numpy.sum(x)/N**3)

def diagnostics(u, uhat):
    tke = mpiavg(u*u)
    eps = [ mpiavg(fft.backward(1j*k[i]*uhat[j])**2)
            for i in range(3) for j in range(3) ]

    if rank == 0:
        print(time, 0.5*tke, nu*sum(eps), flush=True)


def curl(a, b):
    fft.backward(1j*(k[1]*a[2]-k[2]*a[1]), b[0])
    fft.backward(1j*(k[2]*a[0]-k[0]*a[2]), b[1])
    fft.backward(1j*(k[0]*a[1]-k[1]*a[0]), b[2])
    return b


def cross(a, b, c):
    fft.forward(a[1]*b[2]-a[2]*b[1], c[0])
    fft.forward(a[2]*b[0]-a[0]*b[2], c[1])
    fft.forward(a[0]*b[1]-a[1]*b[0], c[2])
    return c


def rhs(uhat, du):
    # Compute du/dx in physical space
    curl(uhat, vorticity)
    
    # Compute "non-linear" term in spectral space
    cross(u, vorticity, nl)

    numpy.einsum("ij...,j...->i...", P, nl, out=du)
    du -= nu*k2*uhat

    return du


time0 = walltime()
diagnostics(u, uhat)
lastdump = 0

while time<tfinal-1e-8:
    time += dt
    uhat += dt*rhs(uhat, du)

    for i in range(3):
        fft.backward(uhat[i], u[i])

    if time-lastdump>tdump-1e-8:
        diagnostics(u, uhat)
        lastdump = time
    
totaltime = mpi4py.MPI.COMM_WORLD.reduce(walltime()-time0)
if rank == 0:
    print("Total compute time = ", totaltime, file=sys.stderr)
