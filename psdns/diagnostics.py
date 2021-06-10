import csv
import sys

import numpy


class Diagnostics(object):
    def __init__(self, tdump, grid, outfile=sys.stderr):
        self.tdump = tdump
        self.lastdump = -1e9
        self._needs_close = False
        if grid.comm.rank != 0:
            return
        if hasattr(outfile, 'write'):
            self.outfile = outfile
        else:
            self.outfile = open(outfile, 'w')
            self._needs_close = True

    def __del__(self):
        if self._needs_close:
            self.outfile.close()

    def __call__(self, time, equations, uhat):
        if time-self.lastdump<self.tdump-1e-8:
            return
        self.diagnostic(time, equations, uhat)
        self.lastdump = time

    def diagnostic(self, time, equations, uhat):
        return NotImplemented


class StandardDiagnostics(Diagnostics):
    def __init__(self, fields=['tke', 'dissipation'], **kwargs):
        super().__init__(**kwargs)
        self.fields = [
            ( label, getattr(self, label) )
            for label in fields
            ]
        if kwargs['grid'].comm.rank != 0:
            return
        self.writer = csv.DictWriter(
            self.outfile,
            [ 'time', ] + [ field[0] for field in self.fields ]
            )
        self.writer.writeheader()

    def tke(self, equations, uhat):
        u2 = uhat[:3].norm()
        if uhat.grid.comm.rank == 0:
            return 0.5*u2
    
    def dissipation(self, equations, uhat):
        enstrophy = [ (1j*uhat.grid.k[i]*uhat[j]).norm()
              for i in range(3) for j in range(3) ]
        if uhat.grid.comm.rank == 0:
            return 2*equations.nu*sum(enstrophy)
        
    def urms(self, equations, uhat):
        urms = uhat[0].norm()
        if uhat.grid.comm.rank == 0:
            return urms

    def vrms(self, equations, uhat):
        vrms = uhat[1].norm()
        if uhat.grid.comm.rank == 0:
            return vrms
        
    def wrms(self, equations, uhat):
        wrms = uhat[2].norm()
        if uhat.grid.comm.rank == 0:
            return wrms

    def S(self, equations, uhat):
        gradu = uhat.grad().to_physical()
        S = [
            (gradu[i, k]*gradu[i, l]*gradu[l, k]).to_spectral().get_mode([0,0,0])
            for i in range(3) for k in range(3) for l in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum( [ s.real for s in S ] )
        
    def G(self, equations, uhat):
        G = [
            (-uhat.grid.k[j]*uhat.grid.k[l]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)
   
    def G3(self, equations, uhat):
        G = [
            (-1j*uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G4(self, equations, uhat):
        G = [
            (uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat.grid.k[n]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G5(self, equations, uhat):
        G = [
            (1j*uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat.grid.k[n]*uhat.grid.k[o]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3) for o in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def diagnostic(self, time, equations, uhat):
        row = dict(time=time, **{
            label: func(equations, uhat) for label, func in self.fields
            })
        if uhat.grid.comm.rank == 0:
            self.writer.writerow(row)
            self.outfile.flush()


class Spectra(Diagnostics):
    def diagnostic(self, time, equations, uhat):
        kmag = numpy.sqrt(numpy.sum(uhat.grid.k**2, axis=0))
        kmax = numpy.amax(kmag)
        nbins = int(max(kmag.shape)/2)
        dk = kmax/nbins
        wavenumbers = dk*numpy.arange(nbins+1)
        spectrum = numpy.zeros([nbins+1, 3])
        ispectrum = numpy.zeros([nbins+1], dtype=int)

        for k, u in numpy.nditer([kmag, (uhat[:3]*uhat[:3].conjugate()).real()]):
            spectrum[int(k/dk)] += u
            ispectrum[int(k/dk)] += 1
        spectrum *= 4*numpy.pi*(wavenumbers**2/ispectrum)[:,numpy.newaxis]
        
        for i, s in zip(wavenumbers, spectrum):
            self.outfile.write("{} {}\n".format(i, sum(s)))
        self.outfile.write("\n\n")
        self.outfile.flush()


class FieldDump(Diagnostics):
    def diagnostic(self, time, equations, uhat):
        numpy.save(
            "data{:04g}".format(time),
            uhat
            )
