import csv
import sys

import numpy


class Diagnostics(object):
    def __init__(self, tdump, outfile=sys.stderr):
        self.tdump = tdump
        self.lastdump = -1e9
        self.outfile = outfile if hasattr(outfile, 'write') else open(outfile, 'w')
        
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

        self.writer = csv.DictWriter(
            self.outfile,
            [ 'time', ] + [ field[0] for field in self.fields ]
            )
        self.writer.writeheader()

    def tke(self, equations, uhat):
        return 0.5*uhat[:3].norm()
    
    def dissipation(self, equations, uhat):
        return 2*equations.nu*sum(
            [ (1j*uhat.k[i]*uhat[j]).norm()
              for i in range(3) for j in range(3) ]
            )

    def G(self, equations, uhat):
        return sum( [
            (-uhat.k[j]*uhat.k[l]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3)
            ] )
   
    def G3(self, equations, uhat):
        return sum( [
            (-1j*uhat.k[j]*uhat.k[l]*uhat.k[m]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            ] )

    def G4(self, equations, uhat):
        return sum( [
            (uhat.k[j]*uhat.k[l]*uhat.k[m]*uhat.k[n]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3)
            ] )

    def G5(self, equations, uhat):
        return sum( [
            (1j*uhat.k[j]*uhat.k[l]*uhat.k[m]*uhat.k[n]*uhat.k[o]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3) for o in range(3)
            ] )

    def diagnostic(self, time, equations, uhat):
        self.writer.writerow(
            dict(time=time, **{
                label: func(equations, uhat) for label, func in self.fields
                })
            )
        self.outfile.flush()


class Spectra(Diagnostics):
    def diagnostic(self, time, equations, uhat):
        kmag = numpy.sqrt(numpy.sum(uhat.k**2, axis=0))
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
