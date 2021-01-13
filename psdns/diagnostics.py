import numpy


class Diagnostics:
    def __init__(self, tdump, **kwargs):
        super().__init__(**kwargs)
        self.tdump = tdump
        self.lastdump = -1e9
    
    def diagnostics(self, time, uhat):
        if time-self.lastdump<self.tdump-1e-8:
            return
        
        tke = self.spectral_norm(uhat)
        eps = [ self.spectral_norm(1j*self.k[i]*uhat[j])
                for i in range(3) for j in range(3) ]
        # G = [ self.spectral_norm(-self.k[j]*self.k[l]*uhat[i])
        #       for i in range(3) for j in range(3)
        #       for l in range(3) ]
        # G3 = [ self.spectral_norm(-1j*self.k[j]*self.k[l]*self.k[m]*uhat[i])
        #        for i in range(3) for j in range(3)
        #        for l in range(3) for m in range(3) ]
        # G4 = [ self.spectral_norm(self.k[j]*self.k[l]*self.k[m]*self.k[n]*uhat[i])
        #        for i in range(3) for j in range(3)
        #        for l in range(3) for m in range(3)
        #        for n in range(3) ]
        # G5 = [ self.spectral_norm(1j*self.k[j]*self.k[l]*self.k[m]*self.k[n]*self.k[o]*uhat[i])
        #        for i in range(3) for j in range(3)
        #        for l in range(3) for m in range(3)
        #        for n in range(3) for o in range(3) ]

        if self.rank == 0:
            print(
                time,
                0.5*tke,
                2*self.nu*sum(eps),
                #sum(G),
                #sum(G3),
                #sum(G4),
                #sum(G5),
                flush=True
                )

        self.lastdump = time
