import numpy



def spectrum(binctr):
    spectrum = binctr**(-2)
    return spectrum

#def band(self, grid, kmin, kmax, E0, seed=123):
#       if seed is None:
#            warnings.warn(
#                "A seed of None for Boussinesq.band() will result "
#                "in inconsistent initialization across MPI ranks."
#                )
                
        # Extract 2D grid slice for computation
        #x = grid.x[:2, :, :, 0]
 #        z = numpy.zeros_like(x[0])
        
        # Initialize RNG with consistent seed
rng = numpy.random.default_rng(123)
count = 0

phase_p = []
phase_q = []
radik_v = []

kmin = 1
kmax = 10
E0 = 1

#initialize
for p in range(-kmax, kmax + 1):
    for q in range(-kmax, kmax + 1):
        k = numpy.sqrt(p**2 + q**2)
        if kmin <= k <= kmax:
            count += 1
            phase_p.append (rng.random())
            phase_q.append (rng.random())
            radik_v.append (k)


phase_p = numpy.array(phase_p)
phase_q = numpy.array(phase_q)
radik_v = numpy.array(radik_v)

#build the bins: Nbins hardcoded
Nb = 40


binedges = numpy.linspace(kmin,kmax,Nb+1)
binctr = (binedges[1:] + binedges[0:-1])*0.5

dk = binedges[1] - binedges[0] 

BinId = numpy.digitize(radik_v,binedges[:-2],right=True)
nModePerBin = numpy.bincount(BinId)

print(BinId)



#energy per bin 
Ebin = spectrum(binctr)*binctr*dk

#energy per mode 
EtildaBin = Ebin[BinId]/nModePerBin[BinId]

#correction factor
alpha = numpy.sqrt(E0/sum(EtildaBin))

#correct energy for each mode
Emode = alpha**2*EtildaBin

print(numpy.sum(Emode) - E0)




