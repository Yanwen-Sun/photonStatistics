import numpy as np
from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

kbar0 = 1e-2 # nominal count rate
Np = 600000 # number of pixels
Nf = 200000 # number of frames

n = 5. # number of mode, contrast is inverse of it
R = 1

nphot = 11 # dimebsion of _p

if rank == 0:
    ps = np.zeros((Nf, nphot))
if rank == 1: count = 0

for i in range(Nf):
    if i%(size-1) == rank-1:
        # status reporting
        if rank == 1: 
            count += 1
            if count%1000 == 0: print (i)
        # generate the count rate based on nominal and distribution, mode number is (1/R), average count rate is kbar0*R
        kbar = np.random.gamma(shape = R, scale = kbar0/R, size = 1)[0] 
        p = n/(kbar+n)
        x = np.random.negative_binomial(n = n, p = p, size = Np)
        #_p is the multi-photon probability
        _p =  np.bincount(x, minlength = nphot)[:nphot]/Np
        #calculate the count rate from number of photons. 
        _p[nphot-2] = x.sum()/x.size
        _p[nphot-1] = kbar 
        comm.send(_p, 0, tag = i)
    elif rank == 0:
        _p = comm.recv(source = i%(size-1)+1, tag = i)
        ps[i] = _p.copy()

if rank == 0:
    np.savez('kbar_'+str(kbar0)+\
             'beta'+str(1./n)+'_Np_'+str(Np)+'_Nf_'+str(Nf)+'_R_'+str(R)+'.npz', \
             ps = ps[:,:-1], kbar =ps[:,-1],  n = n)  
         
