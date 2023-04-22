using ITensors
using ITensorMPOCompression

N = 5 #5 sites
NNN = 2 #Include 2nd nearest neighbour interactions
sites = siteinds("S=1/2", N);
H = make_transIsing_MPO(sites, NNN);
@pprint(H[2]) #H[1] is a row vector, so let's see what H[2] looks like
Q, L, iq = block_qx(H[2]); #Block respecting QL
@pprint(Q)
pprint(L, iq) #we need to tell pprint which index is the row index.
