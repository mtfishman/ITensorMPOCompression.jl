using ITensors
using ITensorMPOCompression
N = 10; #10 sites
NNN = 7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2", N);
H = transIsing_MPO(sites, NNN);
pprint(H)
orthogonalize!(H; orth=right)
pprint(H)
bond_spectrum = truncate!(H; orth=left)
pprint(H)
@show bond_spectrum
nothing
