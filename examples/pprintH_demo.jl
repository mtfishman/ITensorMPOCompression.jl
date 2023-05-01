using ITensors
using ITensorMPOCompression

include("../test/hamiltonians/hamiltonians.jl")
N = 10; #10 sites
NNN = 7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2", N);
H = transIsing_MPO(sites, NNN);
pprint(H)
bond_spectrum = truncate!(H,left)
pprint(H)
@show bond_spectrum
nothing
