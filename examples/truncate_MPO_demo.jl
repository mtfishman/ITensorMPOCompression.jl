using ITensors
using ITensorMPOCompression
include("../test/hamiltonians/hamiltonians.jl")

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output
N = 10; #14 sites
NNN = 7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2", N);
H = transIsing_MPO(sites, NNN);
is_regular_form(H,lower) == true
spectrums = truncate!(H,left)
pprint(H[5])
@show get_Dw(H)
@show spectrums
is_regular_form(H,lower) == true
isortho(H, left) == true
check_ortho(H, left) == true
