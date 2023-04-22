using ITensors
using ITensorMPOCompression

N = 10; #10 sites
sites = siteinds("S=1/2", N);
ψ = randomMPS(sites)
orthogonalize!(ψ, 5)
@show ortho_lims(ψ) ψ.llim ψ.rlim
