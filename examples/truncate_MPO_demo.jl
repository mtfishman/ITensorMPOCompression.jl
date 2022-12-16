using ITensors
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output
N=14; #14 sites
NNN=7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2",N);
H=make_transIsing_MPO(sites,NNN);
is_lower_regular_form(H)==true
spectrums=truncate!(H;cutoff=1e-15)
pprint(H[7])
@show get_Dw(H)
@show spectrums
is_lower_regular_form(H)==true
is_orthogonal(H,left)==true