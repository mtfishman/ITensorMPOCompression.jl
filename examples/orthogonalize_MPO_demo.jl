using ITensors
using ITensorMPOCompression

N = 10; #10 sites
NNN = 7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2", N);
H = transIsing_MPO(sites, NNN);
is_lower_regular_form(H) == true
pprint(H[2])
orthogonalize!(H; rr_cutoff=1e-14)
pprint(H[2])
get_Dw(H)
is_lower_regular_form(H) == true
isortho(H, left) == true
@show H.llim H.rlim ortho_lims(H)
pprint(H)
