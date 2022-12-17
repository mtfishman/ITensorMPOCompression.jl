using ITensors
using ITensorMPOCompression

N=10; #10 sites
NNN=7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2",N);
H=make_transIsing_MPO(sites,NNN);
is_lower_regular_form(H)==true
pprint(H[2])
orthogonalize!(H;epsrr=1e-14)
pprint(H[2])
get_Dw(H)
is_lower_regular_form(H)==true
is_orthogonal(H,left)==true
