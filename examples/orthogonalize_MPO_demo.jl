using ITensors, ITensorMPS
using ITensorMPOCompression
include("../test/hamiltonians/hamiltonians.jl")

N = 10; #10 sites
NNN = 7; #Include up to 7th nearest neighbour interactions
sites = siteinds("S=1/2", N);
H = transIsing_MPO(sites, NNN);
is_regular_form(H,lower) == true
pprint(H[2])
orthogonalize!(H,1)
pprint(H[2])
get_Dw(H)
is_regular_form(H,lower) == true
isortho(H, right) == true #looks at cached ortho center limits
check_ortho(H,right) == true #Does the more expensive V*V_dagger==Id contraction and test
pprint(H)
