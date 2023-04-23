# Define the nearest neighbor term `S⋅S` for the Heisenberg model
using ITensors
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.7f", f)
import ITensorMPOCompression: set_𝐀̂𝐜̂_block, insert_Q
N = 5
s = siteinds("S=1/2", N)
H = reg_form_MPO(make_Heisenberg_AutoMPO(s, 1))
ac_orthogonalize!(H,3) #move the ortho center to site 2 row vector on site 1 for now
#
#  Manually move the ortho center to site 2 so we can deal link order 2 MPOs for now
#
#pprint(H)

Wb2=extract_blocks(H[2],left;Ac=true)
Wb3=extract_blocks(H[3],right;Ac=true)
Wb3.𝐀̂𝐜̂=replaceind(Wb3.𝐀̂𝐜̂,Wb3.irAc,Wb2.icAc)
pprint(Wb2.𝐀̂𝐜̂)
pprint(Wb3.𝐀̂𝐜̂)

U = ITensors.randomU(Float64, s[2], s[3])
U = prime(U; plev=1)
Udag = dag(prime(U))
Phi = prime(((Wb2.𝐀̂𝐜̂ * Wb3.𝐀̂𝐜̂) * U) * Udag, -2; tags="Site")
@assert order(Phi)==6
U, ss, V, spec, iu, iv = svd(Phi, Wb2.irAc, s[2], s[2]'; utags="Link,l=2",vtags="Link,l=2", cutoff=1e-15)
@show spec
d=dim(s[1])
U*=sqrt(d)
ss/=sqrt(d)

H[2],iqp=insert_Q(H[2],U,iu,left)
H[3],iqp=insert_Q(H[3],ss*V,iv,right)
pprint(H[2])
pprint(H[3])
@assert is_regular_form(H[2])
@assert is_regular_form(H[3])
gauge_fix!(H)
@assert check_ortho(H[2],left)
#pprint(H)

nothing
