using ITensors
using ITensorMPOCompression

N=10
sites = siteinds("SpinHalf", N;conserve_qns=true)
# state=[isodd(n) ? "Up" : "Dn" for n=1:N]
# psi=randomMPS(sites,state)

#H=make_transIsing_AutoMPO(sites,3)
H=make_Heisenberg_AutoMPO(sites,1)
#H=make_transIsing_AutoMPO(sites,1)
pprint(H[4])
# @show inds(H[1]) flux(H[1])
@show inds(H[4]) flux(H[4])
# @show inds(H[N]) flux(H[N])
# @show flux(H)
nothing