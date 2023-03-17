using ITensors,ITensorMPOCompression


N=10
sites = siteinds("S=1/2",N,conserve_qns=true)
state=[isodd(n) ? "Up" : "Dn" for n=1:length(sites)]
psi = randomMPS(sites,state)
println("Show QN directions.  Arrows should point *away* from ortho centers")
println("MPS as constructed")
show_directions(psi)
println("MPS with ortho center on site 5")
orthogonalize!(psi,5)
show_directions(psi)

H=make_transIsing_AutoMPO(sites,1);
println("MPO as constructed from AutoMPO")
show_directions(H)
println("MPO ortho=left (orth center on site 10)")
orthogonalize!(H;orth=left)
show_directions(H)
println("MPO ortho=right (orth center on site 1)")
orthogonalize!(H;orth=right)
show_directions(H)
