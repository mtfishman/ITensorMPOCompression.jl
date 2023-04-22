# Define the nearest neighbor term `S⋅S` for the Heisenberg model
using ITensors
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.7f", f)

N = 10
s = siteinds("S=1/2", N)
H = make_Heisenberg_AutoMPO(s, 1)
orthogonalize!(H; orth=right)
#
#  Manually move the ortho center to site 2 so we can deal link order 2 MPOs for now
#
H[1], H[2] = orthogonalize!(H[1], H[2], lower; orth=left)
H[2], H[3] = orthogonalize!(H[2], H[3], lower; orth=left)
pprint(H)

n1, n2 = 2, 3
@pprint(H[n1])
Wl, Wr = H[n1], H[n2]
il12 = commonindex(Wl, Wr)
il = noncommonind(Wl, il12)
# Vn1,qn1=getV(H[n1],V_offsets(matrix_state(lower,left)))
# Vn2,qn2=getV(H[n2],V_offsets(matrix_state(lower,right)))
# @pprint(Vn1)
# @pprint(Vn2)
# il12=commonindex(H[n1],H[n2])
# ilv1,=inds(Vn1,tags=tags(il12))
# ilv2,=inds(Vn2,tags=tags(il12))
##il=noncommonind(Vn1,ilv1)
# Vn2=replaceind(Vn2,ilv2,ilv1)
# @show inds(Vn1) inds(Vn2)

U = ITensors.randomU(Float64, s[n1], s[n2])
U = prime(U; plev=1, 1)

Udag = dag(prime(U, 1))
Udag = prime(Udag; plev=2, 1)

Phi = prime(((Wl * Wr) * U) * Udag, -2; tags="Site")
#Phi=prime(((Vn1*Vn2)*U)*Udag,-2,tags="Site")
#Phi=Vn1*Vn2
# @show dims(Phi) inds(Phi,tags="Link")
# @show il
U, ss, V = svd(Phi, il, s[n1], s[n1]'; cutoff=1e-15)
@show diag(ss)
# #@show inds(U)
#@show sqrt(ss)
U = U * sqrt(ss)
U = replacetags(U, "v", "l=$n1")
il2, = inds(U; tags="l=$n1")
# @show inds(U)
# @pprint(U)
#scale=sqrt(ss[1,1])
# scale=U[il=>1,il2=>1,s[n1]=>1,s[n1]'=>1]
#U.*=scale
# pprint(U)
# for i in 1:4
#     @show slice(U,il=>1,il2=>i)
# end
# #@show typeof(scale)
# #@show slice(U,il=>1,il2=>1)*scale
# #@show slice(U,il=>dim(il),il2=>dim(il2))

L, Q = lq(U, il, s[n1], s[n1]'; rr_cutoff=1e-14)
L = replacetags(L, "lq", "l=$n1")
il2, = inds(L; tags="l=$n1")
scale = L[il => 1, il2 => 1, s[n1] => 1, s[n1]' => 1]
L ./= scale
pprint(L)
@show slice(L, il => 1, il2 => 1)
for i in 5:dim(il2)
  @show slice(L, il => dim(il), il2 => i)
end

# Q,L=ql(Phi,il,s[n1],s[n1]';rr_cutoff=1e-14)
# Q=replacetags(Q,"ql","l=$n1")
# il2,=inds(Q,tags="l=$n1")
# pprint(Q)
# for i in 1:dim(il2)
#     @show slice(Q,il=>dim(il),il2=>i)
# end
# for i in 1:4
#     @show slice(Q,il=>1,il2=>i)
# end

# # for i in 4:8
# #     @show slice(L,il=>dim(il),il2=>i)
# # end
# #@show Q
nothing
