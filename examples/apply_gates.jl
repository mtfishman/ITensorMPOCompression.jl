# Define the nearest neighbor term `S⋅S` for the Heisenberg model
using ITensors
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
import ITensorMPOCompression: insert_Q, reg_form_Op

function two_site_gate(s,n::Int64)
  U = ITensors.randomU(Float64, s[n], s[n+1])
  U = prime(U; plev=1)
  Udag = dag(prime(U))
  return U,Udag
end

function apply(U::ITensor,WL::reg_form_Op,WR::reg_form_Op,Udag::ITensor)
  WbL=extract_blocks(WL,left;Ac=true)
  WbR=extract_blocks(WR,right;Ac=true)
  WbR.𝐀̂𝐜̂=replaceind(WbR.𝐀̂𝐜̂,WbR.irAc,WbL.icAc)
  #@show inds(WbL.𝐀̂𝐜̂) inds(WbR.𝐀̂𝐜̂)
  Phi = prime(((WbL.𝐀̂𝐜̂ * WbR.𝐀̂𝐜̂) * U) * Udag, -2; tags="Site")
  #@show inds(Phi)
  @assert order(Phi)==6
  is=siteinds(WL)
  U, ss, V, spec, iu, iv = svd(Phi, WbL.irAc, is...; utags=tags(WL.iright),vtags=tags(WR.ileft), cutoff=1e-15)
  WL⎖,iqpl=insert_Q(WL,U,iu,left)
  WR⎖,iqpr=insert_Q(WR,ss*V,iv,right)
  WR⎖=replaceind(WR⎖,iqpr,iqpl)
  check(WL⎖)
  check(WR⎖)
  
  @assert is_regular_form(WL⎖)
  @assert is_regular_form(WR⎖)
  return WL⎖,WR⎖
end

function gate_sweep!(H::reg_form_MPO)
  N=length(H)
  for n in 1:N-1
    U,Udag=two_site_gate(s,n)
    H[n],H[n+1]=apply(U,H[n],H[n+1],Udag)
  end
end

N,NNN = 5,2
s = siteinds("S=1/2", N)
H = reg_form_MPO(make_Heisenberg_AutoMPO(s, NNN))
@assert is_gauge_fixed(H)
ac_orthogonalize!(H,right) 
@assert is_gauge_fixed(H)

gate_sweep!(H)
# U,Udag=two_site_gate(s,2)
# H[2],H[3]=apply(U,H[2],H[3],Udag)
# #@assert check_ortho(H[2],left)
pprint(H)
ac_orthogonalize!(H,right)
pprint(H)

nothing
