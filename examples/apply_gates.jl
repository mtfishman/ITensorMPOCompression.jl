# Define the nearest neighbor term `S⋅S` for the Heisenberg model
using ITensors
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.5f", f)
using Random
Random.Random.seed!(12345);

import ITensorMPOCompression: insert_Q, reg_form_Op, is_gauge_fixed, gauge_fix!, extract_blocks, check, d, grow
include("../test/hamiltonians/hamiltonians.jl")

# Set up primes as U(s,s'') and U†(s',s''')  U 
function two_site_gate(ElT::Type{<:Number},s,n::Int64)
  U = ITensors.randomU(ElT, dag(s[n]), dag(s[n+1]))
  U = prime(U; plev=1)
  U_dag = dag(prime(U))
  return U,U_dag
end

function apply(U_gate::ITensor,WL::reg_form_Op,WR::reg_form_Op,U_dag_gate::ITensor)
  # @show commonind(WL.W,WR.W) commonind(WR.W,WL.W)
  WbL=extract_blocks(WL,left;Ac=true) #Get the Ac block on the left side of the bond
  WbR=extract_blocks(WR,right;Ac=true) #Get the bA block on the right side of the bond
  WbR.𝐀̂𝐜̂=replaceind(WbR.𝐀̂𝐜̂,WbR.𝐀̂𝐜̂.ileft,WbL.𝐀̂𝐜̂.iright) #Get the indices between Ac and bA to agree.
  Phi = prime(((WbL.𝐀̂𝐜̂.W * WbR.𝐀̂𝐜̂.W) * U_gate) * U_dag_gate, -2; tags="Site")
  @assert order(Phi)==6
  Linds=(WbL.𝐀̂𝐜̂.ileft, siteinds(WL)...)
  Rinds=(WbR.𝐀̂𝐜̂.iright, siteinds(WR)...)
  NL=prod(dims(Linds))
  NR=prod(dims(Rinds))
  N=Base.min(NL,NR) #Dimension across the bond without any compression/trunction.
  U, ss, V, spec, iu, iv = svd(Phi,Linds...; utags=tags(WL.iright),vtags=tags(WR.ileft), cutoff=1e-15)
  dh = d(WbL) #dimension of local Hilbert space.
  @assert abs(dh - round(dh)) == 0.0 #better be an integer!
  U *= sqrt(dh) #Rescale so weight doesn not get shifted to the end of the lattice.
  ss /= sqrt(dh)
  #
  #  Because of the structure of Phi, U is not orthogonal to 𝕀, and guage fixing Ac and/or bA does not fix this.
  #
  WR⎖,iqpr=insert_Q(WR,ss*V,iu,right)
  WL⎖,iqpl=insert_Q(WL,U,dag(iu),left)
  WR⎖=replaceind(WR⎖,iqpr,dag(iqpl))
  check(WL⎖)
  check(WR⎖)
  #
  #  Show truncation stats
  #
  Ns=size(ss,1)
  mins=minimum(diag(ss))
  println("N=min($NL,$NR)=$N compressed down to Ns=$Ns, or $(100.0*(N-Ns)/N)% reduction, min(s)=$mins")
  
  @assert is_regular_form(WL⎖)
  @assert is_regular_form(WR⎖)
  return WL⎖,WR⎖
end

function gate_sweep!(ElT::Type{<:Number},sites,H::reg_form_MPO)
  N=length(H)
  for n in 1:N-1
    U,Udag=two_site_gate(ElT,sites,n)
    H[n],H[n+1]=apply(U,H[n],H[n+1],Udag)
  end
end

ElT=Float64
N,NNN = 10,2
# sites = siteinds("S=1/2", N;conserve_qns=true)
# H = reg_form_MPO(transIsing_AutoMPO(ElT,sites, NNN))
# H = reg_form_MPO(Heisenberg_AutoMPO(ElT,sites, NNN))
sites = siteinds("Electron", N;conserve_qns=true)
H = reg_form_MPO(Hubbard_AutoMPO(ElT,sites, NNN))
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi = random_mps(sites, state)
# show_directions(H)
@show get_Dw(H)
orthogonalize!(H,right) 
@assert check_ortho(H,right)
E0 = inner(psi', MPO(H), psi)
# show_directions(H)

gate_sweep!(ElT,sites,H)
show_directions(H)
#
# Get sweep destroyes orthogonality and energy expectation.
#
@assert !check_ortho(H,right)
@assert !check_ortho(H,left)
E1 = inner(psi', MPO(H), psi)

@show get_Dw(H)
orthogonalize!(H,right)
orthogonalize!(H,left)
E2 = inner(psi', MPO(H), psi)
@show E0 E1 E2

bond_specs=truncate!(H,right)
@show bond_specs

nothing
