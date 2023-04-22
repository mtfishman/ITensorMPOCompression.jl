using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test

##H = ΣⱼΣn (½ S⁺ⱼS⁻ⱼ₊n + ½ S⁻ⱼS⁺ⱼ₊n + SᶻⱼSᶻⱼ₊n)
function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNN"; NNN::Int64)
  opsum = OpSum()
  for n in 1:NNN
    J = 1.0 / n
    opsum += J * 0.5, "S+", 1, "S-", 1 + n
    opsum += J * 0.5, "S-", 1, "S+", 1 + n
    opsum += J, "Sz", 1, "Sz", 1 + n
  end
  return opsum
end

#
# InfiniteMPO has dangling links at the end of the chain.  We contract these on the outside
#   with l,r terminating vectors, to make a finite lattice MPO.
#
function terminate(h::InfiniteMPO)::MPO
  Ncell = nsites(h)
  # left termination vector
  il1 = commonind(h[1], h[2])
  il0, = noncommoninds(h[1], il1; tags="Link")
  l = ITensor(0.0, il0)
  l[il0 => dim(il0)] = 1.0 #assuming lower reg form in h
  # right termination vector
  iln = commonind(h[Ncell - 1], h[Ncell])
  ilnp, = noncommoninds(h[Ncell], iln; tags="Link")
  r = ITensor(0.0, ilnp)
  r[ilnp => 1] = 1.0 #assuming lower reg form in h
  # build up a finite MPO
  hf = MPO(Ncell)
  hf[1] = dag(l) * h[1] #left terminate
  hf[Ncell] = h[Ncell] * dag(r) #right terminate
  for n in 2:(Ncell - 1)
    hf[n] = h[n] #fill in the bulk.
  end
  # @pprint hf[1]
  # @pprint hf[2]
  # @pprint hf[3]
  return hf
end
#
# Terminate and then call expect
# for inf ψ and finite h, which is already supported in src/infinitecanonicalmps.jl
#
function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteMPO)
  return expect(ψ, terminate(h)) #defer to src/infinitecanonicalmps.jl
end

function find_links(H::Matrix{ITensor})
  lx, ly = size(H)
  @assert lx == ly

  Ms = map(n -> H[lx - n + 1, ly - n], 1:(lx - 1)) #Get the sub diagonal into an array.
  #Ms=map(n->H[n+1,n],1:lx-1) #Get the sub diagonal into an array.

  indexT = typeof(inds(Ms[1]; tags="Link")[1])
  left_inds, right_inds = indexT[], indexT[]
  @assert length(inds(Ms[1]; tags="Link")) == 1
  ir, = inds(Ms[1]; tags="Link")
  push!(right_inds, ir)
  for n in 2:length(Ms)
    il, = inds(Ms[n]; tags=tags(ir))
    #@show ir noncommonind(Ms[n],ir,tags="Link")
    ir = noncommonind(Ms[n], il; tags="Link")
    push!(left_inds, il)
    if !isnothing(ir)
      push!(right_inds, ir)
    end
  end
  return left_inds, right_inds, Ms
end

@testset "InfiniteMPOMatrix -> InfiniteMPO" begin
  Ncell = 3
  NNN = 2
  Nfinite = Ncell
  initstate(n) = "↑"
  finitstate = ["Up" for n in 1:Nfinite]
  # initstate(n) = isodd(n) ? "↑" : "↓"
  # finitstate=[isodd(n) ? "Up" : "Dn" for n=1:Ncell]
  model = Model"heisenbergNNN"()
  model_kwargs = (NNN=NNN,)
  s = infsiteinds("S=1/2", Ncell; initstate, conserve_qns=true)
  sf = s[1:Ncell]
  Hm = InfiniteMPOMatrix(model, s; model_kwargs...) #3x3 for

  #ψ = InfMPS(s, initstate) 
  # ψf = MPS(sf, finitstate) 
  # Ha=make_Heisenberg_AutoMPO(sf,NNN)
  Hi = InfiniteMPO(model, s; model_kwargs...)
  ITensors.checkflux(Hi[1])
  # Hs=InfiniteSum{MPO}(model, s;model_kwargs...)
  # Hf=MPO(model,sf;model_kwargs...)
  # Es=expect(ψ,Hs)
  #Ei=expect(ψ,Hi)
  #@show Ei
  # Es1=inner(ψf',Hs[1],ψf)
  # Ef1=inner(ψf',Hf,ψf)
  # Ef=expect(ψ,Hf)
  # Ea1=inner(ψf',Ha,ψf)
  # Ea=expect(ψ,Ha)
  # @show Es sum(Es[1:Ncell-NNN]) Es1 Ei Ef Ef1 Ea Ea1 
  # @pprint Hs[1][1]
  n = 1
  # # @pprint Hs[1][n]
  # # @pprint Hs[2][n]
  # @show inds(Hi[1],tags="Link") inds(Hi[2],tags="Link") inds(Hi[3],tags="Link") inds(Hi[4],tags="Link") 
  # @show inds(Hi[1],tags="Site") inds(Hi[2],tags="Site") inds(Hi[3],tags="Site") inds(Hi[4],tags="Site") 
  @pprint Hi[n]
  #@pprint Hmi[n][1]
  # @pprint Ha[2]
end
