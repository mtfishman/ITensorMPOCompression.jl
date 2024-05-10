using ITensors, ITensorMPS
using ITensorMPOCompression
using Test
using Printf, SparseArrays
include("hamiltonians/hamiltonians.jl")

import ITensorMPOCompression: gauge_fix!, is_gauge_fixed

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

models = [
  [transIsing_MPO, "S=1/2", true],
  [transIsing_AutoMPO, "S=1/2", true],
  [Heisenberg_AutoMPO, "S=1/2", true],
  [Heisenberg_AutoMPO, "S=1", true],
  [Hubbard_AutoMPO, "Electron", false],
]

@testset "Gauge fix finite $(model[1]), ElT=$elt, qns=$qns, ul=$ul" for model in models,
  qns in [false, true],
  ul in [lower, upper],
  elt in [Float64, ComplexF64]

  eps = 1e-14

  N = 10 #5 sites
  NNN = 4 #Include 2nd nearest neighbour interactions
  sites = siteinds(model[2], N; conserve_qns=qns)
  Hrf = reg_form_MPO(model[1](elt,sites, NNN; ul=ul))
  pre_fixed = model[3] #Hamiltonian starts gauge fixed
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]

  H = MPO(Hrf)
  psi = randomMPS(elt,sites, state)
  E0 = inner(psi', H, psi)

  @test is_regular_form(Hrf)
  @test pre_fixed == is_gauge_fixed(Hrf; eps=eps)
  gauge_fix!(Hrf)
  @test is_regular_form(Hrf)
  @test is_gauge_fixed(Hrf; eps=eps)
  He = MPO(Hrf)
  E1 = inner(psi', He, psi)
  @test E0 ≈ E1 atol = eps
end

nothing
