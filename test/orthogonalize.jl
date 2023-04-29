using ITensors
using ITensorMPOCompression
using Revise
using Test
using Printf

include("hamiltonians/hamiltonians.jl")

import ITensorMPOCompression: gauge_fix!, is_gauge_fixed, ac_orthogonalize!


Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

verbose = false #verbose at the outer test level
verbose1 = false #verbose inside orth algos

@testset verbose = verbose "Orthogonalize" begin

  models = [
    [transIsing_MPO, "S=1/2", true],
    [transIsing_AutoMPO, "S=1/2", true],
    [Heisenberg_AutoMPO, "S=1/2", true],
    [Heisenberg_AutoMPO, "S=1", true],
    [Hubbard_AutoMPO, "Electron", false],
  ]

  @testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns, ul=$ul" for model in
                                                                                    models,
    qns in [false, true],
    ul in [lower, upper]

    eps = 1e-14
    pre_fixed = model[3] #Hamiltonian starts gauge fixed
    N = 10 #5 sites
    NNN = 4 #Include 6nd nearest neighbour interactions
    sites = siteinds(model[2], N; conserve_qns=qns)
    Hrf = reg_form_MPO(model[1](sites, NNN; ul=ul))
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state)
    E0 = inner(psi', MPO(Hrf), psi)

    @test is_regular_form(Hrf)
    #
    #  Left->right sweep
    #
    lr = left
    @test pre_fixed == is_gauge_fixed(Hrf)
    NNN >= 7 && ac_orthogonalize!(Hrf, right)
    ac_orthogonalize!(Hrf, left)
    @test is_regular_form(Hrf)
    @test check_ortho(Hrf, left)
    @test isortho(Hrf, left)
    NNN < 7 && @test is_gauge_fixed(Hrf) #Now everything should be fixed, unless NNN is big
    #
    #  Expectation value check.
    #
    E1 = inner(psi', MPO(Hrf), psi)
    @test E0 ≈ E1 atol = eps
    #
    #  Right->left sweep
    #
    ac_orthogonalize!(Hrf, right)
    @test is_regular_form(Hrf)
    @test check_ortho(Hrf, right)
    @test isortho(Hrf, right)
    @test is_gauge_fixed(Hrf) #Should still be gauge fixed
    #
    # #  Expectation value check.
    # #
    E2 = inner(psi', MPO(Hrf), psi)
    @test E0 ≈ E2 atol = eps
  end

  # @testset "Compare Dws for Ac orthogonalized hand built MPO, vs Auto MPO, NNN=$NNN, ul=$ul, qns=$qns" for NNN in
  #                                                                                                          [
  #     1, 5, 8, 12
  #   ],
  #   ul in [lower, upper],
  #   qns in [false, true]

  #   N = 2 * NNN + 4
  #   sites = siteinds("S=1/2", N; conserve_qns=qns)
  #   Hhand = reg_form_MPO(transIsing_MPO(sites, NNN; ul=ul))
  #   Hauto = transIsing_AutoMPO(sites, NNN; ul=ul)
  #   ac_orthogonalize!(Hhand, right)
  #   ac_orthogonalize!(Hhand, left)
  #   @test get_Dw(Hhand) == get_Dw(Hauto)
  # end

  end
nothing
