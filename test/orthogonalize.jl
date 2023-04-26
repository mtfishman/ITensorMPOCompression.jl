using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

verbose = false #verbose at the outer test level
verbose1 = false #verbose inside orth algos

@testset verbose = verbose "Orthogonalize" begin

  models = [
    [make_transIsing_MPO, "S=1/2", true],
    [make_transIsing_AutoMPO, "S=1/2", true],
    [make_Heisenberg_AutoMPO, "S=1/2", true],
    [make_Heisenberg_AutoMPO, "S=1", true],
    [make_Hubbard_AutoMPO, "Electron", false],
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

  

  @testset "Compare Dws for Ac orthogonalized hand built MPO, vs Auto MPO, NNN=$NNN, ul=$ul, qns=$qns" for NNN in
                                                                                                           [
      1, 5, 8, 12
    ],
    ul in [lower, upper],
    qns in [false, true]

    N = 2 * NNN + 4
    sites = siteinds("S=1/2", N; conserve_qns=qns)
    Hhand = reg_form_MPO(make_transIsing_MPO(sites, NNN; ul=ul))
    Hauto = make_transIsing_AutoMPO(sites, NNN; ul=ul)
    ac_orthogonalize!(Hhand, right)
    ac_orthogonalize!(Hhand, left)
    @test get_Dw(Hhand) == get_Dw(Hauto)
  end

  models = [
    (make_transIsing_iMPO, "S=1/2"),
    (make_transIsing_AutoiMPO, "S=1/2"),
    (make_Heisenberg_AutoiMPO, "S=1/2"),
    (make_Heisenberg_AutoiMPO, "S=1"),
    (make_Hubbard_AutoiMPO, "Electron"),
  ]

  @testset "Orthogonalize iMPO Check gauge relations, H=$(model[1]), ul=$ul, qbs=$qns, N=$N, NNN=$NNN" for model in
                                                                                                           models,
    ul in [lower,upper],
    qns in [false, true],
    N in [1, 2, 3, 4],
    NNN in [1, 4]

    eps = NNN * 1e-14
    initstate(n) = "↑"
    si = infsiteinds(model[2], N; initstate, conserve_qns=qns)
    H0 = reg_form_iMPO(model[1](si, NNN; ul=ul))
    HL = copy(H0)
    @test is_regular_form(HL)
    GL = ac_orthogonalize!(HL, left; verbose=verbose1)
    DwL = Base.max(get_Dw(HL)...)
    @test is_regular_form(HL)
    @test check_ortho(HL, left) #expensive does V_dagger*V=Id
    for n in 1:N
      @test norm(HL[n].W * GL[n] - GL[n - 1] * H0[n].W) ≈ 0.0 atol = eps
    end

    HR = copy(H0)
    GR = ac_orthogonalize!(HR, right; verbose=verbose1)
    DwR = Base.max(get_Dw(HR)...)
    @test is_regular_form(HR)
    @test check_ortho(HR, right) #expensive does V_dagger*V=Id
    for n in 1:N
      @test norm(GR[n - 1] * HR[n].W - H0[n].W * GR[n]) ≈ 0.0 atol = eps
    end
    HR1 = copy(HL)
    G = ac_orthogonalize!(HR1, right; verbose=verbose1)
    DwLR = Base.max(get_Dw(HR1)...)
    @test is_regular_form(HR1)
    @test check_ortho(HR1, right) #expensive does V_dagger*V=Id
    for n in 1:N
      # D1=G[n-1]*HR1[n].W
      # @assert order(D1)==4
      # D2=HL[n].W*G[n]
      # @assert order(D2)==4
      @test norm(G[n - 1] * HR1[n].W - HL[n].W * G[n]) ≈ 0.0 atol = eps
    end
  end
end
nothing
