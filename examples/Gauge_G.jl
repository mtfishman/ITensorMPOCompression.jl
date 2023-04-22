using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf

import NDTensors: matrix
import ITensorMPOCompression: gauge_tranform!, need_guage_fix

Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)

#@testset verbose=false "LGL^-1 gauge transform, qns=$qns, lr=$lr, ul=$ul, N=$N, NNN=$NNN" for qns in [false], lr in [left], ul in [lower], N in [1], NNN in [3,4,5,6]
@testset verbose = false "LGL^-1 gauge transform, qns=$qns, lr=$lr, ul=$ul, N=$N, NNN=$NNN" for qns in
                                                                                                [
    false, true
  ],
  lr in [left, right],
  ul in [lower, upper],
  N in [1, 2, 3, 4],
  NNN in [1, 2, 3]

  initstate(n) = "↑"
  rr_cutoff = 1e-14
  eps = 1e-14
  ms = matrix_state(ul, lr)
  si = infsiteinds("Electron", N; initstate, conserve_qns=qns)
  H0 = make_Hubbard_AutoiMPO(si, NNN; ul=ul)
  # si = infsiteinds("S=1/2", N; initstate, conserve_qns=false)
  # H0=make_Heisenberg_AutoiMPO(si,NNN;ul=ul)
  if lr == left
    HL = copy(H0)
    orthogonalize!(HL, ul; orth=mirror(lr), rr_cutoff=rr_cutoff, max_sweeps=1)
    HR = copy(HL)
    # HR.llim=HL.llim #sometimes the limits dont get copied
    # HR.rlim=HL.rlim
    Gs = orthogonalize!(HL, ul; orth=lr, rr_cutoff=rr_cutoff, max_sweeps=1)
  else
    HR = copy(H0)
    orthogonalize!(HR, ul; orth=mirror(lr), rr_cutoff=rr_cutoff, max_sweeps=1)
    HL = copy(HR)
    # HL.llim=HR.llim
    # HL.rlim=HR.rlim
    Gs = orthogonalize!(HR, ul; orth=lr, rr_cutoff=rr_cutoff, max_sweeps=1)
  end
  @test is_regular_form(HL)
  @test isortho(HL, left)
  @test check_ortho(HL, left) #expensive does V_dagger*V=Id
  @test is_regular_form(HR)
  @test isortho(HR, right)
  @test check_ortho(HR, right) #expensive does V_dagger*V=Id

  @test norm(Gs[0] * HR[1] - HL[1] * Gs[1]) ≈ 0.0 atol = eps
  @test need_guage_fix(Gs, HL, 1, ms)
  @test length(Gs) == N
  @test nsites(HL) == N
  @test nsites(HR) == N

  #    Gp,HLp,HRp=gauge_tranform(Gs,HL,HR,ms)
  gauge_tranform!(Gs, HL, HR, ms)

  @test is_regular_form(HL)
  @test is_regular_form(HR)
  @test isortho(HL, left)
  @test isortho(HR, right)
  #
  #  Confirm the gauge transform breaks orthogonality for one of HLp/HRp
  #
  if lr == left
    @test !check_ortho(HL, left) #expensive does V_dagger*V=Id
    @test check_ortho(HR, right) #expensive does V_dagger*V=Id
  else
    @test check_ortho(HL, left) #expensive does V_dagger*V=Id
    @test !check_ortho(HR, right) #expensive does V_dagger*V=Id
  end

  for n in 1:N
    @test norm(Gs[n - 1] * HR[n] - HL[n] * Gs[n]) ≈ 0.0 atol = eps
    @test !need_guage_fix(Gs, HL, n, ms)
  end
end
nothing
