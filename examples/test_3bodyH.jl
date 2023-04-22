using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test

Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output

@testset "Investigate suprise effects of the inital sweep direction for 3 body Hamiltonian" begin
  ul = lower
  initstate(n) = "↑"
  verbose = false
  @printf("                               max(Dw) dumb-summed \n")
  @printf("                        Finite                      Infinite \n")
  @printf("   N  Raw autoMPO    L1    L2    R1    R2        L1    L2    R1    R2  \n")

  for N in 3:10
    # N needs to be big enough that there is block in the middle of lattice which 
    # exhibits no edge effects.
    sites = siteinds("S=1/2", N)
    si = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
    HhandL = make_3body_MPO(sites, N)
    Hhand_pbcL = make_3body_MPO(si, N; pbc=true)
    Hauto = make_3body_AutoMPO(sites)
    Hhand_infL = InfiniteMPO([Hhand_pbcL[1]])
    Dw_raw, Dw_auto = maxlinkdim(Hhand_infL), maxlinkdim(Hauto)
    HhandR = copy(HhandL)
    Hhand_infR = copy(Hhand_infL)

    orthogonalize!(HhandL; verbose=verbose, orth=left, max_sweeps=1, rr_cutoff=1e-14)
    orthogonalize!(HhandR; verbose=verbose, orth=right, max_sweeps=1, rr_cutoff=1e-14)
    orthogonalize!(Hhand_infL; verbose=verbose, orth=left, max_sweeps=1, rr_cutoff=1e-14)
    orthogonalize!(Hhand_infR; verbose=verbose, orth=right, max_sweeps=1, rr_cutoff=1e-14)
    Dw1_L, Dw1_R, Dw1i_L, Dw1i_R = maxlinkdim(HhandL),
    maxlinkdim(HhandR), maxlinkdim(Hhand_infL),
    maxlinkdim(Hhand_infR)
    orthogonalize!(HhandL; verbose=verbose, orth=right, max_sweeps=1, rr_cutoff=1e-14)
    orthogonalize!(HhandR; verbose=verbose, orth=left, max_sweeps=1, rr_cutoff=1e-14)
    orthogonalize!(Hhand_infL; verbose=verbose, orth=right, max_sweeps=1, rr_cutoff=1e-14)
    orthogonalize!(Hhand_infR; verbose=verbose, orth=left, max_sweeps=1, rr_cutoff=1e-14)
    Dw2_L, Dw2_R, Dw2i_L, Dw2i_R = maxlinkdim(HhandL),
    maxlinkdim(HhandR), maxlinkdim(Hhand_infL),
    maxlinkdim(Hhand_infR)

    @printf(
      "%4i  %4i  %4i   %4i  %4i  %4i  %4i      %4i  %4i  %4i  %4i \n",
      N,
      Dw_raw,
      Dw_auto,
      Dw1_L,
      Dw2_L,
      Dw1_R,
      Dw2_R,
      Dw1i_L,
      Dw2i_L,
      Dw1i_R,
      Dw2i_R
    )
  end
end

@testset "Tabulate MPO Dw reduction for 3 body Hamiltonian" begin
  ul = lower
  initstate(n) = "↑"
  @printf("     |--------------------------max(Dw)-------------------------|\n")
  @printf("          AutoMPO               hand built\n")
  @printf("          Finite               Finite              Infinite\n")
  @printf("  N    raw orth trunc   raw orth tr1 tr2 tr3   orth trunc\n")
  for N in 3:10
    sites = siteinds("S=1/2", N)
    si = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
    Hhand = make_3body_MPO(sites, N)
    Hauto = make_3body_AutoMPO(sites)
    Hhand_pbc = make_3body_MPO(si, N; pbc=true)
    Hhand_inf = InfiniteMPO([Hhand_pbc[1]])
    Dw_hand_raw, Dw_auto_raw = maxlinkdim(Hhand), maxlinkdim(Hauto)
    orthogonalize!(Hhand; rr_cutoff=1e-14)
    orthogonalize!(Hauto; rr_cutoff=1e-14)
    Gs_hand = orthogonalize!(Hhand_inf; rr_cutoff=1e-14)
    Dw_hand_orth, Dw_auto_orth = maxlinkdim(Hhand), maxlinkdim(Hauto)
    Dw_handinf_orth = maxlinkdim(Hhand_inf)
    ss_hand = truncate!(Hhand; cutoff=1e-15, rr_cutoff=1e-15)
    ss_auto = truncate!(Hauto; cutoff=1e-15, rr_cutoff=1e-15)
    truncate!(Hhand_inf, Gs_hand, left; cutoff=1e-15, rr_cutoff=1e-15)
    Dw_hand_trunc1, Dw_auto_trunc = maxlinkdim(Hhand), maxlinkdim(Hauto)
    Dw_handinf_trunc = maxlinkdim(Hhand_inf)
    ss_hand = truncate!(Hhand; cutoff=1e-15, rr_cutoff=1e-15)
    Dw_hand_trunc2 = maxlinkdim(Hhand)
    ss_hand = truncate!(Hhand; cutoff=1e-15, rr_cutoff=1e-15)
    Dw_hand_trunc3 = maxlinkdim(Hhand)
    @printf(
      "%3i   %3i  %3i  %3i    %3i  %3i  %3i  %3i  %3i  %3i  %3i     \n",
      N,
      Dw_auto_raw,
      Dw_auto_orth,
      Dw_auto_trunc,
      Dw_hand_raw,
      Dw_hand_orth,
      Dw_hand_trunc1,
      Dw_hand_trunc2,
      Dw_hand_trunc3,
      Dw_handinf_orth,
      Dw_handinf_trunc,
    )

    #
    #  Make sure the bond spectrum for auto, presummed amd dum summed hamiltonians
    #  are all identical to machine precision.
    #
    @test length(ss_hand) == length(ss_auto)
    for nb in 1:length(ss_hand)
      ss_h = eigs(ss_hand[nb])
      ss_a = eigs(ss_auto[nb])
      ha = .√(ss_h) - .√(ss_a)
      #@show sqrt(sum(pd.^2))/N
      @test sqrt(sum(ha .^ 2)) / N ≈ 0.0 atol = 1e-12
    end
  end
end
