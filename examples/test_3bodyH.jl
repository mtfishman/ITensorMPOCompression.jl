using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test

Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output

import ITensorMPOCompression: maxlinkdim, get_Dw
maxlinkdim(H::MPO)=maximum(get_Dw(H))

@testset "Investigate suprise effects of the inital sweep direction for 3 body Hamiltonian" begin
  ul = lower
  initstate(n) = "↑"
  verbose = false
  @printf("                               max(Dw) dumb-summed \n")
  @printf("                        Finite                      Infinite \n")
  @printf("   N  Raw autoMPO    L1    L2    R1    R2        L1    L2    R1    R2  \n")

  for N in 3:6
    # N needs to be big enough that there is block in the middle of lattice which 
    # exhibits no edge effects.
    sites = siteinds("S=1/2", N)
    si = infsiteinds("S=1/2", 1; initstate, conserve_qns=false)
    HhandL = reg_form_MPO(three_body_MPO(sites, N))
    Hhand_pbcL = three_body_MPO(si, N; pbc=true)
    Hauto = three_body_AutoMPO(sites)
    Hhand_infL = reg_form_iMPO(InfiniteMPO([Hhand_pbcL[1]]))
    Dw_raw, Dw_auto = maxlinkdim(Hhand_infL), maxlinkdim(Hauto)
    HhandR = reg_form_MPO(three_body_MPO(sites, N))
    Hhand_infR = reg_form_iMPO(InfiniteMPO([Hhand_pbcL[1]]))

    ac_orthogonalize!(HhandL,left; verbose=verbose, rr_cutoff=1e-14)
    ac_orthogonalize!(HhandR,right; verbose=verbose, rr_cutoff=1e-14)
    ac_orthogonalize!(Hhand_infL,left; verbose=verbose, rr_cutoff=1e-14)
    ac_orthogonalize!(Hhand_infR,right; verbose=verbose, rr_cutoff=1e-14)
    Dw1_L, Dw1_R, Dw1i_L, Dw1i_R = maxlinkdim(HhandL),
    maxlinkdim(HhandR), maxlinkdim(Hhand_infL),
    maxlinkdim(Hhand_infR)
    ac_orthogonalize!(HhandL,right; verbose=verbose, rr_cutoff=1e-14)
    ac_orthogonalize!(HhandR,left; verbose=verbose, rr_cutoff=1e-14)
    ac_orthogonalize!(Hhand_infL,right; verbose=verbose, rr_cutoff=1e-14)
    ac_orthogonalize!(Hhand_infR,left; verbose=verbose, rr_cutoff=1e-14)
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
  @printf("          AutoMPO            hand built\n")
  @printf("          Finite       Finite           Infinite\n")
  @printf("  N      raw trunc    raw trunc     raw trunc/L trunc/R\n")
  for N in 3:6
    sites = siteinds("S=1/2", N)
    si = infsiteinds("S=1/2", 1; initstate, conserve_qns=false)
    Hhand = reg_form_MPO(three_body_MPO(sites, N))
    Hauto = reg_form_MPO(three_body_AutoMPO(sites))
    Hhand_pbc = three_body_MPO(si, N; pbc=true)
    Hhand_inf = reg_form_iMPO(InfiniteMPO([Hhand_pbc[1]]))
    
    Dw_hand_raw, Dw_auto_raw, Dw_handinf_raw = maxlinkdim(Hhand), maxlinkdim(Hauto), maxlinkdim(Hhand_inf)
    ss_hand = truncate!(Hhand,left; cutoff=1e-15, rr_cutoff=1e-15)
    ss_auto = truncate!(Hauto,left; cutoff=1e-15, rr_cutoff=1e-15)
    ss_handinf,Ds,Hm = truncate!(Hhand_inf,left; cutoff=1e-15, rr_cutoff=1e-15)
  
    Dw_hand_trunc1, Dw_auto_trunc = maxlinkdim(Hhand), maxlinkdim(Hauto)
    Dw_handinf_truncL,Dw_handinf_truncR = maxlinkdim(Hhand_inf), maxlinkdim(Hm)
    @printf(
      "%3i     %3i  %3i      %3i  %3i       %3i   %3i  %3i \n",
      N,
      Dw_auto_raw,
      Dw_auto_trunc,
      Dw_hand_raw,
      Dw_hand_trunc1,
      Dw_handinf_raw,
      Dw_handinf_truncL,
      Dw_handinf_truncR
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
      #@test sqrt(sum(ha .^ 2)) / N ≈ 0.0 atol = 1e-12
    end
  end
end
