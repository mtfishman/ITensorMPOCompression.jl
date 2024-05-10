using ITensors, ITensorMPS
using ITensorMPOCompression
using Test

import ITensors: maxlinkdim

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output
include("../test/hamiltonians/hamiltonians.jl")


maxlinkdim(H::MPO)=maximum(get_Dw(H))
maxlinkdim(H::reg_form_MPO)=maximum(get_Dw(MPO(H)))

@testset "Investigate suprise effects of the inital sweep direction for 3 body Hamiltonian" begin
  ul = lower
  initstate(n) = "↑"
  verbose = false
  @printf("                      max(Dw) dumb-summed \n")
  @printf("   N  Raw   autoMPO    L1    L2    R1    R2  \n")

  for N in 3:15
    # N needs to be big enough that there is block in the middle of lattice which 
    # exhibits no edge effects.
    sites = siteinds("S=1/2", N)
    HhandL = reg_form_MPO(three_body_MPO(sites, N))
    HhandR = reg_form_MPO(three_body_MPO(sites, N))
    Hauto = three_body_AutoMPO(sites)
    Dw_raw, Dw_auto = maxlinkdim(HhandL), maxlinkdim(Hauto)
    
    orthogonalize!(HhandL,left; verbose=verbose, atol=1e-14)
    orthogonalize!(HhandR,right; verbose=verbose, atol=1e-14)
    Dw1_L, Dw1_R = maxlinkdim(HhandL), maxlinkdim(HhandR)
    orthogonalize!(HhandL,right; verbose=verbose, atol=1e-14)
    orthogonalize!(HhandR,left; verbose=verbose, atol=1e-14)
    Dw2_L, Dw2_R = maxlinkdim(HhandL),  maxlinkdim(HhandR)

    @printf(
      "%4i  %4i  %4i   %4i  %4i  %4i  %4i \n",
      N,
      Dw_raw,
      Dw_auto,
      Dw1_L,
      Dw2_L,
      Dw1_R,
      Dw2_R,
    )
  end
end

@testset "Tabulate MPO Dw reduction for 3 body Hamiltonian" begin
  ul = lower
  initstate(n) = "↑"
  @printf("     |--------max(Dw)-----------|\n")
  @printf("          AutoMPO      hand built\n")
  @printf("  N      raw trunc    raw trunc  \n")
  for N in 3:15
    sites = siteinds("S=1/2", N)
    Hhand = reg_form_MPO(three_body_MPO(sites, N))
    Hauto = reg_form_MPO(three_body_AutoMPO(sites))
    
    Dw_hand_raw, Dw_auto_raw = maxlinkdim(Hhand), maxlinkdim(Hauto)
    ss_hand = truncate!(Hhand,left; cutoff=1e-15, atol=1e-15)
    ss_auto = truncate!(Hauto,left; cutoff=1e-15, atol=1e-15)
    
    Dw_hand_trunc1, Dw_auto_trunc = maxlinkdim(Hhand), maxlinkdim(Hauto)
    @printf(
      "%3i     %3i  %3i      %3i  %3i   \n",
      N,
      Dw_auto_raw,
      Dw_auto_trunc,
      Dw_hand_raw,
      Dw_hand_trunc1,
    )

    #
    #  Make sure the bond spectrum for auto, presummed amd dum summed hamiltonians
    #  are all identical to machine precision.
    #
    @test length(ss_hand) == length(ss_auto)
    for nb in eachindex(ss_hand)
      ss_h = eigs(ss_hand[nb])
      ss_a = eigs(ss_auto[nb])
      ha = .√(ss_h) - .√(ss_a)
      #@show sqrt(sum(ha.^2))/N
      @test sqrt(sum(ha .^ 2)) / N ≈ 0.0 atol = 3e-12
    end
  end
end
