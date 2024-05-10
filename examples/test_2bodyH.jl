using ITensors, ITensorMPS
using ITensorMPOCompression
using Test

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output
include("../test/hamiltonians/hamiltonians.jl")

@testset "Investigate suprise effects of the inital sweep direction" begin
  ul = lower
  initstate(n) = "↑"
  verbose = false
  @printf("               max(Dw) dumb-summed \n")
  @printf("   N  NNN   Raw   L1    L2    R1    R2\n")

  for NNN in [1, 5, 8, 12, 14, 20, 25]
    # N needs to be big enough that there is block in the middle of lattice which 
    # exhibits no edge effects.
    N = 2 * NNN + 4
    sites = siteinds("S=1/2", N)
    HdumbL = two_body_MPO(sites, NNN; Jprime=1.0, presummed=false)
    HdumbR = deepcopy(HdumbL)
    Dw_raw=maxlinkdim(HdumbL)
    orthogonalize!(HdumbL, left; verbose=verbose, atol=1e-14)
    orthogonalize!(HdumbR, right; verbose=verbose, atol=1e-14)
    Dw1_L, Dw1_R = maxlinkdim(HdumbL), maxlinkdim(HdumbR)
    orthogonalize!(HdumbL, right; verbose=verbose, atol=1e-14)
    orthogonalize!(HdumbR, left; verbose=verbose, atol=1e-14)
    Dw2_L, Dw2_R = maxlinkdim(HdumbL), maxlinkdim(HdumbR)
    @printf("%4i %4i   %4i  %4i  %4i  %4i  %4i \n", N, NNN, Dw_raw, Dw1_L, Dw2_L, Dw1_R, Dw2_R)
  end
end

@testset "Tabulate MPO Dw reduction for 2 body NNN interactions" begin
  ul=lower
  initstate(n) = "↑"
  @printf("             |--------------------------max(Dw)-------------------------|\n")
  @printf("             AutoMPO         pre-summed        dumb-summed  \n")
  @printf("              Finite           Finite            finite     \n")
  @printf("  N  NNN  raw orth trunc   raw orth trunc     raw orth trunc\n")
  for NNN in 1:15
      # N needs to be big enough that there is block in the middle of lattice which 
      # exhibits no edge effects.
      N=2*NNN+4 
      #Nmid=div(N,2)
      #NNNd= NNN>15 ? 1 : NNN #don't bother with the dumb version for largeish NNN
      sites = siteinds("S=1/2",N;conserve_qns=false);
      Hpres=two_body_MPO(sites,NNN;presummed=true)
      Hdumb=two_body_MPO(sites,NNN;presummed=false)
      Hauto=two_body_AutoMPO(sites,NNN;nexp=4)
      Dw_ps_raw,Dw_ds_raw,Dw_auto_raw=maxlinkdim(Hpres),maxlinkdim(Hdumb),maxlinkdim(Hauto)
      orthogonalize!(Hpres,left;atol=1e-15)
      orthogonalize!(Hpres,right;atol=1e-15)
      orthogonalize!(Hdumb,left;atol=1e-15)
      orthogonalize!(Hdumb,right;atol=1e-15)
      orthogonalize!(Hauto,left;atol=1e-15)
      orthogonalize!(Hauto,right;atol=1e-15)
      Dw_ps_orth,Dw_ds_orth,Dw_auto_orth=maxlinkdim(Hpres),maxlinkdim(Hdumb),maxlinkdim(Hauto)
      ss_pres=truncate!(Hpres,left;cutoff=1e-15)
      ss_dumb=truncate!(Hdumb,left;cutoff=1e-15)
      ss_auto=truncate!(Hauto,left;cutoff=1e-15)
      Dw_ps_trunc,Dw_ds_trunc,Dw_auto_trunc=maxlinkdim(Hpres),maxlinkdim(Hdumb),maxlinkdim(Hauto)
  
      @printf("%3i %3i  %3i  %3i  %3i    %3i  %3i  %3i        %3i  %3i  %3i\n",
      N,NNN,
      Dw_auto_raw,Dw_auto_orth,Dw_auto_trunc,
      Dw_ps_raw,Dw_ps_orth,Dw_ps_trunc,
      Dw_ds_raw,Dw_ds_orth,Dw_ds_trunc,
      )

    #
    #  Make sure the bond spectrum for auto, presummed amd dum summed hamiltonians
    #  are all identical to machine precision.
    #
    @test length(ss_pres)==length(ss_auto)
    @test length(ss_dumb)==length(ss_auto)
    for nb in eachindex(ss_pres)
      ss_p=eigs(ss_pres[nb])
      ss_d=eigs(ss_dumb[nb])
      ss_a=eigs(ss_auto[nb])
      if !isnothing(ss_d)
        pd=.√(ss_p)-.√(ss_d)
        pa=.√(ss_p)-.√(ss_a)
        #@show sqrt(sum(pa.^2))/N
        @test sqrt(sum(pd.^2))/N  ≈ 0.0 atol = 1e-14*N
        @test sqrt(sum(pa.^2))/N  ≈ 0.0 atol = 4e-12*N
      end
    end
  end
end
