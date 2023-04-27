using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS

using Printf
using Test
Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output

@testset "Investigate suprise effects of the inital sweep direction" begin
  ul = lower
  initstate(n) = "↑"
  verbose = false
  @printf("                         max(Dw) dumb-summed \n")
  @printf("                    Finite                      Infinite \n")
  @printf("   N  NNN     L1    L2    R1    R2        L1    L2    R1    R2  \n")

  for NNN in [1, 5, 8, 12, 14]
    # N needs to be big enough that there is block in the middle of lattice which 
    # exhibits no edge effects.
    N = 2 * NNN + 4
    Nmid = div(N, 2)
    NNNd = NNN > 15 ? 1 : NNN #don't bother with the dumb version for largeish NNN
    sites = siteinds("S=1/2", N)
    si = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
    HdumbL = two_body_MPO(sites, NNNd; Jprime=1.0, presummed=false)
    @show get_Dw(HdumbL)
    #Hdumb_pbc=two_body_MPO(si,NNN;Jprime=1.0,presummed=false,pbc=true)
    #Hdumb_infL=InfiniteMPO([Hdumb_pbc[1]])
    #@show inds(Hdumb_infL[1])
    #@mpoc_assert false
    #Dw_raw=maxlinkdim(Hdumb_infL)
    HdumbR = deepcopy(HdumbL)
    #Hdumb_infR=copy(Hdumb_infL)

    HdumbL = ac_orthogonalize!(HdumbL, left; verbose=verbose, cutoff=1e-14)
    HdumbR = ac_orthogonalize!(HdumbR, right; verbose=verbose, cutoff=1e-14)
    # orthogonalize!(Hdumb_infL;verbose=verbose,orth=left,max_sweeps=1,rr_cutoff=1e-14)
    # orthogonalize!(Hdumb_infR;verbose=verbose,orth=right,max_sweeps=1,rr_cutoff=1e-14)
    @show get_Dw(HdumbL)
    Dw1_L, Dw1_R = maxlinkdim(HdumbL), maxlinkdim(HdumbR)
    # #Dw1i_L,Dw1i_R=maxlinkdim(Hdumb_infL),maxlinkdim(Hdumb_infR)
    HdumbL = ac_orthogonalize!(HdumbL, right; verbose=verbose, cutoff=1e-14)
    HdumbR = ac_orthogonalize!(HdumbR, left; verbose=verbose, cutoff=1e-14)
    # #ac_orthogonalize!(Hdumb_infL;verbose=verbose,orth=right,max_sweeps=1,rr_cutoff=1e-14)
    # #ac_orthogonalize!(Hdumb_infR;verbose=verbose,orth=left,max_sweeps=1,rr_cutoff=1e-14)
    @show get_Dw(HdumbL)
    Dw2_L, Dw2_R = maxlinkdim(HdumbL), maxlinkdim(HdumbR)
    # #Dw2i_L,Dw2i_R=maxlinkdim(Hdumb_infL),maxlinkdim(Hdumb_infR)
    @printf("%4i %4i   %4i  %4i  %4i  %4i \n", N, NNN, Dw1_L, Dw2_L, Dw1_R, Dw2_R)
    # @printf("%4i %4i   %4i  %4i  %4i  %4i      %4i  %4i  %4i  %4i \n",N,NNN,
    # Dw1_L,Dw2_L,Dw1_R,Dw2_R,Dw1i_L,Dw2i_L,Dw1i_R,Dw2i_R)
  end
end

# @testset "Tabulate MPO Dw reduction for 2 body NNN interactions" begin
#     ul=lower
#     initstate(n) = "↑"
#     @printf("             |--------------------------max(Dw)-------------------------|\n")
#     @printf("             AutoMPO               pre-summed                     dumb-summed \n")
#     @printf("              Finite           Finite      Infinite          finite       Infinte \n")
#     @printf("  N  NNN  raw orth trunc   raw orth trunc  orth trunc    raw orth trunc  orth trunc\n")
#     for NNN in 1:5
#         # N needs to be big enough that there is block in the middle of lattice which 
#         # exhibits no edge effects.
#         N=2*NNN+4 
#         #Nmid=div(N,2)
#         NNNd= NNN>15 ? 1 : NNN #don't bother with the dumb version for largeish NNN
#         sites = siteinds("S=1/2",N);
#         si = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
#         Hpres=two_body_MPO(sites,NNN;presummed=true)
#         Hdumb=two_body_MPO(sites,NNNd;presummed=false)
#         Hauto=two_body_AutoMPO(sites,NNN)
#         Hpres_pbc=two_body_MPO(si,NNN;presummed=true,pbc=true)
#         Hdumb_pbc=two_body_MPO(si,NNN;presummed=false,pbc=true)
#         # Hpres_inf=InfiniteMPO([Hpres_pbc[1]])
#         # Hdumb_inf=InfiniteMPO([Hdumb_pbc[1]])
#     #     Dw_ps_raw,Dw_ds_raw,Dw_auto_raw=maxlinkdim(Hpres),maxlinkdim(Hdumb),maxlinkdim(Hauto)
#     #     orthogonalize!(Hpres;rr_cutoff=1e-14)
#     #     orthogonalize!(Hdumb;rr_cutoff=1e-14)
#     #     orthogonalize!(Hauto;rr_cutoff=1e-14)
#     #     # Gs_pres=orthogonalize!(Hpres_inf;rr_cutoff=1e-14)
#     #     # Gs_dumb=orthogonalize!(Hdumb_inf;rr_cutoff=1e-14)
#     #     Dw_ps_orth,Dw_ds_orth,Dw_auto_orth=maxlinkdim(Hpres),maxlinkdim(Hdumb),maxlinkdim(Hauto)
#     #     # Dw_psinf_orth,Dw_dsinf_orth=maxlinkdim(Hpres_inf),maxlinkdim(Hdumb_inf)
#     #     ss_pres=truncate!(Hpres;cutoff=1e-15,rr_cutoff=1e-15)
#     #     ss_dumb=truncate!(Hdumb;cutoff=1e-15,rr_cutoff=1e-15)
#     #     ss_auto=truncate!(Hauto;cutoff=1e-15,rr_cutoff=1e-15)
#     #     truncate!(Hpres_inf,Gs_pres,left;cutoff=1e-15,rr_cutoff=1e-15)
#     #     truncate!(Hdumb_inf,Gs_dumb,left;cutoff=1e-15,rr_cutoff=1e-15)
#     #     Dw_ps_trunc,Dw_ds_trunc,Dw_auto_trunc=maxlinkdim(Hpres),maxlinkdim(Hdumb),maxlinkdim(Hauto)
#     #     Dw_psinf_trunc,Dw_dsinf_trunc=maxlinkdim(Hpres_inf),maxlinkdim(Hdumb_inf)

#     #     @printf("%3i %3i  %3i  %3i  %3i    %3i  %3i  %3i    %3i  %3i      %3i  %3i  %3i   %3i  %3i \n",
#     #     N,NNN,
#     #     Dw_auto_raw,Dw_auto_orth,Dw_auto_trunc,
#     #     Dw_ps_raw,Dw_ps_orth,Dw_ps_trunc,Dw_psinf_orth,Dw_psinf_trunc,
#     #     Dw_ds_raw,Dw_ds_orth,Dw_ds_trunc,Dw_dsinf_orth,Dw_dsinf_trunc,
#     #     )

#     #     #
#     #     #  Make sure the bond spectrum for auto, presummed amd dum summed hamiltonians
#     #     #  are all identical to machine precision.
#     #     #
#     #     @test length(ss_pres)==length(ss_auto)
#     #     @test length(ss_dumb)==length(ss_auto)
#     #     for nb in 1:length(ss_pres)
#     #         ss_p=eigs(ss_pres[nb])
#     #         ss_d=eigs(ss_dumb[nb])
#     #         ss_a=eigs(ss_auto[nb])
#     #         pd=.√(ss_p)-.√(ss_d)
#     #         pa=.√(ss_p)-.√(ss_a)
#     #         #@show sqrt(sum(pd.^2))/N
#     #         @test sqrt(sum(pd.^2))/N  ≈ 0.0 atol = 1e-12
#     #         @test sqrt(sum(pa.^2))/N  ≈ 0.0 atol = 1e-12
#     #      end
#     end
# end
