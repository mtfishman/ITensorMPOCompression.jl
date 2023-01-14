using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test

Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output

@testset "Investigate suprise effects of the inital sweep direction for 3 body Hamiltonian" begin
    ul=lower
    initstate(n) = "↑"
    verbose=false
    @printf("                               max(Dw) dumb-summed \n")
    @printf("                        Finite                      Infinite \n")
    @printf("   N  Raw autoMPO    L1    L2    R1    R2        L1    L2    R1    R2  \n")

    for N in 3:10
        # N needs to be big enough that there is block in the middle of lattice which 
        # exhibits no edge effects.
        sites = siteinds("S=1/2",N);
        si = infsiteinds("S=1/2",1; initstate, conserve_szparity=false)
        HhandL=make_3body_MPO(sites,N)
        Hhand_pbcL=make_3body_MPO(si,N;pbc=true)
        Hauto=make_3body_AutoMPO(sites)
        Hhand_infL=InfiniteMPO([Hhand_pbcL[1]])
        Dw_raw,Dw_auto=max_Dw(Hhand_infL),max_Dw(Hauto)
        HhandR=copy(HhandL)
        Hhand_infR=copy(Hhand_infL)

        orthogonalize!(HhandL;verbose=verbose,orth=left,max_sweeps=1,rr_cutoff=1e-14)
        orthogonalize!(HhandR;verbose=verbose,orth=right,max_sweeps=1,rr_cutoff=1e-14)
        orthogonalize!(Hhand_infL;verbose=verbose,orth=left,max_sweeps=1,rr_cutoff=1e-14)
        orthogonalize!(Hhand_infR;verbose=verbose,orth=right,max_sweeps=1,rr_cutoff=1e-14)
        Dw1_L,Dw1_R,Dw1i_L,Dw1i_R=max_Dw(HhandL),max_Dw(HhandR),max_Dw(Hhand_infL),max_Dw(Hhand_infR)
        orthogonalize!(HhandL;verbose=verbose,orth=right,max_sweeps=1,rr_cutoff=1e-14)
        orthogonalize!(HhandR;verbose=verbose,orth=left,max_sweeps=1,rr_cutoff=1e-14)
        orthogonalize!(Hhand_infL;verbose=verbose,orth=right,max_sweeps=1,rr_cutoff=1e-14)
        orthogonalize!(Hhand_infR;verbose=verbose,orth=left,max_sweeps=1,rr_cutoff=1e-14)
        Dw2_L,Dw2_R,Dw2i_L,Dw2i_R=max_Dw(HhandL),max_Dw(HhandR),max_Dw(Hhand_infL),max_Dw(Hhand_infR)

        @printf("%4i  %4i  %4i   %4i  %4i  %4i  %4i      %4i  %4i  %4i  %4i \n",N,Dw_raw,Dw_auto,
        Dw1_L,Dw2_L,Dw1_R,Dw2_R,Dw1i_L,Dw2i_L,Dw1i_R,Dw2i_R)
    end
end

@testset "Tabulate MPO Dw reduction for 3 body Hamiltonian" begin
    ul=lower
    initstate(n) = "↑"
    @printf("     |--------------------------max(Dw)-------------------------|\n")
    @printf("          AutoMPO               hand built\n")
    @printf("          Finite               Finite              Infinite\n")
    @printf("  N    raw orth trunc   raw orth tr1 tr2 tr3   orth trunc\n")
    for N in 3:10
        sites = siteinds("S=1/2",N);
        si = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
        Hhand=make_3body_MPO(sites,N)
        Hauto=make_3body_AutoMPO(sites)
        Hhand_pbc=make_3body_MPO(si,N;pbc=true)
        Hhand_inf=InfiniteMPO([Hhand_pbc[1]])
        Dw_hand_raw,Dw_auto_raw=max_Dw(Hhand),max_Dw(Hauto)
        orthogonalize!(Hhand;rr_cutoff=1e-14)
        orthogonalize!(Hauto;rr_cutoff=1e-14)
        Gs_hand=orthogonalize!(Hhand_inf;rr_cutoff=1e-14)
        Dw_hand_orth,Dw_auto_orth=max_Dw(Hhand),max_Dw(Hauto)
        Dw_handinf_orth=max_Dw(Hhand_inf)
        ss_hand=truncate!(Hhand;cutoff=1e-15,rr_cutoff=1e-15)
        ss_auto=truncate!(Hauto;cutoff=1e-15,rr_cutoff=1e-15)
        truncate!(Hhand_inf,Gs_hand,left;cutoff=1e-15,rr_cutoff=1e-15)
        Dw_hand_trunc1,Dw_auto_trunc=max_Dw(Hhand),max_Dw(Hauto)
        Dw_handinf_trunc=max_Dw(Hhand_inf)
        ss_hand=truncate!(Hhand;cutoff=1e-15,rr_cutoff=1e-15)
        Dw_hand_trunc2=max_Dw(Hhand)   
        ss_hand=truncate!(Hhand;cutoff=1e-15,rr_cutoff=1e-15)
        Dw_hand_trunc3=max_Dw(Hhand)   
        @printf("%3i   %3i  %3i  %3i    %3i  %3i  %3i  %3i  %3i  %3i  %3i     \n",
        N,
        Dw_auto_raw,Dw_auto_orth,Dw_auto_trunc,
        Dw_hand_raw,Dw_hand_orth,Dw_hand_trunc1,Dw_hand_trunc2,Dw_hand_trunc3,Dw_handinf_orth,Dw_handinf_trunc,
        )
        
        #
        #  Make sure the bond spectrum for auto, presummed amd dum summed hamiltonians
        #  are all identical to machine precision.
        #
        @test length(ss_hand)==length(ss_auto)
        for nb in 1:length(ss_hand)
            ss_h=eigs(ss_hand[nb])
            ss_a=eigs(ss_auto[nb])
            ha=.√(ss_h)-.√(ss_a)
            #@show sqrt(sum(pd.^2))/N
            @test sqrt(sum(ha.^2))/N  ≈ 0.0 atol = 1e-12
         end
    end
end

# @testset "Test if autoMPO 3 body Hamiltonians can be further compressed" begin
#     ul=lower
#     initstate(n) = "↑"
#     @printf("     |-----max(Dw)-----|\n")
#     @printf("          AutoMPO\n")
#     @printf("          Finite\n")
#     @printf("  N    raw orth trunc\n")
#     for N in 3:30
#         sites = siteinds("S=1/2",N);
#         si = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
#         Hauto=make_3body_AutoMPO(sites)
#         Dw_auto_raw=max_Dw(Hauto)
#         orthogonalize!(Hauto;rr_cutoff=1e-14)
#         Dw_auto_orth=max_Dw(Hauto)
#         ss_auto=truncate!(Hauto;cutoff=1e-15,rr_cutoff=1e-15)
#         Dw_auto_trunc=max_Dw(Hauto)
#         mins=ITensorMPOCompression.min(ss_auto)
        
#         @printf("%3i   %3i  %3i  %3i  %1.2e\n", N,  Dw_auto_raw,Dw_auto_orth,Dw_auto_trunc,mins)
    
#     end
# end

# @testset "Verify auto MPO and hand built 3-body Hamiltonians as identical" begin
#     #l,r=Index(1,"Link,c=0,l=1"),Index(1,"Link,c=1,l=1")
#     l,r=Index(1,"Link,l=0"),Index(1,"Link,l=1")
#     ul=lower
#     for N in 2:8
#         sites = siteinds("S=1/2",N);
#         Hhand=make_3body_MPO(sites,N;J=1.0,Jprime=1.0)
#         Hauto=make_3body_AutoMPO(sites;J=1.0,Jprime=1.0)
#         println("N=$N, Hand Dw=$(max_Dw(Hhand))")
#         #orthogonalize!(Hhand)
#         truncate!(Hhand;cutoff=1e-15,rr_cutoff=1e-15)
#         ss_hand=truncate!(Hhand;orth=right,cutoff=1e-15,rr_cutoff=1e-15)
#         truncate!(Hauto;cutoff=1e-15,rr_cutoff=1e-15)
#         ss_auto=truncate!(Hauto;orth=right,cutoff=1e-15,rr_cutoff=1e-15)
#         @test length(ss_hand)==length(ss_auto)
#         for nb in 1:length(ss_hand)
#             sh=eigs(ss_hand[nb])
#             sa=eigs(ss_auto[nb])
#             ds=.√(sh)-.√(sa)
#             @show sqrt(sum(ds.^2))/N
#             @test sqrt(sum(ds.^2))/N  ≈ 0.0 atol = 1e-14
#         end
#     end
# end

# function my_MPO(sites,NNN::Int64;kwargs...)
#     pbc::Bool=get(kwargs,:pbc,false)
#     N=length(sites)
#     H=MPO(N)
#     if false#pbc
#         l,r=Index(1,"Link,c=0,l=1"),Index(1,"Link,c=1,l=1")
#     else
#         l,r=Index(1,"Link,l=0"),Index(1,"Link,l=1")
#     end
#     ul=lower
#     H[1] = make_1body_op(sites[1],l,r,ul;kwargs...)
    
#     for n in 2:N
#         H[n] = make_1body_op(sites[n],l,r,ul;kwargs...)
#     end
    
#     W=H[1]
#     if get(kwargs,:Jprime,1.0)!=0.0
#         for m in 1:NNN
#             W = add_ops(W,make_2body_op(sites[1],l,r,m,ul;kwargs...))
#         end
#     end 
    
#     if get(kwargs,:J,1.0)!=0.0
#         for n in 2:NNN
#             for m in NNN+1:NNN+1
#                 #@show n m 
#                 Wnm=make_3body_op(sites[1],l,r,n,m,ul;kwargs...)
#                 @pprint(Wnm)
#                 W=add_ops(W,Wnm)
#             end
#         end
#     end

#     H[1]=add_ops(H[1],W)
    
#     for n in 2:N
#         #@show inds(H[n])
#         iw1,iw2=inds(W,tags="Link")
#         ih1,ih2=inds(H[n],tags="Link")
#         replacetags!(W,tags(iw2),tags(ih2))
#         replacetags!(W,tags(iw1),tags(ih1))
#         is=sites[n]
#         replaceinds!(W,inds(W,tags="Site"),(is,dag(is)'))
#         H[n]=add_ops(H[n],W)
#     end
    
#     for n in 2:N
#         ln=inds(H[n],tags="l=0")[1]
#         replacetags!(H[n],"l=1","l=$n")
#         l1,r1=parse_links(H[n-1])
#         replaceind!(H[n],ln,r1)
#     end

#     if !pbc
#         H=ITensorMPOCompression.to_openbc(H) #contract with l* and *r at the edges.
#     end
    
#     return H
# end

# function findJs(sites,n::Int64,W::ITensor)
#     Sz=op(sites[n],"Sz")
#     Id=op(sites[n],"Id")
#     r,c=parse_links(W)
#     nr,nc=dim(r),dim(c)
#     for ir in 2:nr
#         for ic in 1:nc-1
#             op=slice(W,r=>ir,c=>ic)
#             ss=(Sz*op)[]
#             si=(Id*op)[]
#             if abs(ss)>1e-14
#                 println("S($ir,$ic)=$(ss*2), s^2=$(4*ss*ss)")
#             elseif abs(si)>1e-14
#                 println("I($ir,$ic)=$(si)")
#             else
#                 ns=norm(op)
#                 if ns>1e-14
#                 @show ir,ic,slice(W,r=>ir,c=>ic)
#                 end
#             end
        
#         end
#     end
# end


# Base.show(io::IO, f::Float64) = @printf(io, "%1.5f", f) #dumb way to control float output

# function make_3body(sites,NNN::Int64;kwargs...)::MPO
#     N=length(sites)
#     os = OpSum()
#     for k in 1:N-NNN
#     m=k+NNN
#     J::Float64=get(kwargs,:J,1.0)
#     for n=k+1:m-1
#         Jkn=J/abs(n-k)^1
        
#         Jnm=J/abs(m-n)^1
#         # if k==1
#         #@show k,n,m,Jkn,Jnm
#         # end
#         add!(os, Jnm*Jkn    ,"Sz", k, "Sz", n,"Sz",m)
    
#     end
#     end
    
#     return MPO(os,sites;kwargs...)
# end

# function swap(A::ITensor,r1::Int64,r2::Int64)::ITensor
#     iar,iac=parse_links(A)
#     B=copy(A)
#     ibr,ibc=parse_links(B)
#     for jv in eachindval(iac)
#         op=slice(A,iar=>r1,jv)
#         assign!(B,op,ibr=>r2,ibc=>jv.second)
#         op=slice(A,iar=>r2,jv)
#         assign!(B,op,ibr=>r1,ibc=>jv.second)
#     end
#     #@pprint(B)
#     C=copy(B)
#     for iv  in eachindval(iar)
#         op=slice(B,iv,iac=>r1)
#         assign!(C,op,ibr=>iv.second,ibc=>r2)
#         op=slice(B,iv,iac=>r2)
#         assign!(C,op,ibr=>iv.second,ibc=>r1)
#     end
#     return C
# end

# @testset "Deduce presummed 3 body MPO using AutoMPO" begin
#     NNN=4
#     sites = siteinds("S=1/2",15);
#     H=make_3body(sites,NNN;J=1.0,cutoff=1e-15)
#     @show get_Dw(H)
#     W=H[5]
#     @pprint(W)
#     #swaps for NNN=3
#     # W1=swap(W,2,5)
#     # @pprint(W1)
#     # W2=swap(W1,2,3)
#     # @pprint(W2)
#     # swaps for NNN=4
#     # When J=1 (2,6) (2,3) (6,7)
#     W1=swap(W,2,6)
#     @pprint(W1)
#     #@pprint(swap(W1,2,3))
#     # @pprint(swap(W1,2,4))
#     # @pprint(swap(W1,2,5))
#     # W2=swap(W1,2,3)
#     # @pprint(W2)
#     # W3=swap(W2,6,7)
#     # @pprint(W3)
#     # findJs(sites,5,W3)
#     @show truncate!(H)
#     @show get_Dw(H)
#     @pprint(H[7])
# #    findJs(sites,5,H[5])
# end


# sites = siteinds("S=1/2",21);
# mpo=my_MPO(sites,4;Jprime=0.0,J=1.0)
# pprint(mpo[10])
# findJs(sites,mpo)
# @show get_Dw(mpo)
# orthogonalize!(mpo;rr_cutoff=1e-12,verbose=true)
# # orthogonalize!(mpo;max_sweeps=1,orth=right,rr_cutoff=1e-12,verbose=true)
# # orthogonalize!(mpo;max_sweeps=1,orth=left,rr_cutoff=1e-12,verbose=true)
# # orthogonalize!(mpo;max_sweeps=1,orth=right,rr_cutoff=1e-12,verbose=true)
# ss=truncate!(mpo;cutoff=1e-15,rr_cutoff=1e-12,verbose=true)
# @show get_Dw(mpo) 
# pprint(mpo[10],1e-14)
# findJs(sites,mpo)

# l,r=Index(1,"Link,l=0"),Index(1,"Link,l=1")
# W=make_2body_sum(sites[1],r,c,4,lower)
# pprint(W)
# findJs(sites,W)
nothing