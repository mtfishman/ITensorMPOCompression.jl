using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf
using Profile

using ITensorMPOCompression: orthogonalize!,truncate!

#brute force method to control the default float display format.
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

#
#  We need consistent output from randomMPS in order to avoid flakey unit testset
#  So we just pick any seed so that randomMPS (hopefull) gives us the same 
#  pseudo-random output for each test run.
#
using Random
Random.Random.seed!(12345);
ITensors.ITensors.disable_debug_checks() 

verbose=false #verbose at the outer test level
verbose1=false #verbose inside orth algos

function test_truncate(makeH,N::Int64,NNN::Int64,hx::Float64,ms::matrix_state,epsSVD::Float64,
    rr_cutoff::Float64,eps::Float64,qns::Bool)
    mlr=mirror(ms.lr)

    sites = siteinds("SpinHalf", N;conserve_qns=qns)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    #
    # Make right canonical, then compress to left canonical
    #
    H=makeH(sites,NNN;hx=hx,ul=ms.ul)
    E0l=inner(psi',H,psi)/(N-1)
    @test is_regular_form(H,ms.ul,eps)
    orthogonalize!(H;verbose=verbose1,orth=ms.lr,rr_cutoff=rr_cutoff)
    
    E1l=inner(psi',H,psi)/(N-1)
    RE=abs((E0l-E1l)/E0l)
    #@printf "E0=%1.5f E1=%1.5f rel. error=%.5e  \n" E0l E1l RE 
    @test RE ≈ 0 atol = 2*eps
    @test is_regular_form(H,ms.ul,eps)
    @test isortho(H,ms.lr)
    @test check_ortho(H,ms,eps)

    truncate!(H;verbose=verbose1,orth=mlr,cutoff=epsSVD)
    truncate!(H;verbose=verbose1,orth=ms.lr,cutoff=epsSVD)
    if rr_cutoff<0.0 #no rank reduction so do two more sweeps to make sure
        truncate!(H;verbose=verbose1,orth=mlr,cutoff=epsSVD)
        truncate!(H;verbose=verbose1,orth=ms.lr,cutoff=epsSVD)
    end
    @test is_regular_form(H,ms.ul,eps)
    @test isortho(H,ms.lr)
    @test check_ortho(H,ms,eps)
    # make sure the energy in unchanged
    E2l=inner(psi',H,psi)/(N-1)
    RE=abs((E0l-E2l)/E0l)
    REs= RE/sqrt(epsSVD)
    if verbose
        @printf "E0=%8.5f Etrunc=%8.5f rel. error=%7.1e RE/sqrt(espSVD)=%6.2f \n" E0l E2l RE REs
    end
    @test REs<1.0 
    
end

@testset verbose=true "Truncate/Compress" begin
# #these test are slow.  Uncomment and test if you mess with the truncate algo details.
# @testset "Compress insideous cases with no rank reduction, no QNs" begin
#     eps=2e-13
#     epsSVD=1e-12
#     rr_cutoff=-1.0
#     ll=matrix_state(lower,left)
#     ul=matrix_state(upper,left)
#     lr=matrix_state(lower,right)
#     ur=matrix_state(upper,right)
#     hx=0.5

#     #                                 V=N sites
#     #                                   V=Num Nearest Neighbours in H
#     for N in 4:15
#         test_truncate(make_transIsing_MPO,N,N,hx,ll,epsSVD,rr_cutoff,eps,false)
#         test_truncate(make_transIsing_MPO,N,N,hx,ul,epsSVD,rr_cutoff,eps,false)
#         test_truncate(make_transIsing_MPO,N,N,hx,lr,epsSVD,rr_cutoff,eps,false)
#         test_truncate(make_transIsing_MPO,N,N,hx,ur,epsSVD,rr_cutoff,eps,false)
#     end
# end 


@testset "Compress full MPO no QNs" begin
    eps=2e-13
    epsSVD=1e-12
    rr_cutoff=1e-12
    ll=matrix_state(lower,left)
    ul=matrix_state(upper,left)
    lr=matrix_state(lower,right)
    ur=matrix_state(upper,right)
    hx=0.5

    #                                 V=N sites
    #                                   V=Num Nearest Neighbours in H
    test_truncate(make_transIsing_MPO,5,1,hx,ll,epsSVD,rr_cutoff,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,ll,epsSVD,rr_cutoff,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,ul,epsSVD,rr_cutoff,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,lr,epsSVD,rr_cutoff,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,ur,epsSVD,rr_cutoff,eps,false)
    test_truncate(make_transIsing_MPO,15,10,hx,ll,epsSVD,rr_cutoff,eps,false) 
    test_truncate(make_transIsing_MPO,15,10,hx,lr,epsSVD,rr_cutoff,eps,false) 
    # Rank revealing QR/RQ  won't fully reduce these two cases
    test_truncate(make_transIsing_MPO,14,13,hx,ll,epsSVD,rr_cutoff,eps,false) 
    test_truncate(make_transIsing_MPO,14,13,hx,lr,epsSVD,rr_cutoff,eps,false) 

    # epsSVD=.00001
    test_truncate(make_transIsing_MPO,10,7,hx,ll,epsSVD,rr_cutoff,eps,false) 
    test_truncate(make_transIsing_MPO,10,7,hx,lr,epsSVD,rr_cutoff,eps,false) 
    test_truncate(make_transIsing_MPO,10,7,hx,ur,epsSVD,rr_cutoff,eps,false)
    test_truncate(make_transIsing_MPO,10,7,hx,ul,epsSVD,rr_cutoff,eps,false)

    #
    # Heisenberg from AutoMPO
    #
    # epsSVD=.00001
    test_truncate(make_Heisenberg_AutoMPO,10,7,hx,lr,epsSVD,rr_cutoff,eps,false)

end 

@testset "Compress full MPO with QNs" begin
    eps=2e-13
    epsSVD=1e-12
    rr_cutoff=1e-12
    hx=0.0
    ll=matrix_state(lower,left)
    ul=matrix_state(upper,left)
    lr=matrix_state(lower,right)
    ur=matrix_state(upper,right)

    #                                 V=N sites
    #                                   V=Num Nearest Neighbours in H
    test_truncate(make_transIsing_MPO,5,1,hx,ll,epsSVD,rr_cutoff,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,ll,epsSVD,rr_cutoff,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,ul,epsSVD,rr_cutoff,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,lr,epsSVD,rr_cutoff,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,ur,epsSVD,rr_cutoff,eps,true)
    test_truncate(make_transIsing_MPO,15,10,hx,ll,epsSVD,rr_cutoff,eps,true) 
    test_truncate(make_transIsing_MPO,15,10,hx,lr,epsSVD,rr_cutoff,eps,true) 
    # Rank revealing QR/RQ  won't fully reduce these two cases
    test_truncate(make_transIsing_MPO,14,13,hx,ll,epsSVD,rr_cutoff,eps,true) 
    test_truncate(make_transIsing_MPO,14,13,hx,lr,epsSVD,rr_cutoff,eps,true) 

    epsSVD=.00001
    test_truncate(make_transIsing_MPO,10,7,hx,ll,epsSVD,rr_cutoff,eps,true)  
    test_truncate(make_transIsing_MPO,10,7,hx,lr,epsSVD,rr_cutoff,eps,true) 
    test_truncate(make_transIsing_MPO,10,7,hx,ur,epsSVD,rr_cutoff,eps,true)
    test_truncate(make_transIsing_MPO,10,7,hx,ul,epsSVD,rr_cutoff,eps,true)  

    # #
    # # Heisenberg from AutoMPO
    # #
    # epsSVD=.00001
    # test_truncate(make_Heisenberg_AutoMPO,10,7,lr,epsSVD,rr_cutoff,eps,false)

end 

@testset "Test ground states" begin
    eps=3e-13
    epsSVD=1e-12
    rr_cutoff=1e-12
    N=10
    NNN=5
    hx=0.5

    sites = siteinds("SpinHalf", N)
    H=make_transIsing_MPO(sites,NNN;hx=hx)

    E0,psi0=fast_GS(H,sites)
    truncate!(H;verbose=verbose1,orth=left,cutoff=epsSVD,rr_cutoff=rr_cutoff)
    
    E1,psi1=fast_GS(H,sites)

    overlap=abs(inner(psi0,psi1))
    RE=abs((E0-E1)/E0)
    if verbose
        @printf "Trans. Ising E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
    end
    @test E0 ≈ E1 atol = eps
    @test overlap ≈ 1.0 atol = eps

    hx=0.0
    H=make_Heisenberg_AutoMPO(sites,NNN;hx=hx)
    E0,psi0=fast_GS(H,sites)

    truncate!(H;verbose=verbose1,orth=right,cutoff=epsSVD,rr_cutoff=rr_cutoff)

    E1,psi1=fast_GS(H,sites)
    overlap=abs(inner(psi0,psi1))
    RE=abs((E0-E1)/E0)
    if verbose
        @printf "Heisenberg   E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
    end
    @test E0 ≈ E1 atol = eps
    @test overlap ≈ 1.0 atol = eps
end 

@testset "Look at bond singular values for large lattices" begin
    NNN=5

    if verbose
        @printf "                 max(sv)           min(sv)\n"
        @printf "  N    Dw  left     mid    right   mid\n"
    end
    for N in [6,8,10,12,18,20,40,80]
        sites = siteinds("SpinHalf", N)
        H=make_transIsing_MPO(sites,NNN;hx=0.5)
        specs=truncate!(H;verbose=verbose1,cutoff=1e-10)
        imid::Int64=div(N,2)
        max_1  =specs[1   ].eigs[1]
        max_mid=specs[imid].eigs[1]
        max_N  =specs[N-1 ].eigs[1]
        min_mid=specs[imid].eigs[end]
        Dw=maximum(get_Dw(H))
        if verbose
            @printf "%4i %4i %1.5f %1.5f %1.5f %1.5f\n" N Dw  max_1 max_mid max_N min_mid
        end
        @test (max_1-max_N)<1e-10
        @test max_mid<1.0
    end
end

@testset "Head to Head autoMPO with 2-body Hamiltonian ul=$ul, QNs=$qns" for ul in [lower,upper],qns in [false,true]
    for N in 3:11
        NNN=div(N,2)
        svd_cutoff=1e-15 #same value auto MPO uses.
        rr_cutoff=1e-14
        sites = siteinds("SpinHalf", N;conserve_qns=qns)
        Hauto=make_transIsing_AutoMPO(sites,NNN;ul=ul) 
        Dw_auto=get_Dw(Hauto)
        Hr=make_transIsing_MPO(sites,NNN;ul=ul) 
        truncate!(Hr;verbose=verbose1,rr_cutoff=rr_cutoff,cutoff=svd_cutoff) #sweep left to right
        delta_Dw=sum(get_Dw(Hr)-Dw_auto)
        #@test delta_Dw<=0
        if verbose && delta_Dw<0 
            println("Compression beat AutoMPO by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
        end
        if verbose && delta_Dw>0
            println("AutoMPO beat Compression by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
        end
    end  
end 

# this is the slooooow test
# @testset "Head to Head autoMPO with 3-body Hamiltonian" begin
#     if verbose
#         @printf "+--------------+---------+--------------------+--------------------+\n"
#         @printf "|              |autoMPO  |    1 truncation    | 2 truncations      |\n"
#         @printf "|  N  epseSVD  | dE      |   dE     RE   Dw   |   dE     RE   Dw   |\n"
#     end

#     for svd_cutoff in [1e-15,1e-12,1e-10]
#         for N in [6,10,16,20,24]
#             sites = siteinds("SpinHalf", N;conserve_qns=false)
#             Hnot=make_3body_AutoMPO(sites;cutoff=-1.0) #No truncation inside autoMPO
#             H=make_3body_AutoMPO(sites) #Truncated by autoMPO
#             #@show get_Dw(Hnot)
#             Dw_auto = get_Dw(H)
#             psi=randomMPS(sites)
#             Enot=inner(psi',Hnot,psi)
#             E=inner(psi',H,psi)
#             @test E ≈ Enot atol = sqrt(svd_cutoff)
#             truncate!(Hnot;verbose=verbose1,rr_cutoff=1e-14,cutoff=svd_cutoff)
#             Dw_1=get_Dw(Hnot)
#             delta_Dw_1=sum(Dw_auto-Dw_1)
#             Enott1=inner(psi',Hnot,psi)
#             @test E ≈ Enott1 atol = sqrt(svd_cutoff)
#             RE1=abs(E-Enott1)/sqrt(svd_cutoff)

#             truncate!(Hnot;verbose=verbose1,cutoff=svd_cutoff)
#             Dw_2=get_Dw(Hnot)
#             delta_Dw_2=sum(Dw_auto-Dw_2)
#             Enott2=inner(psi',Hnot,psi)
#             @test E ≈ Enott2 atol = sqrt(svd_cutoff)
#             RE2=abs(E-Enott2)/sqrt(svd_cutoff)
#             if verbose
#                 @printf "| %3i %1.1e  | %1.1e | %1.1e %1.3f %4i | %1.1e %1.3f %4i | \n" N svd_cutoff abs(E-Enot) abs(E-Enott1) RE1 delta_Dw_1 abs(E-Enott2) RE2 delta_Dw_2 
#             end
#         end
#     end
# end


@testset "Truncate/Compress iMPO Check gauge relations, ul=$ul, qbs=$qns" for ul in [lower,upper], qns in [false,true]
    initstate(n) = "↑"
    svd_cutoff=1e-15 #Same as autoMPO uses.
    if verbose
        @printf "               Dw     Dw    Dw   \n"
        @printf " Ncell  NNN  uncomp. left  right \n"
    end

    for  NNN in [2,4], N in [1,2,4] #3 site unit cell fails inside ITensorInfiniteMPS for qns=true.
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=qns)
        H0=make_transIsing_iMPO(si,NNN;ul=ul)
        @test is_regular_form(H0)
        Dw0=Base.max(get_Dw(H0)...)
        #
        #  Do truncate outputting left ortho Hamiltonian
        #
        HL=copy(H0)
        Ss,ss,HR=truncate!(HL;verbose=verbose1,orth=left,cutoff=svd_cutoff,h_mirror=true)
        @test typeof(storage(Ss[1])) == (qns ? BlockSparse{Float64, Vector{Float64}, 2} : Diag{Float64, Vector{Float64}})
        DwL=Base.max(get_Dw(HL)...)
        @test is_regular_form(HL)
        @test isortho(HL,left)
        @test check_ortho(HL,left)
        #
        #  Now test guage relations using the diagonal singular value matrices
        #  as the gauge transforms.
        #
        for n in 1:N
            @test norm(Ss[n-1]*HR[n]-HL[n]*Ss[n]) ≈ 0.0 atol = 1e-14
        end    
        #
        #  Do truncate from H0 outputting right ortho Hamiltonian
        #
        HR=copy(H0)
        Ss,ss,HL=truncate!(HR;verbose=verbose1,orth=right,cutoff=svd_cutoff,h_mirror=true)
        @test typeof(storage(Ss[1])) == (qns ? BlockSparse{Float64, Vector{Float64}, 2} : Diag{Float64, Vector{Float64}})
        DwR=Base.max(get_Dw(HR)...)
        @test is_regular_form(HR)
        @test isortho(HR,right)
        @test check_ortho(HR,right)
        for n in 1:N
            @test norm(Ss[n-1]*HR[n]-HL[n]*Ss[n]) ≈ 0.0 atol = 1e-14
        end   
        if verbose
            @printf " %4i %4i   %4i   %4i  %4i \n" N NNN Dw0 DwL DwR
        end

    end
end

@testset "Orthogonalize/truncate verify gauge invariace of <ψ|H|ψ>, ul=$ul, qbs=$qns" for ul in [lower,upper], qns in [false,true]
    initstate(n) = "↑"
    for N in [1], NNN in [2,4] #3 site unit cell fails for qns=true.
        svd_cutoff=1e-15 #Same as autoMPO uses.
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=qns)
        ψ = InfMPS(si, initstate)
        for n in 1:N
            ψ[n] = randomITensor(inds(ψ[n]))
        end
        H0=make_transIsing_iMPO(si,NNN;ul=ul)
        Hsum0=InfiniteSum{MPO}(H0,NNN)
        E0=expect(ψ,Hsum0)

        HL=copy(H0)
        orthogonalize!(HL;verbose=verbose1,orth=left)
        HsumL=InfiniteSum{MPO}(HL,NNN)
        EL=expect(ψ,HsumL)
        @test EL ≈ E0 atol = 1e-14

        HR=copy(HL)
        orthogonalize!(HR;verbose=verbose1,orth=right)
        HsumR=InfiniteSum{MPO}(HR,NNN)
        ER=expect(ψ,HsumR)
        @test ER ≈ E0 atol = 1e-14

        HL=copy(H0)
        truncate!(HL;verbose=verbose1,orth=left,cutoff=svd_cutoff)
        HsumL=InfiniteSum{MPO}(HL,NNN)
        EL=expect(ψ,HsumL)
        @test EL ≈ E0 atol = 1e-14

        HR=copy(H0)
        truncate!(HR;verbose=verbose1,orth=right,cutoff=svd_cutoff)
        HsumR=InfiniteSum{MPO}(HR,NNN)
        ER=expect(ψ,HsumR)
        @test ER ≈ E0 atol = 1e-14
        # truncate!(HR;verbose=verbose1,orth=left,cutoff=svd_cutoff)
        # HsumR=InfiniteSum{MPO}(HR,NNN)
        # ER=expect(ψ,HsumR)
        # @test ER ≈ E0 atol = 1e-14

        #@show get_Dw(H0) get_Dw(HL) get_Dw(HR)
    end
end

end #@testset "Truncate/Compress" 
nothing
