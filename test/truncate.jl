using ITensors
using ITensorMPOCompression: truncate!, orthogonalize!
using Revise
using Test
using Printf

#brute force method to control the default float display format.
Base.show(io::IO, f::Float64) = @printf(io, "%1.1e", f)

#
#  We need consistent output from randomMPS in order to avoid flakey unit testset
#  So we just pick any seed so that randomMPS (hopefull) gives us the same 
#  pseudo-random output for each test run.
#
using Random
Random.Random.seed!(12345);

function test_truncate(makeH,N::Int64,NNN::Int64,hx::Float64,ms::matrix_state,epsSVD::Float64,
    epsrr::Float64,eps::Float64,qns::Bool)
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
    orthogonalize!(H;orth=ms.lr,epsrr=epsrr)
    
    E1l=inner(psi',H,psi)/(N-1)
    RE=abs((E0l-E1l)/E0l)
    #@printf "E0=%1.5f E1=%1.5f rel. error=%.5e  \n" E0l E1l RE 
    @test RE ≈ 0 atol = 2*eps
    @test is_regular_form(H,ms.ul,eps)
    @test is_canonical(H,ms,eps)

    truncate!(H;orth=mlr,cutoff=epsSVD)
    truncate!(H;orth=ms.lr,cutoff=epsSVD)
    if epsrr<0.0 #no rank reduction so do two more sweeps to make sure
        truncate!(H;orth=mlr,cutoff=epsSVD)
        truncate!(H;orth=ms.lr,cutoff=epsSVD)
    end
    @test is_regular_form(H,ms.ul,eps)
    @test is_canonical(H,ms,eps)
    # make sure the energy in unchanged
    E2l=inner(psi',H,psi)/(N-1)
    RE=abs((E0l-E2l)/E0l)
    REs= RE/sqrt(epsSVD)
    @printf "E0=%8.5f Etrunc=%8.5f rel. error=%7.1e RE/sqrt(espSVD)=%6.2f \n" E0l E2l RE REs
    @test REs<1.0 
    
end

# #these test are slow.  Uncomment and test if you mess with the truncate algo details.
# @testset "Compress insideous cases with no rank reduction, no QNs" begin
#     eps=2e-13
#     epsSVD=1e-12
#     epsrr=-1.0
#     ll=matrix_state(lower,left)
#     ul=matrix_state(upper,left)
#     lr=matrix_state(lower,right)
#     ur=matrix_state(upper,right)
#     hx=0.5

#     #                                 V=N sites
#     #                                   V=Num Nearest Neighbours in H
#     for N in 4:15
#         test_truncate(make_transIsing_MPO,N,N,hx,ll,epsSVD,epsrr,eps,false)
#         test_truncate(make_transIsing_MPO,N,N,hx,ul,epsSVD,epsrr,eps,false)
#         test_truncate(make_transIsing_MPO,N,N,hx,lr,epsSVD,epsrr,eps,false)
#         test_truncate(make_transIsing_MPO,N,N,hx,ur,epsSVD,epsrr,eps,false)
#     end
# end 


@testset "Compress full MPO no QNs" begin
    eps=2e-13
    epsSVD=1e-12
    epsrr=1e-12
    ll=matrix_state(lower,left)
    ul=matrix_state(upper,left)
    lr=matrix_state(lower,right)
    ur=matrix_state(upper,right)
    hx=0.5

    #                                 V=N sites
    #                                   V=Num Nearest Neighbours in H
    test_truncate(make_transIsing_MPO,5,1,hx,ll,epsSVD,epsrr,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,ll,epsSVD,epsrr,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,ul,epsSVD,epsrr,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,lr,epsSVD,epsrr,eps,false)
    test_truncate(make_transIsing_MPO,5,3,hx,ur,epsSVD,epsrr,eps,false)
    test_truncate(make_transIsing_MPO,15,10,hx,ll,epsSVD,epsrr,eps,false) 
    test_truncate(make_transIsing_MPO,15,10,hx,lr,epsSVD,epsrr,eps,false) 
    # Rank revealing QR/RQ  won't fully reduce these two cases
    test_truncate(make_transIsing_MPO,14,13,hx,ll,epsSVD,epsrr,eps,false) 
    test_truncate(make_transIsing_MPO,14,13,hx,lr,epsSVD,epsrr,eps,false) 

    # epsSVD=.00001
    test_truncate(make_transIsing_MPO,10,7,hx,ll,epsSVD,epsrr,eps,false) 
    test_truncate(make_transIsing_MPO,10,7,hx,lr,epsSVD,epsrr,eps,false) 
    test_truncate(make_transIsing_MPO,10,7,hx,ur,epsSVD,epsrr,eps,false)
    test_truncate(make_transIsing_MPO,10,7,hx,ul,epsSVD,epsrr,eps,false)

    #
    # Heisenberg from AutoMPO
    #
    # epsSVD=.00001
    test_truncate(make_Heisenberg_AutoMPO,10,7,hx,lr,epsSVD,epsrr,eps,false)

end 

@testset "Compress full MPO with QNs" begin
    ITensors.ITensors.enable_debug_checks()
    eps=2e-13
    epsSVD=1e-12
    epsrr=1e-12
    hx=0.0
    ll=matrix_state(lower,left)
    ul=matrix_state(upper,left)
    lr=matrix_state(lower,right)
    ur=matrix_state(upper,right)

    #                                 V=N sites
    #                                   V=Num Nearest Neighbours in H
    test_truncate(make_transIsing_MPO,5,1,hx,ll,epsSVD,epsrr,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,ll,epsSVD,epsrr,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,ul,epsSVD,epsrr,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,lr,epsSVD,epsrr,eps,true)
    test_truncate(make_transIsing_MPO,5,3,hx,ur,epsSVD,epsrr,eps,true)
    test_truncate(make_transIsing_MPO,15,10,hx,ll,epsSVD,epsrr,eps,true) 
    test_truncate(make_transIsing_MPO,15,10,hx,lr,epsSVD,epsrr,eps,true) 
    # Rank revealing QR/RQ  won't fully reduce these two cases
    test_truncate(make_transIsing_MPO,14,13,hx,ll,epsSVD,epsrr,eps,true) 
    test_truncate(make_transIsing_MPO,14,13,hx,lr,epsSVD,epsrr,eps,true) 

    epsSVD=.00001
    test_truncate(make_transIsing_MPO,10,7,hx,ll,epsSVD,epsrr,eps,true)  
    test_truncate(make_transIsing_MPO,10,7,hx,lr,epsSVD,epsrr,eps,true) 
    test_truncate(make_transIsing_MPO,10,7,hx,ur,epsSVD,epsrr,eps,true)
    test_truncate(make_transIsing_MPO,10,7,hx,ul,epsSVD,epsrr,eps,true)  

    # #
    # # Heisenberg from AutoMPO
    # #
    # epsSVD=.00001
    # test_truncate(make_Heisenberg_AutoMPO,10,7,lr,epsSVD,epsrr,eps,false)

end 

@testset "Test ground states" begin
    eps=3e-13
    epsSVD=1e-12
    epsrr=1e-12
    N=10
    NNN=5
    hx=0.5
    db=ITensors.using_debug_checks()

    sites = siteinds("SpinHalf", N)
    H=make_transIsing_MPO(sites,NNN;hx=hx)

    ITensors.ITensors.disable_debug_checks() 
    E0,psi0=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end
    truncate!(H;orth=left,cutoff=epsSVD,epsrr=epsrr)
    
    ITensors.ITensors.disable_debug_checks() 
    E1,psi1=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    overlap=abs(inner(psi0,psi1))
    RE=abs((E0-E1)/E0)
    @printf "Trans. Ising E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
    @test E0 ≈ E1 atol = eps
    @test overlap ≈ 1.0 atol = eps

    hx=0.0
    H=make_Heisenberg_AutoMPO(sites,NNN;hx=hx)

    ITensors.ITensors.disable_debug_checks() 
    E0,psi0=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    truncate!(H;orth=right,cutoff=epsSVD,epsrr=epsrr)

    ITensors.ITensors.disable_debug_checks() 
    E1,psi1=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    overlap=abs(inner(psi0,psi1))
    RE=abs((E0-E1)/E0)
    @printf "Heisenberg   E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
    @test E0 ≈ E1 atol = eps
    @test overlap ≈ 1.0 atol = eps
end 

@testset "Look at bond singular values for large lattices" begin
    NNN=5

    @printf "                 max(sv)           min(sv)\n"
    @printf "  N    Dw  left     mid    right   mid\n"
    for N in [6,8,10,12,18,20,40,80]
        sites = siteinds("SpinHalf", N)
        H=make_transIsing_MPO(sites,NNN;hx=0.5)
        specs=truncate!(H,cutoff=1e-10)
        imid::Int64=N/2-1
        max_1  =specs[1   ].spectrum[1]
        max_mid=specs[imid].spectrum[1]
        max_N  =specs[N-1 ].spectrum[1]
        min_mid=specs[imid].spectrum[end]
        Dw=maximum(get_Dw(H))
        @printf "%4i %4i %1.5f %1.5f %1.5f %1.5f\n" N Dw  max_1 max_mid max_N min_mid
        @test (max_1-max_N)<1e-10
        @test max_mid<1.0
    end
end

@testset "Head to Head autoMPO with 2-body Hamiltonian ul=$ul, QNs=$qns" for ul in [lower,upper],qns in [false,true]
    for N in 3:15
        NNN=div(N,2)
        svd_cutoff=1e-15 #same value auto MPO uses.
        rr_cutoff=1e-14
        sites = siteinds("SpinHalf", N;conserve_qns=qns)
        Hauto=make_transIsing_AutoMPO(sites,NNN;ul=ul) 
        Dw_auto=get_Dw(Hauto)
        Hr=make_transIsing_MPO(sites,NNN;ul=ul) 
        truncate!(Hr;orth=right,epsrr=rr_cutoff,cutoff=svd_cutoff) #sweep left to right
        @test is_canonical(Hr,matrix_state(ul,right),1e-12)
        delta_Dw=sum(get_Dw(Hr)-Dw_auto)
        #@test delta_Dw<=0
        if delta_Dw<0
            println("Compression beat AutoMPO by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
        end
        if delta_Dw>0
            println("AutoMPO beat Compression by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
        end
        Hl=make_transIsing_MPO(sites,NNN;ul=ul) 
        truncate!(Hl;orth=left,epsrr=rr_cutoff,cutoff=svd_cutoff) #sweep right to left
        @test is_canonical(Hl,matrix_state(ul,left),1e-12)
        delta_Dw=sum(get_Dw(Hr)-Dw_auto)
        if delta_Dw<0
            println("Compression beat AutoMPO by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=left ,ul=$ul,QNs=$qns")
        end
        if delta_Dw>0
            println("AutoMPO beat Compression by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
        end
    end  
end 

@testset "Head to Head autoMPO with 3-body Hamiltonian" begin
    @printf "+--------------+---------+------------------+------------------+\n"
    @printf "|              |autoMPO  |    1 truncation  | 2 truncations    |\n"
    @printf "|  N  epseSVD  | dE      |   dE     RE   Dw |   dE     RE   Dw |\n"

    for svd_cutoff in [1e-15,1e-12,1e-10]
        for N in [6,10,16,20,24]
            sites = siteinds("SpinHalf", N;conserve_qns=false)
            Hnot=make_Parker(sites;truncate=false) #No truncation inside autoMPO
            H=make_Parker(sites;truncate=true) #Truncated by autoMPO
            #@show get_Dw(Hnot)
            Dw_auto = get_Dw(H)
            psi=randomMPS(sites)
            Enot=inner(psi',Hnot,psi)
            E=inner(psi',H,psi)
            @test E ≈ Enot atol = sqrt(svd_cutoff)
            truncate!(Hnot;dir=right,cutoff=svd_cutoff)
            Dw_1=get_Dw(Hnot)
            delta_Dw_1=sum(Dw_auto-Dw_1)
            Enott1=inner(psi',Hnot,psi)
            @test E ≈ Enott1 atol = sqrt(svd_cutoff)
            RE1=abs(E-Enott1)/sqrt(svd_cutoff)

            truncate!(Hnot;dir=left,cutoff=svd_cutoff)
            Dw_2=get_Dw(Hnot)
            delta_Dw_2=sum(Dw_auto-Dw_2)
            Enott2=inner(psi',Hnot,psi)
            @test E ≈ Enott2 atol = sqrt(svd_cutoff)
            RE2=abs(E-Enott2)/sqrt(svd_cutoff)

            @printf "| %3i %1.1e  | %1.1e | %1.1e %1.3f %2i | %1.1e %1.3f %2i | \n" N svd_cutoff abs(E-Enot) abs(E-Enott1) RE1 delta_Dw_1 abs(E-Enott2) RE2 delta_Dw_2 
        end
    end
end


nothing
