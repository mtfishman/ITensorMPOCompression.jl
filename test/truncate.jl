using ITensors
using ITensorMPOCompression
using Revise
using Test
using Printf

import ITensorMPOCompression.truncate!
import ITensorMPOCompression.truncate
import ITensorMPOCompression.orthogonalize!

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
    H=makeH(sites,NNN,hx,ms.ul)
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

#these test are slow.  Uncomment and test if you mess with the truncate algo details.
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
    H=make_transIsing_MPO(sites,NNN,hx,lower)

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
    H=make_Heisenberg_AutoMPO(sites,NNN,hx,lower)

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
    hx=0.5

    @printf "                 max(sv)           min(sv)\n"
    @printf "  N    Dw  left     mid    right   mid\n"
    for N in [6,8,10,12,18,20,40,80]
        sites = siteinds("SpinHalf", N)
        H=make_transIsing_MPO(sites,NNN,hx,lower)
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
