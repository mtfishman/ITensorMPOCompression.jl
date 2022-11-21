using ITensors
using ITensorMPOCompression
using Revise
using Test
using Printf

import ITensorMPOCompression.truncate!
import ITensorMPOCompression.truncate
import ITensorMPOCompression.orthogonalize!

#brute force method to control the default float display format.
# Base.show(io::IO, f::Float64) = @printf(io, "%1.5f", f)

#
#  We need consistent output from randomMPS in order to avoid flakey unit testset
#  So we just pick any seed so that randomMPS (hopefull) gives us the same 
#  pseudo-random output for each test run.
#
using Random
Random.Random.seed!(12345);

function test_truncate(makeH,N::Int64,NNN::Int64,ms::matrix_state,epsSVD::Float64,epsrr::Float64,eps::Float64)
    mlr=mirror(ms.lr)
    hx=0.5

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # Make right canonical, then compress to left canonical
    #
    H=makeH(sites,NNN,hx,ms.ul)
    E0l=inner(psi',H,psi)
    @test is_regular_form(H,ms.ul,eps)
    #@show get_Dw(H)
    orthogonalize!(H;orth=ms.lr,epsrr=epsrr)
    #@show get_Dw(H)

    E1l=inner(psi',H,psi)
    RE=abs((E0l-E1l)/E0l)
    #@printf "E0=%1.5f E1=%1.5f rel. error=%.5e  \n" E0l E1l RE 
    @test RE ≈ 0 atol = 2*eps
    @test is_regular_form(H,ms.ul,eps)
    @test is_canonical(H,ms,eps)

    truncate!(H;orth=mlr,cutoff=epsSVD)
    #@show get_Dw(H)
    ss=truncate!(H;orth=ms.lr,cutoff=epsSVD)
    #@show get_Dw(H)
    @test is_regular_form(H,ms.ul,eps)
    @test is_canonical(H,ms,eps)
    #@show ITensorMPOCompression.min(ss) ITensorMPOCompression.max(ss)
    # make sure the energy in unchanged
    E2l=inner(psi',H,psi)
    RE=abs((E0l-E2l)/E0l)
    @printf "E0=%8.5f Etrunc=%8.5f rel. error=%7.1e RE/espSVD=%6.2f \n" E0l E2l RE RE/epsSVD
    if epsSVD<=1e-12
        @test (RE/epsSVD)<1.0 #typically this will remove any meaningless SVs
    else
        @test (RE/epsSVD)<1000.0 #now we may remove real SVs and expect much bigger energy errors
    end
end



@testset "Compress full MPO" begin
    eps=2e-13
    epsSVD=1e-12
    epsrr=1e-12
    ll=matrix_state(lower,left)
    ul=matrix_state(upper,left)
    lr=matrix_state(lower,right)
    ur=matrix_state(upper,right)

    #                                 V=N sites
    #                                   V=Num Nearest Neighbours in H
    test_truncate(make_transIsing_MPO,5,1,ll,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,5,3,ll,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,5,3,ul,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,5,3,lr,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,5,3,ur,epsSVD,epsrr,eps)
    #test_truncate(make_transIsing_MPO,15,10,ll,epsSVD,epsrr,eps) #know fail, triggers fixing RL_prime
    test_truncate(make_transIsing_MPO,15,10,lr,epsSVD,epsrr,eps) 

    epsSVD=.00001
    test_truncate(make_transIsing_MPO,10,7,ll,epsSVD,epsrr,eps) 
    test_truncate(make_transIsing_MPO,10,7,lr,epsSVD,epsrr,eps) 
    test_truncate(make_transIsing_MPO,10,7,ur,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,10,7,ul,epsSVD,epsrr,eps)

    #
    # Heisenberg from AutoMPO
    #
    epsSVD=.00001
    test_truncate(make_Heisenberg_AutoMPO,10,7,lr,epsSVD,epsrr,eps)

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

    overlap=abs(inner(psi0',psi1))
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

    overlap=abs(inner(psi0',psi1))
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
#= 
@testset "Test with conserved QNs" begin
    N = 10
    NNN = 1
    hx=0.0 #can't make and sx op with QNs in play
    sites = siteinds("S=1/2",N;conserve_qns=true)
    # to build our own MPO we need to put QNs on all the link indices.
    # from autoMPO we see QN("Sz",0) => 3 as the qn for a link index.
    H=make_transIsing_MPO(sites,NNN,hx,upper)
    #H=make_transIsing_AutoMPO(sites,NNN,hx,lower)
    pprint(H[2],1e-14)
    orthogonalize!(H)
    #pprint(H[2],1e-14)
    #@show H[2]
    #i=Index(QN("Sz",0)=>3;dir=ITensors.In,tags="Link,l=1")
    #@show i
end
 =#