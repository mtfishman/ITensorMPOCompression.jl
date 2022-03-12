using ITensors
using ITensorMPOCompression
using Revise
using Test
using Printf

import ITensorMPOCompression.truncate!
import ITensorMPOCompression.truncate
import ITensorMPOCompression.orthogonalize!

include("hamiltonians.jl")

# Base.show(io::IO, f::Float64) = @printf(io, "%1.5f", f)

function test_truncate(makeH,N::Int64,NNN::Int64,ms::matrix_state,epsSVD::Float64,epsrr::Float64,eps::Float64)
    mlr=mirror(ms.lr)
    hx=0.5
    J=1.0

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # Make right canonical, then compress to left canonical
    #
    H=makeH(sites,NNN,J,hx,ms.ul)
    E0l=inner(psi',H,psi)
    @test is_regular_form(H,ms.ul,eps)
    #@show get_Dw(H)
    orthogonalize!(H;dir=ms.lr,epsrr=epsrr)
    #@show get_Dw(H)

    E1l=inner(psi',H,psi)
    RE=abs((E0l-E1l)/E0l)
    #@printf "E0=%1.5f E1=%1.5f rel. error=%.5e  \n" E0l E1l RE 
    @test RE<2*eps
    @test is_regular_form(H,ms.ul,eps)
    @test is_canonical(H,ms,eps)

    truncate!(H;dir=mlr,cutoff=epsSVD)
    #@show get_Dw(H)
    ss=truncate!(H;dir=ms.lr,cutoff=epsSVD)
    #@show get_Dw(H)
    @test is_regular_form(H,ms.ul,eps)
    @test is_canonical(H,ms,eps)
    #@show ITensorMPOCompression.min(ss) ITensorMPOCompression.max(ss)
    # make sure the energy in unchanged
    E2l=inner(psi',H,psi)
    RE=abs((E0l-E2l)/E0l)
    @printf "E0=%8.5f Etrunc=%8.5f rel. error=%7.1e RE/espSVD=%6.2f \n" E0l E2l RE RE/epsSVD
    if epsSVD<=1e-12
        @test (RE/epsSVD)<1.0 #typically this will remove any meaning SVs
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
    # test_truncate(make_transIsing_MPO,15,10,ll,epsSVD,epsrr,eps) #know fail, triggers fixing RL_prime
    test_truncate(make_transIsing_MPO,15,10,lr,epsSVD,epsrr,eps)

    epsSVD=.00001
    test_truncate(make_transIsing_MPO,10,7,ll,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,10,7,lr,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,10,7,ur,epsSVD,epsrr,eps)
    test_truncate(make_transIsing_MPO,10,7,ur,epsSVD,epsrr,eps)

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
    J=1.0
    db=ITensors.using_debug_checks()

    sites = siteinds("SpinHalf", N)
    H=make_transIsing_MPO(sites,NNN,J,hx,lower)

    ITensors.ITensors.disable_debug_checks() 
    E0,psi0=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    truncate!(H;dir=right,cutoff=epsSVD,epsrr=epsrr)

    ITensors.ITensors.disable_debug_checks() 
    E1,psi1=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    overlap=abs(inner(psi0',psi1))
    RE=abs((E0-E1)/E0)
    @printf "Trans. Ising E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
    @test abs(E0-E1)<eps
    @test abs(overlap-1.0)<eps

    hx=0.0
    H=make_Heisenberg_AutoMPO(sites,NNN,J,hx,lower)

    ITensors.ITensors.disable_debug_checks() 
    E0,psi0=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    truncate!(H;dir=right,cutoff=epsSVD,epsrr=epsrr)

    ITensors.ITensors.disable_debug_checks() 
    E1,psi1=fast_GS(H,sites)
    if db  ITensors.ITensors.enable_debug_checks() end

    overlap=abs(inner(psi0',psi1))
    RE=abs((E0-E1)/E0)
    @printf "Heisenberg E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
    @test abs(E0-E1)<eps
    @test abs(overlap-1.0)<eps
end