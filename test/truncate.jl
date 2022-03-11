using ITensors
using ITensorMPOCompression
using Revise
using Test

import ITensorMPOCompression.truncate!
import ITensorMPOCompression.truncate
import ITensorMPOCompression.orthogonalize!

include("hamiltonians.jl")

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.5f", f)
# println("-----------Start--------------")

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
    @printf "E0=%.5f Etrunc=%.5f rel. error=%.5e RE/espSVD=%.2f \n" E0l E2l RE RE/epsSVD
    if epsSVD<=1e-12
        @test (RE/epsSVD)<1.0 #typically this will remove any meaning SVs
    else
        @test (RE/epsSVD)<1000.0 #now we may remove real SVs and expect much begger energy errors
    end
end

@testset "Compress full MPO" begin
    hx=0.5
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
