using ITensorMPOCompression
using Revise
using Test

include("hamiltonians.jl")

@testset "Bring MPO into canonical form" begin
    N=5
    NNN=4
    hx=0.5
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    H=make_transIsing_MPO(sites,NNN,hx,pbc=true) 
    E0=inner(psi',to_openbc(H),psi)
    @show E0

    canonical!(H)
    
    E1=inner(psi',to_openbc(H),psi)
    @test abs(E0-E1)<1e-14
    msl=matrix_state(lower,left )
    msr=matrix_state(lower,right)
    for n in 1:N-1
        @test  is_canonical(H[n],msl,eps)
        @test !is_canonical(H[n],msr,eps)
    end
end