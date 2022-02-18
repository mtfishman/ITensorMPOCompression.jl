using ITensors
using ITensorMPOCompression
using Revise
using Test

include("../test/hamiltonians.jl")



function runtest()
    N=5
    NNN=1
    hx=0.5
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    H=make_transIsing_MPO(sites,NNN,hx,pbc=true) 
    canonical!(H)
    @test is_canonical(H[2],matrix_state(lower,left),eps)
    @test !is_canonical(H[2],matrix_state(lower,right),eps)
end

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
println("-----------Start--------------")
runtest()