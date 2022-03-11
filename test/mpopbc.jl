
using ITensorMPOCompression
using Revise
using Test

#ITensors.ITensors.enable_debug_checks()

include("hamiltonians.jl")

function fast_GS(H::MPO,sites)::Tuple{Float64,MPS}
    psi0  = randomMPS(sites,length(H))
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 2,4,8,16,32)
    setcutoff!(sweeps, 1E-10)
    E,psi= dmrg(H,psi0, sweeps;outputlevel=0)
    return E,psi
end

function test_auto_vs_direct(sites,NNN::Int64,J::Float64,hx::Float64,Eexpected::Float64,eps::Float64)
    #
    #  Use autoMPO to make H
    #
    Hauto=make_transIsing_AutoMPO(sites,NNN,J,hx) #make and MPO only to get the indices
    Eauto,psi=fast_GS(Hauto,sites)
    @test abs(Eauto-Eexpected)<eps
    Eauto1=inner(psi',Hauto,psi)
    @test abs(Eauto1-Eexpected)<eps
    
    #
    #  Make H directly ... should be lower triangular
    #
    Hdirect=make_transIsing_MPO(sites,NNN,J,hx) #defaults to lower reg form
    @test order(Hdirect[1])==3    
    @test abs(inner(psi',Hdirect,psi)-Eexpected)<eps
    Edirect,psidirect=fast_GS(Hdirect,sites)
    @test abs(Edirect-Eexpected)<eps
    overlap=abs(inner(psi',psidirect))
    @test abs(overlap-1.0)<eps
end

#using Printf
#Base.show(io::IO, f::Float64) = @printf(io, "%1.16f", f)
#println("-----------Start--------------")

@testset "MPOs with periodic boundary conditions" begin

    N=5
    hx=0.5
    J=1.0
    eps=4e-14 #this is right at the lower limit for passing the tests.
    sites = siteinds("SpinHalf", N;conserve_qns=false)
 
    # It seems we need to jump through hoops to avoid the 
    # "type Dense has no field blockoffsets" from inside DMRG errors
    # I tried to move these disables inside fast_GS with no success?!?!?
    db=ITensors.using_debug_checks()
    if db  ITensors.ITensors.disable_debug_checks() end
    test_auto_vs_direct(sites,1,J,hx,-1.5066685458330529,eps) #1st neighbour interactions
    test_auto_vs_direct(sites,2,J,hx,-1.4524087749432490,eps) #1&2 neighbour interactions
    test_auto_vs_direct(sites,3,J,hx,-1.4516941302867301,eps) #1->3 neighbour interactions
    test_auto_vs_direct(sites,4,J,hx,-1.4481111362390489,eps) #1->4 neighbour interactions
    if db  ITensors.ITensors.enable_debug_checks() end
end
