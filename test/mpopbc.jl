
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

function test_auto_vs_direct(sites,NNN::Int64,hx::Float64,Eexpected::Float64,eps::Float64)
    #
    #  Use autoMPO to make H
    #
    Hauto=make_transIsing_AutoMPO(sites,NNN,hx) #make and MPO only to get the indices
    Eauto,psi=fast_GS(Hauto,sites)
    #@show Eauto
    @test abs(Eauto-Eexpected)<eps
    Eauto1=inner(psi',Hauto,psi)
    
    #pprint(H[1],eps)
    #
    #  Make H directly ... should be lower triangular
    #
    Hdirect=make_transIsing_MPO(sites,NNN,hx) 
    
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
    eps=4e-14 #this is right at the lower limit for passing the tests.
    sites = siteinds("SpinHalf", N)

    # It seems we need to jump through hoops to avoid the 
    # "type Dense has no field blockoffsets" from inside DMRG errors
    # I tried to move these disables inside fast_GS with no success?!?!?
    db=ITensors.using_debug_checks()
    if db  ITensors.ITensors.disable_debug_checks() end
    test_auto_vs_direct(sites,1,hx,-1.5066685458330529,eps) #1st neighbour interactions
    test_auto_vs_direct(sites,2,hx,-1.4524087749432490,eps) #1&2 neighbour interactions
    test_auto_vs_direct(sites,3,hx,-1.4516941302867301,eps) #1->3 neighbour interactions
    test_auto_vs_direct(sites,4,hx,-1.4481111362390489,eps) #1->4 neighbour interactions
    if db  ITensors.ITensors.enable_debug_checks() end
end


@testset "Check obc=false setting for auto MPO" begin
    N=5
    NNN=3
    hx=0.5
    eps=1e-14
    sites = siteinds("SpinHalf", N; conserve_qns=false)

    H_pbc=make_transIsing_AutoMPO(sites,NNN,hx;obc=false) 
    @test has_pbc(H_pbc)
    @test is_regular_form(H_pbc,lower,eps)
    H_obc=make_transIsing_AutoMPO(sites,NNN,hx;obc=true) 
    @test !has_pbc(H_obc)

    # It seems we need to jump through hoops to avoid the 
    # "type Dense has no field blockoffsets" from inside DMRG errors
    # I tried to move these disables inside fast_GS with no success?!?!?
    db=ITensors.using_debug_checks()
    if db  ITensors.ITensors.disable_debug_checks() end
    E_pbc,psi=fast_GS(to_openbc(H_pbc),sites)
    E_obc,psi=fast_GS(H_obc,sites)
    if db  ITensors.ITensors.enable_debug_checks() end
    @show E_pbc E_obc
    @test abs(E_pbc-E_obc)<eps

    H_pbc=make_Heisenberg_AutoMPO(sites,NNN,hx;obc=false) 
    @test has_pbc(H_pbc)
    @test is_regular_form(H_pbc,lower,eps)
    H_obc=make_Heisenberg_AutoMPO(sites,NNN,hx;obc=true) 
    @test !has_pbc(H_obc)
    if db  ITensors.ITensors.disable_debug_checks() end
    E_pbc,psi=fast_GS(to_openbc(H_pbc),sites)
    E_obc,psi=fast_GS(H_obc,sites)
    if db  ITensors.ITensors.enable_debug_checks() end
    @show E_pbc E_obc
    @test abs(E_pbc-E_obc)<eps
end