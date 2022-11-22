
using ITensorMPOCompression
using Revise
using Test

NNEs=[(1,-1.5066685458330529),(2,-1.4524087749432490),(3,-1.4516941302867301),(4,-1.4481111362390489)]
@testset "MPOs hand coded versus autoMPO" for nne in NNEs

    N=5
    hx=0.5
    eps=4e-14 #this is right at the lower limit for passing the tests.
    NNN=nne[1]
    Eexpected=nne[2]
    
    sites = siteinds("SpinHalf", N;conserve_qns=false)
    ITensors.ITensors.disable_debug_checks() #dmrg crashes when this in.
    #
    #  Use autoMPO to make H
    #
    Hauto=make_transIsing_AutoMPO(sites,NNN,hx,lower) 
    Eauto,psi=fast_GS(Hauto,sites)
    @test Eauto ≈ Eexpected atol = eps
    Eauto1=inner(psi',Hauto,psi)
    @test Eauto1 ≈ Eexpected atol = eps
    
    #
    #  Make H directly ... should be lower triangular
    #
    Hdirect=make_transIsing_MPO(sites,NNN,hx,lower) #defaults to lower reg form
    @test order(Hdirect[1])==3    
    @test inner(psi',Hdirect,psi) ≈ Eexpected atol = eps
    Edirect,psidirect=fast_GS(Hdirect,sites)
    @test Edirect ≈ Eexpected atol = eps
    overlap=abs(inner(psi',psidirect))
    @test overlap ≈ 1.0 atol = eps 
end 

nothing