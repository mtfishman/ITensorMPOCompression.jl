
using ITensors
using ITensorMPOCompression
using Revise
using Test

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)


function make_random_qindex(d::Int64,nq::Int64)::Index
    qns=Pair{QN, Int64}[]
    for n in 1:nq
        append!(qns,[QN()=>rand(1:d)])
    end
    return Index(qns,"Link,l=1")
end


NNEs=[(1,-1.5066685458330529),(2,-1.4524087749432490),(3,-1.4516941302867301),(4,-1.4481111362390489)]
@testset "MPOs hand coded versus autoMPO give same GS energies" for nne in NNEs

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
    overlap=abs(inner(psi,psidirect))
    @test overlap ≈ 1.0 atol = eps 
end 

@testset "Redim function with non-trivial QN spaces" begin
   
    for offset in 0:2
        for d in 1:5
            for nq in 1:5
                il=make_random_qindex(d,nq)
                Dw=dim(il)
                if Dw>1+offset
                    #@show il
                    ilr=redim(il,Dw-offset-1,offset)
                    #@show ilr
                    @test dim(ilr)==Dw-offset-1
                end
            
            end #for nq
        end #for d
    end #for offset
end #@testset

makeHs=[make_transIsing_AutoMPO,make_transIsing_MPO,make_Heisenberg_AutoMPO]
@testset "Auto MPO Ising Ham with Sz blocking" for makeH in makeHs
    N=5
    hx=0.0 #Hx!=0 breaks symmetry.
    eps=4e-14 #this is right at the lower limit for passing the tests.
    NNN=2
    
    sites = siteinds("SpinHalf", N;conserve_qns=true)
    H=makeH(sites,NNN,hx,lower) 
    il=filterinds(inds(H[2]),tags="Link")
    for i in 1:2
        for start_offset in 0:2
            for end_offset in 0:2
                Dw=dim(il[i])
                Dw_new=Dw-start_offset-start_offset
                if Dw_new>0 
                    ilr=redim(il[i],Dw_new,start_offset)
                    @test dim(ilr)==Dw_new
                end #if
            end #for end_offset
        end #for start_offset 
    end #for i
end #@testset

@testset "hand coded versus autoMPO with conserve_qns=true have the same index directions" begin
    N=5
    hx=0.0 #Hx!=0 breaks symmetry.
    eps=4e-14 #this is right at the lower limit for passing the tests.
    NNN=2
    
    sites = siteinds("SpinHalf", N;conserve_qns=true)
    Hauto=make_transIsing_AutoMPO(sites,NNN,hx,lower) 
    Hhand=make_transIsing_MPO(sites,NNN,hx,lower) 
    for (Wauto,Whand) in zip(Hauto,Hhand)
        for ia in inds(Wauto)
            ih=filterinds(Whand,tags=tags(ia),plev=plev(ia))[1]
            @test dir(ia)==dir(ih)
        end
    end
end

@testset "Parker eq. 34 3-body Hamiltonian" begin
    N=15
    sites = siteinds("SpinHalf", N;conserve_qns=false)
    Hnot=make_Parker(sites;truncate=false) #No truncation inside autoMPO
    H=make_Parker(sites;truncate=true) #Truncated by autoMPO
    psi=randomMPS(sites)
    Enot=inner(psi',Hnot,psi)
    E=inner(psi',H,psi)
    @test E ≈ Enot atol = 1e-9
   
end
nothing