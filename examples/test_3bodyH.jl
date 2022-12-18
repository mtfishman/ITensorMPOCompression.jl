using ITensors
using ITensorMPOCompression
using Printf
using Test

@testset "Verify auto MPO and hand built 3-body Hamiltonians as identical" begin
    #l,r=Index(1,"Link,c=0,l=1"),Index(1,"Link,c=1,l=1")
    l,r=Index(1,"Link,l=0"),Index(1,"Link,l=1")
    ul=lower
    for N in 2:7
        sites = siteinds("S=1/2",N);
        Hhand=make_3body_MPO(sites)
        Hauto=make_3body_AutoMPO(sites)
        truncate!(Hhand;orth=left,cutoff=1e-15,epsrr=1e-15)
        ss_hand=truncate!(Hhand;orth=right,cutoff=1e-15,epsrr=1e-15)
        truncate!(Hauto;orth=left,cutoff=1e-15,epsrr=1e-15)
        ss_auto=truncate!(Hauto;orth=right,cutoff=1e-15,epsrr=1e-15)
        @test length(ss_hand)==length(ss_auto)
        for nb in 1:length(ss_hand)
            sh=eigs(ss_hand[nb])
            sa=eigs(ss_auto[nb])
            ds=.√(sh)-.√(sa)
            @test sqrt(sum(ds.^2))  ≈ 0.0 atol = 1e-15
        end
    end
end

nothing