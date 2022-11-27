using ITensors
using ITensorMPOCompression
using Test
using Printf
#import ITensors: rq



#using Printf
#Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
#println("-----------Start--------------")

@testset "QR,QL,LQ,RQ decomposition with rank revealing" begin
    N=10
    NNN=6
    hx=0.5
    eps=2e-15
    sites = siteinds("SpinHalf", N)
    #
    #  use lower tri MPO to get some zero pivots for QL and RQ.
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower)
    W=H[2]
    d,n,r,c=parse_links(W)

    Lind=noncommoninds(inds(W),c)
    Rind=noncommoninds(inds(W),r)
    @assert dim(c)==dim(r)
    #
    #  RQ decomp
    #
    R,Q,iq=rq(W,r;positive=true,epsrr=1e-10)
    @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
    @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
    @test W ≈ R*Q atol = eps
    #
    #  QL decomp
    #
    Q,L,iq=ql(W,Rind;positive=true,epsrr=1e-10)
    @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
    @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
    @test W ≈ L*Q atol = eps
    
    #
    #  use upper tri MPO to get some zero pivots for LQ and QR.
    #
    H=make_transIsing_MPO(sites,NNN,hx,upper)
    W=H[2]
    d,n,r,c=parse_links(W)

    Lind=noncommoninds(inds(W),c)
    Rind=noncommoninds(inds(W),r)
    @assert dim(c)==dim(r)

    #
    #  QR decomp
    #
    Q,R,iq=qr(W,Rind;positive=true,epsrr=1e-10)
    @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
    @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
    @test W ≈ R*Q atol = eps   
    #
    #  LQ decomp
    #
    L,Q,iq=lq(W,r;positive=true,epsrr=1e-10)
    @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
    @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
    @test W ≈ L*Q atol = eps
    
end

