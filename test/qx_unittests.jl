using ITensors
using ITensorMPOCompression
using Test
using Printf
import ITensors: rq



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
    R,Q=rq(W,r;positive=true,epsrr=1e-10)
    iq=commonindex(Q,R)
    @printf "RQ decomposition %4i rows were removed from R\n" dim(c)-dim(iq)
    Id=Q*prime(Q,iq)
    Idq=delta(iq,iq')
    @test norm(Id-Idq)<eps
    @test norm(W-R*Q)<eps
    #
    #  QL decomp
    #
    Q,L=ql(W,Rind;positive=true,epsrr=1e-10)
    iq=commonindex(Q,L)
    @printf "QL decomposition %4i rows were removed from L\n" dim(c)-dim(iq)
    Id=Q*prime(Q,iq)
    Idq=delta(iq,iq')
    @test norm(Id-Idq)<eps
    @test norm(W-L*Q)<eps    
    
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
    Q,R=qr(W,Rind;positive=true,epsrr=1e-10)
    iq=commonindex(Q,R)
    @printf "QR decomposition %4i rows were removed from R\n" dim(c)-dim(iq)
    Id=Q*prime(Q,iq)
    Idq=delta(iq,iq')
    @test norm(Id-Idq)<eps
    @test norm(W-R*Q)<eps
    #
    #  LQ decomp
    #
    L,Q=lq(W,r;positive=true,epsrr=1e-10)
    iq=commonindex(Q,L)
    @printf "LQ decomposition %4i rows were removed from L\n" dim(c)-dim(iq)
    Id=Q*prime(Q,iq)
    Idq=delta(iq,iq')
    @test norm(Id-Idq)<eps
    @test norm(W-L*Q)<eps

    

end
 