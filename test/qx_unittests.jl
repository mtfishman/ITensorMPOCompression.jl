using ITensors
using Test

#
# Single index tests
#
function test_qx(::Type{ElT},qx::Function,M::Int64,N::Int64,pos::Bool) where {ElT<:Number}
    ir=Index(M,"row")
    ic=Index(N,"col")
    A=randomITensor(ElT,ir,ic)
    Q,LR=qx(A,ir;positive=pos)
    norm(A-Q*LR)
end

function test_xq(::Type{ElT},xq::Function,M::Int64,N::Int64,pos::Bool) where {ElT<:Number}
    ir=Index(M,"row")
    ic=Index(N,"col")
    A=randomITensor(ElT,ir,ic)
    LR,Q=xq(A,ir;positive=pos)
    norm(A-LR*Q)
end

@testset "QR decomposition, single index" begin
    eps=1e-14
    M,N=5,5
    M1,N1=10,10
    @test test_qx(Float64   ,qr,M ,N,false)<eps
    @test test_qx(Float64   ,qr,M ,N1,false)<eps
    @test test_qx(Float64   ,qr,M1,N1,false)<eps
    @test test_qx(Float64   ,qr,M ,N ,true )<eps
    @test test_qx(Float64   ,qr,M ,N1,true )<eps
    @test test_qx(Float64   ,qr,M1,N1,true )<eps
    @test test_qx(ComplexF64,qr,M ,N ,false)<eps
    @test test_qx(ComplexF64,qr,M ,N1,false)<eps
    @test test_qx(ComplexF64,qr,M1,N1,false)<eps
    @test test_qx(ComplexF64,qr,M ,N ,true )<eps
    @test test_qx(ComplexF64,qr,M ,N1,true )<eps
    @test test_qx(ComplexF64,qr,M1,N1,true )<eps
end

@testset "QL decomposition, single index" begin
    eps=1e-14
    M,N=5,5
    M1,N1=10,10
    @test test_qx(Float64   ,ql,M ,N,false)<eps
    @test test_qx(Float64   ,ql,M ,N1,false)<eps
    @test test_qx(Float64   ,ql,M1,N1,false)<eps
    @test test_qx(Float64   ,ql,M ,N ,true )<eps
    @test test_qx(Float64   ,ql,M ,N1,true )<eps
    @test test_qx(Float64   ,ql,M1,N1,true )<eps
    @test test_qx(ComplexF64,ql,M ,N ,false)<eps
    @test test_qx(ComplexF64,ql,M ,N1,false)<eps
    @test test_qx(ComplexF64,ql,M1,N1,false)<eps
    @test test_qx(ComplexF64,ql,M ,N ,true )<eps
    @test test_qx(ComplexF64,ql,M ,N1,true )<eps
    @test test_qx(ComplexF64,ql,M1,N1,true )<eps
end

@testset "LQ decomposition, single index" begin
    eps=1e-14
    M,N=5,5
    M1,N1=10,10
    @test test_xq(Float64   ,lq,M ,N,false)<eps
    @test test_xq(Float64   ,lq,M ,N1,false)<eps
    @test test_xq(Float64   ,lq,M1,N1,false)<eps
    @test test_xq(Float64   ,lq,M ,N ,true )<eps
    @test test_xq(Float64   ,lq,M ,N1,true )<eps
    @test test_xq(Float64   ,lq,M1,N1,true )<eps
    @test test_xq(ComplexF64,lq,M ,N ,false)<eps
    @test test_xq(ComplexF64,lq,M ,N1,false)<eps
    @test test_xq(ComplexF64,lq,M1,N1,false)<eps
    @test test_xq(ComplexF64,lq,M ,N ,true )<eps
    @test test_xq(ComplexF64,lq,M ,N1,true )<eps
    @test test_xq(ComplexF64,lq,M1,N1,true )<eps
end

#
#  Test multiple indicies using MPO matrices
#

include("hamiltonians.jl")

@testset "QR,QL,LQ decomposition fo MPO matrices" begin
    N=5
    NNN=4
    hx=0.5
    eps=2e-15
    sites = siteinds("SpinHalf", N)
    H=make_transIsing_MPO(sites,NNN,hx,pbc=true)
    W=H[1]
    d,n,r,c=parse_links(W)

    Lind=noncommoninds(inds(W),c)
    Rind=noncommoninds(inds(W),r)

    Q,R=qr(W,Lind;positive=true)
    iqr=commonindex(Q,R)
    Id=Q*prime(Q,iqr)
    Idq=delta(iqr,iqr)
    @test norm(Id-Idq)<eps
    @test norm(W-Q*R)<eps

    Q,L=ql(W,Lind;positive=true)
    iql=commonindex(Q,L)
    Id=Q*prime(Q,iql)
    Idq=delta(iql,iql)
    @test norm(Id-Idq)<eps
    @test norm(W-Q*L)<eps

    L,Q=lq(W,Rind;positive=true)
    iql=commonindex(Q,L)
    Id=Q*prime(Q,iql)
    Idq=delta(iql,iql)
    @test norm(Id-Idq)<eps
    @test norm(W-L*Q)<eps


end