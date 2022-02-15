using ITensors
using Test

function test_qx(::Type{ElT},qx::Function,M::Int64,N::Int64,pos::Bool) where {ElT<:Number}
    ir=Index(M,"row")
    ic=Index(N,"col")
    A=randomITensor(ElT,ir,ic)
    Q,LR=qx(A,ir;positive=pos)
    norm(A-Q*LR)
end
@testset "QR decomposition" begin
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

@testset "QL decomposition" begin
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