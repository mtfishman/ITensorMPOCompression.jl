using ITensors
using ITensorMPOCompression
using Test
using Printf



#using Printf
#Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
#println("-----------Start--------------")

@testset "Block respecting QX decomposition" for ul in [lower,upper], lr in [left,right]
    N=6
    NNN=4
    model_kwargs = (hx=0.5, ul=ul )
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    ms=matrix_state(ul,lr)
    #
    #  test lower triangular MPO 
    #
    H=make_transIsing_MPO(sites,NNN;model_kwargs...)
    for n in sweep(H,lr)
        W=H[n]
        @test is_regular_form(W,ms.ul)
        Q,RL,lq=block_qx(W,ms.ul;orth=ms.lr)
        @test is_canonical(Q,ms,eps)
    end
end
    
# @testset "QR,QL,LQ,RQ decomposition with rank revealing" begin
#     N=10
#     NNN=6
#     model_kwargs = (hx=0.5, ul=lower)
#     eps=2e-15
#     sites = siteinds("SpinHalf", N)
#     #
#     #  use lower tri MPO to get some zero pivots for QL and RQ.
#     #
#     H=make_transIsing_MPO(sites,NNN;model_kwargs...)
#     W=H[2]
#     r,c=parse_links(W)

#     Lind=noncommoninds(inds(W),c)
#     Rind=noncommoninds(inds(W),r)
#     @mpoc_assert dim(c)==dim(r)
#     #
#     #  RQ decomp
#     #
#     R,Q,iq=rq(W,r;positive=true,rr_cutoff=1e-10)
#     @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
#     @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
#     @test W ≈ R*Q atol = eps
#     #
#     #  QL decomp
#     #
#     Q,L,iq=ql(W,Rind;positive=true,rr_cutoff=1e-10)
#     @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
#     @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
#     @test W ≈ L*Q atol = eps
    
#     #
#     #  use upper tri MPO to get some zero pivots for LQ and QR.
#     #
#     model_kwargs = (hx=0.5, ul=upper)
#     H=make_transIsing_MPO(sites,NNN;model_kwargs...)
#     W=H[2]
#     r,c=parse_links(W)

#     Lind=noncommoninds(inds(W),c)
#     Rind=noncommoninds(inds(W),r)
#     @mpoc_assert dim(c)==dim(r)

#     #
#     #  QR decomp
#     #
#     Q,R,iq=qr(W,Rind;positive=true,rr_cutoff=1e-10)
#     @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
#     @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
#     @test W ≈ R*Q atol = eps   
#     #
#     #  LQ decomp
#     #
#     L,Q,iq=lq(W,r;positive=true,rr_cutoff=1e-10)
#     @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
#     @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
#     @test W ≈ L*Q atol = eps
    
# end

nothing
