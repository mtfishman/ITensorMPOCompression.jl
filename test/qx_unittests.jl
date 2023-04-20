using ITensors
using ITensorMPOCompression
using Test
using Printf



#using Printf
#Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
#println("-----------Start--------------")

models=[
    [make_transIsing_MPO,"S=1/2",true],
    [make_transIsing_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1",true],
    [make_Hubbard_AutoMPO,"Electron",false],
]

@testset "Ac Block respecting QX decomposition $(model[1]), qns=$qns, ul=$ul, lr=$lr" for model in models, qns in [false,true], ul in [lower,upper], lr in [right,left]
    N=6
    NNN=4
    model_kwargs = (hx=0.5, ul=ul )
    eps=1e-15
    sites = siteinds(model[2], N)
    pre_fixed=model[3] #Hamiltonian starts gauge fixed
    #
    #  test lower triangular MPO 
    #
    H=reg_form_MPO(model[1](sites,NNN))
    rng=sweep(H,lr)
    for n in rng
        W=H[n]
        @test is_regular_form(W)
        W1,X,lq=ac_qx(W,lr;cutoff=1e-14)
        @test pre_fixed==check_ortho(W1,lr,eps)
    end
end
    
@testset "QR,QL,LQ,RQ decomposition with rank revealing" begin
    N=10
    NNN=6
    model_kwargs = (hx=0.5, ul=lower)
    eps=2e-15
    rr_cutoff=1e-15
    sites = siteinds("SpinHalf", N)
    #
    #  use lower tri MPO to get some zero pivots for QL and RQ.
    #
    H=make_transIsing_MPO(sites,NNN;model_kwargs...)
    W=H[2]
    r,c=parse_links(W)

    Lind=noncommoninds(inds(W),c)
    Rind=noncommoninds(inds(W),r)
    @assert dim(c)==dim(r)
   
    
    #
    #  use upper tri MPO to get some zero pivots for LQ and QR.
    #
    model_kwargs = (hx=0.5, ul=upper)
    H=make_transIsing_MPO(sites,NNN;model_kwargs...)
    W=H[2]
    r,c=parse_links(W)

    Lind=noncommoninds(inds(W),c)
    Rind=noncommoninds(inds(W),r)
    @assert dim(c)==dim(r)

    #
    #  QR decomp
    #
    Q,R,iq=qr(W,Rind;positive=true,cutoff=rr_cutoff)
    @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
    @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
    @test W ≈ R*Q atol = eps   
    #
    #  LQ decomp
    #
    L,Q,iq=lq(W,r;positive=true,cutoff=rr_cutoff)
    @test dim(c)-dim(iq) == 5 #make sure rank reduction worked.
    @test Q * prime(Q,iq) ≈ δ(Float64, iq, iq') atol = eps
    @test W ≈ L*Q atol = eps
    
end



nothing
