using ITensors
using ITensorMPOCompression
using Test,Printf

import ITensors: tensor
import ITensorMPOCompression: @checkflux, mpoc_checkflux, insert_xblock

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

#
#      1 0 0
#  W = b A 0, For lr=left, slice out c, for lr=right slice out b.
#      d c I
#
function get_bc_block(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state)::ITensor
    @assert hasinds(W,ilb,ilf)
    Dwf,Dwb=dim(ilf),dim(ilb)
    @assert ms.ul==lower
    if ms.lr==left
        bc_block= W[ilb=>Dwb:Dwb,ilf=>2:Dwf-1]
    else
        bc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
    end
    return bc_block
end

#
#      1 0 0
#  W = b A 0, For lr=left, slice out [A], for lr=right slice out [b A].
#      d c I                         [c]                          
#
function get_Abc_block(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state)::Tuple{ITensor,Index,Index}
    Dwf,Dwb=dim(ilf),dim(ilb)
    @assert ms.ul==lower
    if ms.lr==left
        if Dwb>1
            Abc_block= W[ilb=>2:Dwb,ilf=>2:Dwf-1]
        else
            Abc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
        end
    else
        if Dwb>1
            Abc_block= W[ilb=>1:Dwb-1,ilf=>2:Dwf-1]
        else
            Abc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
        end
    end
    ilf1,=inds(Abc_block,tags=tags(ilf))
    ilb1,=inds(Abc_block,tags=tags(ilb)) #New indices.
    return Abc_block,ilf1,ilb1
end
#
#      1 0 0                           
#  W = b A 0, make a new ITensor Wp, 
#      d c I  
#                          [1]
#  for lr=left 1) copy the [b] column from W into Wp, 2) assign Abc into [A]. 3) set bottom corner I
#                          [d]                                           [c]
#                          
#  for lr=right 1) copy the [d c I] row from W into Wp, 2) assign Abc into [b A]. 3) set top corner I
#  
function set_Abc_block(W::ITensor,Abc::ITensor,ilf::Index,ilb::Index,iq::Index,ms::matrix_state)
    is=noncommoninds(W,ilf,ilb)
    @assert hasinds(W,ilf,ilb)
    @assert hasinds(Abc,iq,is...)
    Dwb,Dwf,Dwq=dim(ilb),dim(ilf),dim(iq)+2
    # Todo provide the correct QN("??") space here.  Can we pluck it out of ilf?
    ilqp=redim(iq,Dwq,1) #replaces ilf: 
    Wp=ITensor(0.0,ilb,ilqp,is)
    if ms.lr==left
        Wp[ilb=>1:Dwb,ilqp=>1:1]=W[ilb=>1:Dwb,ilf=>1:1]
        Wp[ilb=>Dwb:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>Dwb:Dwb,ilf=>Dwf:Dwf]
        if Dwb>1
            Wp[ilb=>2:Dwb,ilqp=>2:Dwq-1]=Abc
        else
            Wp[ilb=>1:1,ilqp=>2:Dwq-1]=Abc
        end
    else
        Wp[ilb=>1:1,ilqp=>1:1]=W[ilb=>1:1,ilf=>1:1]
        Wp[ilb=>1:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>1:Dwb,ilf=>Dwf:Dwf]
        if Dwb>1
            Wp[ilb=>1:Dwb-1,ilqp=>2:Dwq-1]=Abc
        else
            Wp[ilb=>1:1,ilqp=>2:Dwq-1]=Abc
        end
    end
    return Wp,ilqp
end


function ac_qx(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state;kwargs...)::Tuple{ITensor,ITensor,Index}
    @checkflux(W)
    @assert hasinds(W,ilf)
    @assert hasinds(W,ilb)
    eps=1e-15
    I=slice(W,ilf=>1,ilb=>1)
    is=inds(I)
    d=dim(is[1])  
    bc=get_bc_block(W,ilf,ilb,ms) #TODO capture removed space
    Dwf,Dwb=dim(ilf),dim(ilb)
    # println("W0")
    # pprint(ilb,W*dag(I)/d,ilf)
    if norm(bc*dag(I))>eps && Dwb>2
        println("------------LWL transform-----------------")
        # @show ms ilb ilf
        Wg=reduce_and_transform(W,ilb,ilf,ms,eps)
        # println("Wg0")
        # pprint(ilb,Wg*dag(I)/d,ilf)
        bcz=get_bc_block(Wg,ilf,ilb,ms)
        @assert norm(bcz*dag(I))<eps
    end
    Abc,ilf1,_=get_Abc_block(W,ilf,ilb,ms)
    @checkflux(Abc)
    if ms.lr==left
        Qinds=noncommoninds(Abc,ilf1)
        Q,R,iq=qr(Abc,Qinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf1
        R,Q,iq=rq(Abc,Rinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    end
    @checkflux(Q)
    @checkflux(R)
    Q*=sqrt(d)
    R/=sqrt(d)
    Wp,iqp=set_Abc_block(W,Q,ilf,ilb,iq,ms) #TODO inject correct removed space
    R=prime(R,ilf1)
    #  TODO fix mimatched spaces when H=non auto MPO.  Need QN()=>1,QN()=>Chi,QN()=>1 space in MPO
    #@show  inds(R) dag(iqp) ilf
    Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    return Wp,Rp,iqp
end

function add_dummy_links!(H::MPO)
    N=length(H)
    ils=map(n->linkind(H,n),1:N-1)
    ts=ITensors.trivial_space(ils[1])
    T=eltype(H[1])
    il0=Index(ts;tags="Link,l=0",dir=dir(dag(ils[1])))
    ilN=Index(ts;tags="Link,l=$N",dir=dir(ils[1]))
    d0=onehot(T, il0 => 1)
    dN=onehot(T, ilN => 1)
    H[1]*=d0
    H[N]*=dN
    return [il0,ils...,ilN],d0,dN
end

function get_identity(W::ITensor,ir::Index,ic::Index)
    
    Id=dim(ir)>1 ? slice(W,ir=>1,ic=>1) : slice(W,ir=>1,ic=>dim(ic))
    d=dims(Id)[1]
    return Id,d
end

function get_bc_block(W0::Matrix,ms::matrix_state)
    @assert ms.ul==lower
    nr,nc=size(W0)
    if ms.lr==left
        bc0=W0[nr,2:nc-1]
    else
        bc0=W0[2:nr-1,1]
    end
    return bc0
end

function extract_blocks(W::ITensor,ir::Index,ic::Index,ms::matrix_state)
    nr,nc=dim(ir),dim(ic)
    @assert nr>1 || nc>1
    𝕀= nr>1 ? slice(W,ir=>1,ic=>1) : slice(W,ir=>1,ic=>nc)
    𝑨= nr>1 && nc>1 ? W[ir=>2:nr-1,ic=>2:nc-1] : nothing
    𝒃= nr>1 ? W[ir=>2:nr-1,ic=>1:1] : nothing
    𝒄= nc>1 ? W[ir=>nr:nr,ic=>2:nc-1] : nothing
    𝒅= nr >1 ? W[ir=>nr:nr,ic=>1:1] : W[ir=>1:1,ic=>nc:nc]

    ird,=inds(𝒅,tags=tags(ir))
    icd,=inds(𝒅,tags=tags(ic))
    if !isnothing(𝒄)
        irc,=inds(𝒄,tags=tags(ir))
        icc,=inds(𝒄,tags=tags(ic))
        𝒄=replaceind(𝒄,irc,ird)
    end
    if !isnothing(𝒃)
        irb,=inds(𝒃,tags=tags(ir))
        icb,=inds(𝒃,tags=tags(ic))
        𝒃=replaceind(𝒃,icb,icd)
    end
    if !isnothing(𝑨)
        irA,=inds(𝑨,tags=tags(ir))
        icA,=inds(𝑨,tags=tags(ic))
        𝑨=replaceinds(𝑨,[irA,icA],[irb,icc])
    end
    return 𝕀,𝑨,𝒃,𝒄,𝒅
end

#  𝕀 𝑨 𝒃 𝒄 𝒅 ⌃ c₀ x0
function gauge_transform!(W::ITensor,ir::Index,ic::Index,tprev::Matrix{Float64},ms::matrix_state)
    𝕀,𝑨,𝒃,𝒄,𝒅=extract_blocks(W,ir,ic,ms)
    if ms.lr==right
        𝒃,𝒄=𝒄,𝒃
    end
    d=𝕀*𝕀
    nr,nc=dim(ir),dim(ic)
    if nr==1 || nc==1
        t=𝒄*dag(𝕀)/d #c0
        𝒄⎖=𝒄-𝕀*t
        𝒅⎖=𝒅
    else
        ict=commonind(𝒃,𝑨,tags="Link")
        irt=commonind(𝒅,𝒄,tags="Link")
        tprevT=ITensor(tprev,irt,ict)

        𝒄₀=𝒄*dag(𝕀)/d
        𝑨₀=𝑨*dag(𝕀)/d
        t=tprevT*𝑨₀+𝒄₀
        𝒄⎖=𝒄+tprevT*𝑨-t*𝕀
        𝒅⎖=𝒅+tprevT*𝒃
    end
    #@show norm(𝒄⎖*𝕀)
    W[ir=>nr:nr,ic=>1:1]=𝒅⎖
    if ms.lr==left
        W[ir=>nr:nr,ic=>2:nc-1]=𝒄⎖
    else
        W[ir=>2:nr-1,ic=>1:1]=𝒄⎖    
    end
       
    return matrix(t)
end


# function back_transform(W::ITensor,ir::Index,ic::Index,t::Vector{Float64},ms::matrix_state)
#     L=insert_xblock(1.0*Matrix(LinearAlgebra.I,n,n),t,ms)
#     Linv=insert_xblock(1.0*Matrix(LinearAlgebra.I,n,n),-t,ms)
# end
models=[
    # [make_transIsing_AutoMPO,"S=1/2"],
    # [make_Heisenberg_AutoMPO,"S=1/2"],
    [make_Hubbard_AutoMPO,"Electron"],
    ]

# @testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns" for model in models, qns in [false]
#     eps=1e-14
#     N=5 #5 sites
#     NNN=3 #Include 2nd nearest neighbour interactions
#     sites = siteinds(model[2],N,conserve_qns=qns);
#     H=model[1](sites,NNN);
#     @show get_Dw(H)
#     state=[isodd(n) ? "Up" : "Dn" for n=1:N]
#     psi=randomMPS(sites,state)
#     E0=inner(psi',H,psi)
#     ms=matrix_state(lower,left)
#     ils,d0,dN=add_dummy_links!(H)
#     ilb=ils[1]
#     for n in sweep(H,left)
#         @show n
#         ilf=linkind(H,n)
#         if dim(ilb)==1
#             W,R,iqp=block_qx(H[n],ilf,ms.ul;orth=ms.lr)
#         else
#             W,R,iqp=ac_qx(H[n],ilf,ilb,ms)
#         end
#         @test norm(H[n]-W*R)<1e-15
#         @test check_ortho(W,ms)
#         H[n]=W
#         H[n+1]=R*H[n+1]
#         ilb=dag(iqp)
#     end
#     @test check_ortho(H,ms)
#     qns && show_directions(H)
#     H[1]*=dag(d0)
#     H[N]*=dag(dN)
#     E1=inner(psi',H,psi)
#     @test E0 ≈ E1 atol = eps

#     # ils,d0,dN=add_dummy_links!(H)
#     # ms=matrix_state(lower,right)
#     # ilb=ils[N+1]
#     # for n in sweep(H,right)
#     #     ilf=dag(linkind(H,n-1))
#     #     W,R,iqp=ac_qx(H[n],ilf,ilb,ms)
#     #     @test norm(H[n]-W*R)<1e-15
#     #     @test check_ortho(W,ms)
#     #     H[n]=W
#     #     H[n-1]=R*H[n-1]
#     #     ilb=dag(iqp)
#     # end
#     # @test check_ortho(H,ms)
#     # qns && show_directions(H)
#     # H[1]*=dag(d0)
#     # H[N]*=dag(dN)
#     # E2=inner(psi',H,psi)
#     # @test E0 ≈ E2 atol = eps
# end

@testset "Gauge transform rectangular W" begin
    eps=1e-14
    
    N=10 #5 sites
    NNN=5 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=false)
    H=make_Hubbard_AutoMPO(sites,NNN)
    ils,d0,dN=add_dummy_links!(H)

    ms=matrix_state(lower,left)
    ir=ils[1]
    t=Matrix{Float64}(undef,1,1)
    for n in 1:N-1
        W=copy(H[n])
        ic =linkind(H,n)
        I,d=get_identity(W,ir,ic)    
        t=gauge_transform!(W,ir,ic,t,ms)
        c=get_bc_block(W,ic,ir,ms)
        c0=c*dag(I)/d
        @test norm(c0)<eps
        ir=ic
    end

    ms=matrix_state(lower,right)
    ic=ils[N+1]
    t=Matrix{Float64}(undef,1,1)
    for n in N:-1:N-2
        W=copy(H[n])
        ir =linkind(H,n-1)
        I,d=get_identity(W,ir,ic)    
        t=gauge_transform!(W,ir,ic,t,ms)
        c=get_bc_block(W,ir,ic,ms)
        c0=c*dag(I)/d
        @test norm(c0)<eps
        ic=ir
    end
end

# @testset "Extract blocks" begin
#     eps=1e-15
#     N=5 #5 sites
#     NNN=2 #Include 2nd nearest neighbour interactions
#     sites = siteinds("Electron",N,conserve_qns=false)
#     d=dim(inds(sites[1])[1])
#     H=make_Hubbard_AutoMPO(sites,NNN)
#     ils,d0,dN=add_dummy_links!(H)

#     ms=matrix_state(lower,left)
#     ir,ic=ils[1],linkind(H,1)
#     nr,nc=dim(ir),dim(ic)
#     W=H[1]
#     #pprint(W)
#     𝕀,𝑨,𝒃,𝒄,𝒅=extract_blocks(W,ir,ic,ms)
#     @test norm(matrix(𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
#     @test isnothing(𝑨)    
#     @test isnothing(𝒃)  
#     @test norm(array(𝒅)-array(W[ir=>1:1,ic=>nc:nc]))<eps
#     @test norm(array(𝒄)-array(W[ir=>nr:nr,ic=>2:nc-1]))<eps

#     W=H[N]
#     ir,ic=linkind(H,N-1),ils[N+1]
#     nr,nc=dim(ir),dim(ic)
#     #pprint(W)
#     𝕀,𝑨,𝒃,𝒄,𝒅=extract_blocks(W,ir,ic,ms)
#     @test norm(matrix(𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
#     @test isnothing(𝑨)    
#     @test isnothing(𝒄)  
#     @test norm(array(𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
#     @test norm(array(𝒃)-array(W[ir=>2:nr-1,ic=>1:1]))<eps

#     W=H[2]
#     ir,ic=linkind(H,1),linkind(H,2)
#     nr,nc=dim(ir),dim(ic)
#     𝕀,𝑨,𝒃,𝒄,𝒅=extract_blocks(W,ir,ic,ms)
#     @test norm(matrix(𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
#     @test norm(array(𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
#     @test norm(array(𝒃)-array(W[ir=>2:nr-1,ic=>1:1]))<eps
#     @test norm(array(𝒄)-array(W[ir=>nr:nr,ic=>2:nc-1]))<eps
#     @test norm(array(𝑨)-array(W[ir=>2:nr-1,ic=>2:nc-1]))<eps
# end

nothing