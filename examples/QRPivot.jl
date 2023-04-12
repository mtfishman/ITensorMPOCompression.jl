using ITensors
using ITensorMPOCompression
using Test,Printf

import ITensors: tensor
import ITensorMPOCompression: @checkflux, mpoc_checkflux, insert_xblock

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

#-------------------------------------------------------------------------------
#
#  These functions are the dev. testing only.
#
function calculate_ts(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    ts=Vector{Float64}[]
    ir=ils[1]
    tprev=zeros(1)
    push!(ts,tprev)
    for n in eachindex(H)
        ic=ils[n+1]
        Wb=extract_blocks(H[n],ir,ic,ms;all=true,fix_inds=true)
        𝕀,𝑨,𝒃,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝒃,Wb.𝒄,Wb.𝒅
        dh=𝕀*𝕀
        nr,nc=dim(ir),dim(ic)
        if nr==1 
            c0=𝒄*dag(𝕀)/dh
            t=matrix(c0)[:,1] #c0
        elseif nc==1
            t=zeros(1)
        else
           ict=commonind(𝒃,𝑨,tags="Link")
            irt=commonind(𝒅,𝒄,tags="Link")
            tprevT=ITensor(tprev,irt,ict)

            𝒄₀=𝒄*dag(𝕀)/dh
            𝑨₀=𝑨*dag(𝕀)/dh
            t=matrix(tprevT*𝑨₀+𝒄₀)[1,:]
        end
        #@show t
        push!(ts,t)
        tprev=t
        ir=ic
    end
    return ts
end

function calculate_Ls(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    Ls=Matrix{Float64}[]
    Linvs=Matrix{Float64}[]
    ts=calculate_ts(H,ils,ms)
    @assert length(ts)==length(ils)
    for n in eachindex(ils)
        ic=ils[n]
        Dwc=dim(ic)
        if Dwc==1
            L=1.0*Matrix(LinearAlgebra.I,Dwc,Dwc)
            Linv=1.0*Matrix(LinearAlgebra.I,Dwc,Dwc)
        else
            @assert Dwc==size(ts[n],1)+2
            L=insert_xblock(1.0*Matrix(LinearAlgebra.I,Dwc,Dwc),ts[n],ms)
            Linv=insert_xblock(1.0*Matrix(LinearAlgebra.I,Dwc,Dwc),-ts[n],ms)
        end
        push!(Ls,L)
        push!(Linvs,Linv)
    end
    return Ls,Linvs
end

function apply_Ls!(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    Ls,Linvs=calculate_Ls(H,ils,ms)
    ir=ils[1]
    for n in eachindex(H)
        ic=ils[n+1]
        @assert hasinds(H[n],ir,ic)
        LT=ITensor(Ls[n],ir',ir)
        LinvT=ITensor(Linvs[n+1],ic,ic')
        Wp=noprime(LT*H[n]*LinvT,tags="Link")
        H[n]=Wp
        ir=ic
    end
end

#-------------------------------------------------------------------------------
#
#  Make all ITensors in H of order 4.  This simplifies the code.
#
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

#-------------------------------------------------------------------------------
#
#  Blocking functions
#
#
#  Decisions: 1) Use ilf,ilb==forward,backward  or ir,ic=row,column ?
#             2) extract_blocks gets everything.  Should it defer to get_bc_block for b and c?
#   may best to define W as
#               ul=lower         ul=upper
#                1 0 0           1 b d
#     lr=left    b A 0           0 A c
#                d c I           0 0 I
#
#                1 0 0           1 c d
#     lr=right   c A 0           0 A b
#                d b I           0 0 I
#
#  Use kwargs in extract_blocks so caller can choose what they need. Default is c only
#
mutable struct regform_blocks
    𝕀::Union{ITensor,Nothing}
    𝑨::Union{ITensor,Nothing}
    𝑨𝒄::Union{ITensor,Nothing}
    𝒃::Union{ITensor,Nothing}
    𝒄::Union{ITensor,Nothing}
    𝒅::Union{ITensor,Nothing}
    irA::Index
    icA::Index
    irAc::Index
    icAc::Index
    irb::Index
    icb::Index
    irc::Index
    icc::Index
    ird::Index
    icd::Index    
    regform_blocks()=new(nothing,nothing,nothing,nothing,nothing,nothing)
end

d(rfb::regform_blocks)::Float64=scalar(rfb.𝕀*rfb.𝕀)
b0(rfb::regform_blocks)::ITensor=rfb.𝒃*dag(rfb.𝕀)/d(rfb)
c0(rfb::regform_blocks)::ITensor=rfb.𝒄*dag(rfb.𝕀)/d(rfb)

#  Use recognizably distinct UTF symbols for operators, and op valued vectors and matrices: 𝕀 𝑨 𝒃 𝒄 𝒅 ⌃ c₀ x0
function extract_blocks(W::ITensor,ir::Index,ic::Index,ms::matrix_state;all=false,c=true,b=false,d=false,A=false,Ac=false,I=true,fix_inds=false)::regform_blocks
    @assert hasinds(W,ir,ic)
    @assert tags(ir)!=tags(ic)
    @assert plev(ir)==0
    @assert plev(ic)==0

    #@show ir ic c b ms
    
    if ms.ul==uppercase
        ir,ic=ic,ir #transpose
    end
    nr,nc=dim(ir),dim(ic)
    @assert nr>1 || nc>1
    if all #does not include Ac
        A=b=c=d=I=true
    end
    if fix_inds && !d
        @warn "extract_blocks: fix_inds requires d=true."
        d=true
    end
    if ms.lr==right
        b,c=c,b
    end

    A = A && (nr>1 && nc>1)
    b = b &&  nr>1 
    c = c &&  nc>1

  
    rfb=regform_blocks()
    I && (rfb.𝕀= nr>1 ? slice(W,ir=>1,ic=>1) : slice(W,ir=>1,ic=>nc))
   
    if A
        rfb.𝑨= W[ir=>2:nr-1,ic=>2:nc-1]
        rfb.irA,=inds(rfb.𝑨,tags=tags(ir))
        rfb.icA,=inds(rfb.𝑨,tags=tags(ic))
    end
    if Ac
        if ms.lr==left
            if nr>1
                rfb.𝑨𝒄= W[ir=>2:nr,ic=>2:nc-1]#W[ilb=>2:Dwb,ilf=>2:Dwf-1]
            else
                rfb.𝑨𝒄= W[ir=>1:1,ic=>2:nc-1]#W[ilb=>1:1,ilf=>2:Dwf-1]
            end
        else
            if nc>1
                rfb.𝑨𝒄= W[ir=>2:nr-1,ic=>1:nc-1]#W[ilb=>1:Dwb-1,ilf=>2:Dwf-1]
            else
                rfb.𝑨𝒄= W[ir=>2:nr-1,ic=>1:1]#W[ilb=>1:1,ilf=>2:Dwf-1]
            end
        end
        #rfb.𝑨𝒄= ms.lr == left ?  W[ir=>2:nr,ic=>2:nc-1] :  W[ir=>2:nr-1,ic=>1:nc-1]
        rfb.irAc,=inds(rfb.𝑨𝒄,tags=tags(ir))
        rfb.icAc,=inds(rfb.𝑨𝒄,tags=tags(ic))
    end
    if b
        rfb.𝒃= W[ir=>2:nr-1,ic=>1:1]
        rfb.irb,=inds(rfb.𝒃,tags=tags(ir))
        rfb.icb,=inds(rfb.𝒃,tags=tags(ic))
    end
    if c
        rfb.𝒄= W[ir=>nr:nr,ic=>2:nc-1]
        rfb.irc,=inds(rfb.𝒄,tags=tags(ir))
        rfb.icc,=inds(rfb.𝒄,tags=tags(ic))
    end
    if d
        rfb.𝒅= nr >1 ? W[ir=>nr:nr,ic=>1:1] : W[ir=>1:1,ic=>1:1]
        rfb.ird,=inds(rfb.𝒅,tags=tags(ir))
        rfb.icd,=inds(rfb.𝒅,tags=tags(ic))
    end

    if fix_inds
        if !isnothing(rfb.𝒄)
            rfb.𝒄=replaceind(rfb.𝒄,rfb.irc,rfb.ird)
        end
        if !isnothing(rfb.𝒃)
            rfb.𝒃=replaceind(rfb.𝒃,rfb.icb,rfb.icd)
        end
        if !isnothing(rfb.𝑨)
            rfb.𝑨=replaceinds(rfb.𝑨,[rfb.irA,rfb.icA],[rfb.irb,rfb.icc])
        end
    end
    if ms.lr==right
        rfb.𝒃,rfb.𝒄=rfb.𝒄,rfb.𝒃
    end
    #@show c b 
    return rfb
end

function set_Abc_block(W::ITensor,Abc::ITensor,ilf::Index,ilb::Index,iq::Index,ms::matrix_state)
    is=noncommoninds(W,ilf,ilb)
    @assert hasinds(W,ilf,ilb)
    @assert hasinds(Abc,iq,is...)
    Dwb,Dwf,Dwq=dim(ilb),dim(ilf),dim(iq)+2
    # Todo provide the correct QN("??") space here.  Can we pluck it out of ilf?
    ilqp=redim(iq,Dwq,1) #replaces ilf: 
    
    if ms.lr==left
        Wp=ITensor(0.0,ilb,ilqp,is)
        Wp[ilb=>1:Dwb,ilqp=>1:1]=W[ilb=>1:Dwb,ilf=>1:1] #left column
        Wp[ilb=>Dwb:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>Dwb:Dwb,ilf=>Dwf:Dwf] #bottom right corner
        if Dwb>1
            Wp[ilb=>2:Dwb,ilqp=>2:Dwq-1]=Abc
        else
            Wp[ilb=>1:1,ilqp=>2:Dwq-1]=Abc
        end
    else
        Wp=ITensor(0.0,ilqp,ilf,is)
        Wp[ilqp=>1:1,ilf=>1:1]=W[ilb=>1:1,ilf=>1:1] #Top left corner
        Wp[ilqp=>Dwq:Dwq,ilf=>1:Dwf]=W[ilb=>Dwb:Dwb,ilf=>1:Dwf] #Bottom row
        if Dwf>1
            Wp[ilqp=>2:Dwq-1,ilf=>1:Dwf-1]=Abc
        else
            Wp[ilqp=>2:Dwq-1,ilf=>1:1]=Abc
        end
    end
    return Wp,ilqp
end

#-------------------------------------------------------------------------------
#
#  Gauge fixing functions.  In this conext gauge fixing means setting b₀=<𝒃,𝕀> && c₀=<𝒄,𝕀> to zero
#
function is_gauge_fixed(W::ITensor,ir::Index{T},ic::Index{T},ul::reg_form,eps::Float64;b=true,c=true)::Bool where {T}
    igf=true
    ms=matrix_state(ul,left)
    Wb=extract_blocks(W,ir,ic,ms;c=true,b=true)
    if b && dim(ir)>1
        igf=igf && norm(b0(Wb))<eps
    end
    if c && dim(ic)>1
        igf=igf && norm(c0(Wb))<eps
    end
    return igf
end

function is_gauge_fixed(H::MPO,ils::Vector{Index{T}},ul::reg_form,eps::Float64;kwargs...)::Bool where {T}
    igf=true
    ir=ils[1]   
    for n in eachindex(H)
        ic=ils[n+1]
        igf = igf && is_gauge_fixed(H[n],ir,ic,ul,eps;kwargs...)
        if !igf
            break
        end
        ir=ic
    end
    return igf
end

function gauge_transform!(W::ITensor,ir::Index,ic::Index,tprev::Matrix{Float64},ms::matrix_state)
    Wb=extract_blocks(W,ir,ic,ms;all=true,fix_inds=true)
    𝕀,𝑨,𝒃,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝒃,Wb.𝒄,Wb.𝒅
    dh=𝕀*𝕀
    nr,nc=dim(ir),dim(ic)
    𝒄⎖=nothing
    if ms.lr==left
        if nr==1 
            t=𝒄*dag(𝕀)/dh #c0
            𝒄⎖=𝒄-𝕀*t
            𝒅⎖=𝒅
        elseif nc==1
            il=commonind(𝒃,𝒅,tags="Link")
            ict=noncommonind(𝒅,il,tags="Link")
            irt=noncommonind(𝒃,𝒅,tags="Link")
            tprevT=ITensor(tprev,irt,ict)
            𝒅⎖=𝒅+tprevT*𝒃
            t=tprevT
        else
            ict=commonind(𝒃,𝑨,tags="Link")
            irt=commonind(𝒅,𝒄,tags="Link")
            tprevT=ITensor(tprev,irt,ict)

            𝒄₀=𝒄*dag(𝕀)/dh
            𝑨₀=𝑨*dag(𝕀)/dh
            t=tprevT*𝑨₀+𝒄₀
            𝒄⎖=𝒄+tprevT*𝑨-t*𝕀
            𝒅⎖=𝒅+tprevT*𝒃
        end
    else
        if nc==1 
            t=𝒄*dag(𝕀)/dh #c0
            𝒄⎖=𝒄-𝕀*t
            𝒅⎖=𝒅
        elseif nr==1
            il=commonind(𝒃,𝒅,tags="Link")
            ict=noncommonind(𝒅,il,tags="Link")
            irt=noncommonind(𝒃,𝒅,tags="Link")
            tprevT=ITensor(tprev,irt,ict)
            𝒅⎖=𝒅+tprevT*𝒃
            t=tprevT
        else
            ict=commonind(𝒃,𝑨,tags="Link")
            irt=commonind(𝒅,𝒄,tags="Link")
            tprevT=ITensor(tprev,irt,ict)

            𝒄₀=𝒄*dag(𝕀)/dh
            𝑨₀=𝑨*dag(𝕀)/dh
            t=tprevT*𝑨₀+𝒄₀
            𝒄⎖=𝒄+tprevT*𝑨-t*𝕀
            𝒅⎖=𝒅+tprevT*𝒃
        end
    end
    W[ir=>nr:nr,ic=>1:1]=𝒅⎖
    if !isnothing(𝒄⎖)
        if ms.lr==left 
            W[ir=>nr:nr,ic=>2:nc-1]=𝒄⎖
        else
            W[ir=>2:nr-1,ic=>1:1]=𝒄⎖    
        end
    end
       
    return matrix(t)
end

function gauge_transform!(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    N=length(H)
    t=Matrix{Float64}(undef,1,1)
    ir=ils[1]
    for n in 1:N
        ic=ils[n+1]
        @assert hasinds(H[n],ir,ic)
        t=gauge_transform!(H[n],ir,ic,t,matrix_state(ms.ul,left))
        ir=ic
    end
    t=Matrix{Float64}(undef,1,1)
    ic=ils[N+1]
    for n in N:-1:1
        ir=ils[n]
        @assert hasinds(H[n],ir,ic)
        t=gauge_transform!(H[n],ir,ic,t,matrix_state(ms.ul,right))
        ic=ir
    end
end

#-------------------------------------------------------------------------------
#
#  block qx and orthogonalization of the vcat(𝑨,𝒄) and hcat(𝒃,𝑨) blocks.
#
function ac_qx(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state;kwargs...)
    @checkflux(W)
    @assert hasinds(W,ilf)
    @assert hasinds(W,ilb)
    rfb=extract_blocks(W,ilb,ilf,ms;Ac=true)
    Abc=rfb.𝑨𝒄
    ilf1 = ms.lr==left ? rfb.icAc : rfb.irAc
    dh=d(rfb)
    @checkflux(Abc)
    if ms.lr==left
        Qinds=noncommoninds(Abc,ilf1)
        Q,R,iq=qr(Abc,Qinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf1
        R,Q,iq=rq(Abc,Rinds;positive=true,rr_cutoff=1e-14,tags=tags(ilb))
    end
    @checkflux(Q)
    @checkflux(R)
    Q*=sqrt(dh)
    R/=sqrt(dh)
    Wp,iqp=set_Abc_block(W,Q,ilf,ilb,iq,ms) #TODO inject correct removed space
    R=prime(R,ilf1)
    #  TODO fix mimatched spaces when H=non auto MPO.  Need QN()=>1,QN()=>Chi,QN()=>1 space in MPO
    #@show  inds(R) dag(iqp) ilf
    if ms.lr==left
        Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    else
        Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilb'))
    end
    return Wp,Rp,iqp
end

function ac_orthogonalize!(H::MPO,ils::Vector{Index{T}},ms::matrix_state,eps::Float64) where {T}
    if !is_gauge_fixed(H,ils,ms.ul,eps)
        gauge_transform!(H,ils,ms)
    end
    if ms.lr==left
        ir=ils[1]
        for n in sweep(H,ms.lr)
            ic=ils[n+1]
            W,R,iqp=ac_qx(H[n],ic,ir,ms)
            @test norm(H[n]-W*R)<1e-14
            @test check_ortho(W,ms)
            H[n]=W
            H[n+1]=R*H[n+1]
            ils[n+1]=ir=dag(iqp)
        end
    else
        N=length(H)
        ic=ils[N+1]
        for n in sweep(H,ms.lr)
            ir=ils[n]
            W,R,iqp=ac_qx(H[n],ic,ir,ms)
            @test norm(H[n]-W*R)<1e-14
            @test check_ortho(W,ms)
            H[n]=W
            H[n-1]=R*H[n-1]
            ils[n]=ic=dag(iqp)
        end
    end
end







models=[
    # [make_transIsing_AutoMPO,"S=1/2"],
    # [make_Heisenberg_AutoMPO,"S=1/2"],
    [make_Hubbard_AutoMPO,"Electron"],
    ]

@testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns" for model in models, qns in [false]
    eps=1e-14
    N=10 #5 sites
    NNN=7 #Include 2nd nearest neighbour interactions
    sites = siteinds(model[2],N,conserve_qns=qns);
    H=model[1](sites,NNN);
    @show get_Dw(H)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)

    ils,d0,dN=add_dummy_links!(H)
    #
    #  Left->right sweep
    #
    ms=matrix_state(lower,left)
    @test !is_gauge_fixed(H,ils,ms.ul,eps) #Now everything should be fixed
    ac_orthogonalize!(H,ils,ms,eps)
    @test check_ortho(H,ms)
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Now everything should be fixed
    qns && show_directions(H)
    #
    #  Expectation value check.
    #
    H[1]*=dag(d0)
    H[N]*=dag(dN)
    E1=inner(psi',H,psi)
    @test E0 ≈ E1 atol = eps
    #
    #  Right->left sweep
    #
    ils,d0,dN=add_dummy_links!(H)
    ms=matrix_state(lower,right)
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Should still be gauge fixed
    ac_orthogonalize!(H,ils,ms,eps)
    @test check_ortho(H,ms)
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Should still be gauge fixed
    qns && show_directions(H)
    #
    #  Expectation value check.
    #
    H[1]*=dag(d0)
    H[N]*=dag(dN)
    E2=inner(psi',H,psi)
    @test E0 ≈ E2 atol = eps
end

@testset "Gauge transform rectangular W" begin
    eps=1e-14
    
    N=5 #5 sites
    NNN=2 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=false)
    H=make_Hubbard_AutoMPO(sites,NNN)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)
    
    ils,d0,dN=add_dummy_links!(H)

    ms=matrix_state(lower,left)
    @test !is_gauge_fixed(H,ils,ms.ul,eps)

    H_lwl=deepcopy(H)
    @test !is_gauge_fixed(H_lwl,ils,ms.ul,eps)
    apply_Ls!(H_lwl,ils,ms)
    @test is_gauge_fixed(H_lwl,ils,ms.ul,eps,b=false)
    @test !is_gauge_fixed(H_lwl,ils,ms.ul,eps,c=false)
    H_g=deepcopy(H) 
    #   
    #  Left->Right sweep doing gauge c0==0 transforms
    #
    ir=ils[1]
    t=Matrix{Float64}(undef,1,1)
    for n in 1:N
        ic =ils[n+1]
        t=gauge_transform!(H[n],ir,ic,t,ms)
        @test norm(H_lwl[n]-H[n])<eps
        @test is_gauge_fixed(H[n],ir,ic,ms.ul,eps;b=false)    
        ir=ic
    end
    @test is_gauge_fixed(H,ils,ms.ul,eps,b=false)
    @test !is_gauge_fixed(H,ils,ms.ul,eps,c=false)
    #
    #  Check that the energy expectation is invariant.
    #   
    He=copy(H)
    He[1]*=dag(d0)
    He[N]*=dag(dN)
    E1=inner(psi',He,psi)
    @test E0 ≈ E1 atol = eps
    #
    # Do a full gauge transform on Hg   
    #
    @test !is_gauge_fixed(H,ils,ms.ul,eps) #b0's not done yet
    @test !is_gauge_fixed(H_g,ils,ms.ul,eps,b=false) #only check the c0s
    @test !is_gauge_fixed(H_g,ils,ms.ul,eps,c=false) #only check the b0s
    gauge_transform!(H_g,ils,ms)
    @test !is_gauge_fixed(H,ils,ms.ul,eps) #deepcopy ensures we didn't just (inadvertently) gauge fix H as well
    @test is_gauge_fixed(H_g,ils,ms.ul,eps)
    #
    #  Sweep right to left abd gauge all the b0's==0 .
    #
    ms=matrix_state(lower,right)
    ic=ils[N+1]
    t=Matrix{Float64}(undef,1,1)
    for n in N:-1:1
        W=H[n]
        ir =ils[n]
        t=gauge_transform!(W,ir,ic,t,ms)
        @test norm(H_g[n]-W)<eps
        @test is_gauge_fixed(H[n],ir,ic,ms.ul,eps;b=false)    
        @test is_gauge_fixed(H[n],ir,ic,ms.ul,eps;c=false)    
        ic=ir
    end
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Now everything should be fixed
    
 
    H[1]*=dag(d0)
    H[N]*=dag(dN)
    E2=inner(psi',H,psi)
    @test E0 ≈ E2 atol = eps
end

@testset "Extract blocks" begin
    eps=1e-15
    N=5 #5 sites
    NNN=2 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=false)
    d=dim(inds(sites[1])[1])
    H=make_Hubbard_AutoMPO(sites,NNN)
    ils,d0,dN=add_dummy_links!(H)

    ms=matrix_state(lower,left)
    ir,ic=ils[1],linkind(H,1)
    nr,nc=dim(ir),dim(ic)
    W=H[1]
    #pprint(W)
    rfb=extract_blocks(W,ir,ic,ms;all=true)
    @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
    @test isnothing(rfb.𝑨)    
    @test isnothing(rfb.𝒃)  
    @test norm(array(rfb.𝒅)-array(W[ir=>1:1,ic=>1:1]))<eps
    @test norm(array(rfb.𝒄)-array(W[ir=>nr:nr,ic=>2:nc-1]))<eps
    
    W=H[N]
    ir,ic=linkind(H,N-1),ils[N+1]
    nr,nc=dim(ir),dim(ic)
    #pprint(W)
    rfb=extract_blocks(W,ir,ic,ms;all=true,fix_inds=true)
    @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
    @test isnothing(rfb.𝑨)    
    @test isnothing(rfb.𝒄)  
    @test norm(array(rfb.𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
    @test norm(array(rfb.𝒃)-array(W[ir=>2:nr-1,ic=>1:1]))<eps
    

    W=H[2]
    ir,ic=linkind(H,1),linkind(H,2)
    nr,nc=dim(ir),dim(ic)
    rfb=extract_blocks(W,ir,ic,ms;all=true,fix_inds=true,Ac=true)
    @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
    @test norm(array(rfb.𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
    @test norm(array(rfb.𝒃)-array(W[ir=>2:nr-1,ic=>1:1]))<eps
    @test norm(array(rfb.𝒄)-array(W[ir=>nr:nr,ic=>2:nc-1]))<eps
    @test norm(array(rfb.𝑨)-array(W[ir=>2:nr-1,ic=>2:nc-1]))<eps
    @test norm(array(rfb.𝑨𝒄)-array(W[ir=>2:nr,ic=>2:nc-1]))<eps

    #@show typeof(scalar(rfb.𝕀*rfb.𝕀))

end

@testset "Calculate t's, L's and Linv's" begin
    N=10 #5 sites
    NNN=5 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=false)
    H=make_Hubbard_AutoMPO(sites,NNN)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)

    ils,d0,dN=add_dummy_links!(H)
    ms=matrix_state(lower,left)
    apply_Ls!(H,ils,ms) #only gets the c0's, not the b0's
    #@test is_gauge_fixed(H,ils,ms.ul,1e-15) 
   
    H[1]*=dag(d0)
    H[N]*=dag(dN)
    E2=inner(psi',H,psi)
    @test E0 ≈ E2 atol = 1e-15
end


nothing