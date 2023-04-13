using ITensors
using ITensorMPOCompression
using Test,Printf,Revise

import ITensors: tensor
import ITensorMPOCompression: @checkflux, mpoc_checkflux, insert_xblock

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

#-------------------------------------------------------------------------------
#
#  These functions are the dev. testing only.
#
function calculate_ts(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    ts=Vector{Float64}[]
    ir=dag(ils[1])
    tprev=zeros(1)
    push!(ts,tprev)
    for n in eachindex(H)
        ic=ils[n+1]
        Wb=extract_blocks(H[n],ir,ic,ms;all=true,fix_inds=true)
        𝕀,𝑨,𝒃,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝒃,Wb.𝒄,Wb.𝒅
        dh=d(Wb)
        nr,nc=dim(ir),dim(ic)
        if nr==1 
            c0=𝒄*dag(𝕀)/dh
            t=matrix(c0)[:,1] #c0
        elseif nc==1
            t=zeros(1)
        else
           ict=commonind(𝒃,𝑨,tags="Link")
            irt=commonind(𝒅,𝒄,tags="Link")
            tprevT=ITensor(tprev,irt,dag(ict))

            𝒄₀=𝒄*dag(𝕀)/dh
            𝑨₀=𝑨*dag(𝕀)/dh
            t=matrix(tprevT*𝑨₀+𝒄₀)[1,:]
        end
        #@show t
        push!(ts,t)
        tprev=t
        ir=dag(ic)
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
        #display(L)
        push!(Ls,L)
        push!(Linvs,Linv)
    end
    return Ls,Linvs
end

function apply_Ls!(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    Ls,Linvs=calculate_Ls(H,ils,ms)
    ir=dag(ils[1])
    for n in eachindex(H)
        ic=ils[n+1]
        @assert hasinds(H[n],ir,ic)
        LT=ITensor(Ls[n],ir',ir)
        LinvT=ITensor(Linvs[n+1],ic,ic')
        Wp=noprime(LT*H[n]*LinvT,tags="Link")
        H[n]=Wp
        ir=dag(ic)
    end
end

#-------------------------------------------------------------------------------
#
#  Make all ITensors in H of order 4.  This simplifies the code.
#
function add_edge_links!(H::MPO)
    N=length(H)
    ils=map(n->linkind(H,n),1:N-1)
    ts=ITensors.trivial_space(ils[1])
    T=eltype(H[1])
    il0=Index(ts;tags="Link,l=0",dir=dir(ils[1]))
    ilN=Index(ts;tags="Link,l=$N",dir=dir(ils[1]))
    d0=onehot(T, dag(il0) => 1)
    dN=onehot(T, ilN => 1)
    H[1]*=d0
    H[N]*=dN
    return [il0,ils...,ilN],d0,dN
end

function remove_edge_links!(H::MPO,d0::ITensor,dN::ITensor)
    @assert has_edge_links(H)
    N=length(H)
    H[1]*=dag(d0)
    H[N]*=dag(dN)
    @assert !has_edge_links(H)
end

function remove_edge_links(H::MPO,d0::ITensor,dN::ITensor)::MPO
    @assert has_edge_links(H)
    H1=copy(H)
    remove_edge_links!(H1,d0,dN)
    @assert has_edge_links(H) #did the copy work?
    @assert !has_edge_links(H1)
    return H1
end

function has_edge_links(H::MPO)::Bool
    N=length(H)
    @assert order(H[1])==order(H[N])
    return order(H[1])==4
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
    irA::Union{Index,Nothing}
    icA::Union{Index,Nothing}
    irAc::Union{Index,Nothing}
    icAc::Union{Index,Nothing}
    irb::Union{Index,Nothing}
    icb::Union{Index,Nothing}
    irc::Union{Index,Nothing}
    icc::Union{Index,Nothing}
    ird::Union{Index,Nothing}
    icd::Union{Index,Nothing}    
    regform_blocks()=new(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing)
end

d(rfb::regform_blocks)::Float64=scalar(rfb.𝕀*dag(rfb.𝕀))
b0(rfb::regform_blocks)::ITensor=rfb.𝒃*dag(rfb.𝕀)/d(rfb)
c0(rfb::regform_blocks)::ITensor=rfb.𝒄*dag(rfb.𝕀)/d(rfb)
A0(rfb::regform_blocks)::ITensor=rfb.𝑨*dag(rfb.𝕀)/d(rfb)

#  Use recognizably distinct UTF symbols for operators, and op valued vectors and matrices: 𝕀 𝑨 𝒃 𝒄 𝒅 ⌃ c₀ x0 𝑨𝒄
function extract_blocks(W::ITensor,ir::Index,ic::Index,ms::matrix_state;all=false,c=true,b=false,d=false,A=false,Ac=false,I=true,fix_inds=false)::regform_blocks
    @assert hasinds(W,ir,ic)
    @assert tags(ir)!=tags(ic)
    @assert plev(ir)==0
    @assert plev(ic)==0
    @assert !hasqns(ir) || dir(W,ir)==dir(ir)
    @assert !hasqns(ic) || dir(W,ic)==dir(ic)
    if ms.ul==upper
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
    if ms==matrix_state(lower,right) || ms==matrix_state(upper,left)
        b,c=c,b
    end

    A = A && (nr>1 && nc>1)
    b = b &&  nr>1 
    c = c &&  nc>1

  
    rfb=regform_blocks()
    if ms.ul==lower
        I && (rfb.𝕀= nr>1 ? slice(W,ir=>1,ic=>1) : slice(W,ir=>1,ic=>nc))
    else
        I && (rfb.𝕀= nr>1 ? slice(W,ir=>1,ic=>1) : slice(W,ir=>1,ic=>nc))
    end
   
    if A
        rfb.𝑨= W[ir=>2:nr-1,ic=>2:nc-1]
        rfb.irA,=inds(rfb.𝑨,tags=tags(ir))
        rfb.icA,=inds(rfb.𝑨,tags=tags(ic))
    end
    if Ac
        if ms==matrix_state(lower,left) || ms==matrix_state(upper,right)
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
    if ms==matrix_state(lower,right) || ms==matrix_state(upper,left)
        rfb.𝒃,rfb.𝒄=rfb.𝒄,rfb.𝒃
        rfb.irb,rfb.irc=rfb.irc,rfb.irb
        rfb.icb,rfb.icc=rfb.icc,rfb.icb
    end
    return rfb
end

function set_𝑨𝒄_block(W::ITensor,𝑨𝒄::ITensor,ilb::Index,ilf::Index,iq::Index,ms::matrix_state)
    is=noncommoninds(W,ilb,ilf)
    @assert hasinds(W,ilb,ilf)
    @assert hasinds(𝑨𝒄,iq,is...)
    Dwb,Dwf=dim(ilb),dim(ilf)
    # Todo provide the correct QN("??") space here.  Can we pluck it out of ilf?
    Dwq=dim(iq)+2
    ilqp=redim(iq,Dwq,1) #replaces ilf: 
    Wp=ITensor(0.0,ilb,ilqp,is)
    #
    #  We need to preserve some blocks outside of Ac from the old MPO tensor.
    #
    if ms.ul==lower
        if ms.lr==left
            Wp[ilb=>Dwb:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>Dwb:Dwb,ilf=>Dwf:Dwf] #bottom right corner
            Wp[ilb=>1:Dwb,ilqp=>1:1]=W[ilb=>1:Dwb,ilf=>1:1] #left column
        else
            Wp[ilb=>1:1,ilqp=>1:1]=W[ilb=>1:1,ilf=>1:1] #Top left corner
            Wp[ilb=>1:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>1:Dwb,ilf=>Dwf:Dwf] #Bottom row
        end
    else
        if ms.lr==left
            Wp[ilb=>1:1,ilqp=>1:1]=W[ilb=>1:1,ilf=>1:1] #Top left corner
            Wp[ilb=>1:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>1:Dwb,ilf=>Dwf:Dwf] #right column
        else
            Wp[ilb=>Dwb:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>Dwb:Dwb,ilf=>Dwf:Dwf] # top left corner
            Wp[ilb=>1:Dwb,ilqp=>1:1]=W[ilb=>1:Dwb,ilf=>1:1] #Top row
        end
    end
    #
    #  Fill in the 𝑨𝒄 block
    #
    if ms.ul==lower
        ac_range= Dwb>1 ? (ms.lr==left ? (2:Dwb) : (1:Dwb-1)) : 1:1
    else
        ac_range= Dwb>1 ? (ms.lr==left ? (1:Dwb-1) : (2:Dwb)) : 1:1
    end
    Wp[ilb=>ac_range,ilqp=>2:Dwq-1]=𝑨𝒄
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
    ir=dag(ils[1])   
    for n in eachindex(H)
        ic=ils[n+1]
        igf = igf && is_gauge_fixed(H[n],ir,ic,ul,eps;kwargs...)
        if !igf
            break
        end
        ir=dag(ic)
    end
    return igf
end

#
# We store the tₙ₋₁ as Matrix (instead of ITensor) because the indices change after extract_blocks,
#  because of the way the current subtensor functions are implemented.
#
function gauge_fix!(W::ITensor,ir::Index,ic::Index,tₙ₋₁::Matrix{Float64},ms::matrix_state)
    @assert is_regular_form(W,ms.ul)
    Wb=extract_blocks(W,ir,ic,ms;all=true,fix_inds=true)
    𝕀,𝑨,𝒃,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝒃,Wb.𝒄,Wb.𝒅 #for readability below.
    nr,nc=dim(ir),dim(ic)
    nb,nf = ms.lr==left ? (nr,nc) : (nc,nr)
    #
    #  Make in ITensor with suitable indices from the tprev vector.
    #
    if nb>1
        if ms.ul==lower
            ibd = ms.lr==left ? Wb.ird : Wb.icd #backwards facing index on d block
            ibb = ms.lr==left ? Wb.irb : Wb.icb #backwards facing index on b block
        else
            ibd = ms.lr==left ? Wb.icd : Wb.ird #backwards facing index on d block
            ibb = ms.lr==left ? Wb.icb : Wb.irb #backwards facing index on b block
        end
        𝒕ₙ₋₁=ITensor(tₙ₋₁,dag(ibb),ibd)
    end
    𝒄⎖=nothing
    #
    #  First two if blocks are special handling for row and column vector at the edges of the MPO
    #
    if nb==1 #1xnf at start of sweep.
        𝒕ₙ=c0(Wb) 
        𝒄⎖=𝒄-𝕀*𝒕ₙ
        𝒅⎖=𝒅
    elseif nf==1 #nbx1 at the end of the sweep
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝒃
        𝒕ₙ=ITensor(1.0,Index(1),Index(1)) #Not used, but required for the return statement.
    else
        𝒕ₙ=𝒕ₙ₋₁*A0(Wb)+c0(Wb)
        𝒄⎖=𝒄+𝒕ₙ₋₁*𝑨-𝒕ₙ*𝕀
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝒃
    end
    
    if ms.ul==lower
        W[ir=>nr:nr,ic=>1:1]=𝒅⎖
    else
        W[ir=>1:1,ic=>nc:nc]=𝒅⎖
    end
    @assert is_regular_form(W,ms.ul)

    if !isnothing(𝒄⎖)
        if ms.ul==lower
            if ms.lr==left
                W[ir=>nr:nr,ic=>2:nc-1]=𝒄⎖
            else
                W[ir=>2:nr-1,ic=>1:1]=𝒄⎖    
            end
        else
            if ms.lr==left
                W[ir=>1:1,ic=>2:nc-1]=𝒄⎖
            else
                W[ir=>2:nr-1,ic=>nc:nc]=𝒄⎖    
            end
        end
    end
    @assert is_regular_form(W,ms.ul)

    # 𝒕ₙ is always a vector (or 1xN matrix) but we would have to sort the indices in order for
    # vector(𝒕ₙ) to work.
    return matrix(𝒕ₙ)
end

function gauge_fix!(H::MPO,ils::Vector{Index{T}},ms::matrix_state) where {T}
    N=length(H)
    tₙ=Matrix{Float64}(undef,1,1)
    ir=dag(ils[1])
    for n in 1:N
        ic=ils[n+1]
        @assert hasinds(H[n],ir,ic)
        tₙ=gauge_fix!(H[n],ir,ic,tₙ,matrix_state(ms.ul,left))
        @assert is_regular_form(H[n],ms.ul)
        ir=dag(ic)
    end
    #tₙ=Matrix{Float64}(undef,1,1) end of sweep above already returns this.
    ic=ils[N+1]
    for n in N:-1:1
        ir=dag(ils[n])
        @assert hasinds(H[n],ir,ic)
        tₙ=gauge_fix!(H[n],ir,ic,tₙ,matrix_state(ms.ul,right))
        @assert is_regular_form(H[n],ms.ul)
        ic=dag(ir)
    end
end

#-------------------------------------------------------------------------------
#
#  block qx and orthogonalization of the vcat(𝑨,𝒄) and hcat(𝒃,𝑨) blocks.
#
function ac_qx(W::ITensor,ir::Index,ic::Index,ms::matrix_state;kwargs...)
    @checkflux(W)
    @assert hasinds(W,ic)
    @assert hasinds(W,ir)
    Wb=extract_blocks(W,ir,ic,ms;Ac=true)
    if ms.ul==lower
        ilf_Ac = ms.lr==left ? Wb.icAc : Wb.irAc
    else
        ilf_Ac = ms.lr==left ? Wb.irAc : Wb.icAc
    end
    ilb,ilf =  ms.lr==left ? (ir,ic) : (ic,ir) #Backward and forward indices.
    @checkflux(Wb.𝑨𝒄)
    if ms.lr==left
        Qinds=noncommoninds(Wb.𝑨𝒄,ilf_Ac) 
        Q,R,iq=qr(Wb.𝑨𝒄,Qinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf_Ac
        R,Q,iq=rq(Wb.𝑨𝒄,Rinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    end
    @checkflux(Q)
    @checkflux(R)
    # Re-scale
    dh=d(Wb) #dimension of local Hilbert space.
    Q*=sqrt(dh)
    R/=sqrt(dh)
    Wp,iqp=set_𝑨𝒄_block(W,Q,ilb,ilf,iq,ms) 
    @assert is_regular_form(Wp,ms.ul)
    R=prime(R,ilf_Ac) #both inds or R have the same tags, so we prime one of them so the grow function can distinguish.
    Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    return Wp,Rp,iqp
end

function ac_orthogonalize!(H::MPO,ils::Vector{Index{T}},ms::matrix_state,eps::Float64) where {T}
    if !is_gauge_fixed(H,ils,ms.ul,eps)
        gauge_fix!(H,ils,ms)
    end
    rng=sweep(H,ms.lr)
    if ms.lr==left
        ir=dag(ils[1])
        for n in rng
            nn=n+rng.step
            ic=ils[n+1]
            H[n],R,iqp=ac_qx(H[n],ir,ic,ms)
            H[nn]=R*H[nn]
            @assert is_regular_form(H[nn],ms.ul)
            ils[n+1]=iqp
            ir=dag(iqp)
        end
    else
        ic=ils[length(H)+1]
        for n in rng
            nn=n+rng.step
            ir=dag(ils[n])
            H[n],R,iqp=ac_qx(H[n],ir,ic,ms)
            H[nn]=R*H[nn]
            @assert is_regular_form(H[nn],ms.ul)
            ils[n]=ic=dag(iqp)
        end
    end
end


models=[
    [make_transIsing_MPO,"S=1/2",true],
    [make_transIsing_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1",true],
    [make_Hubbard_AutoMPO,"Electron",false],
    ]

@testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns" for model in models, qns in [false,true], ul=[lower,upper]
    eps=1e-14
    N=10 #5 sites
    NNN=7 #Include 2nd nearest neighbour interactions
    sites = siteinds(model[2],N,conserve_qns=qns);
    H=model[1](sites,NNN;ul=ul);
    pre_fixed=model[3] #Hamiltonian starts gauge fixed
    # @show get_Dw(H)
    @assert is_regular_form(H,ul)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)

    ils,d0,dN=add_edge_links!(H)
    @assert is_regular_form(H,ul)
    #
    #  Left->right sweep
    #
    ms=matrix_state(ul,left)
    @test pre_fixed == is_gauge_fixed(H,ils,ms.ul,eps) 
    ac_orthogonalize!(H,ils,mirror(ms),eps)
    ac_orthogonalize!(H,ils,ms,eps)
    @test check_ortho(H,ms)
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Now everything should be fixed
    qns && show_directions(H)
    #
    #  Expectation value check.
    #
    remove_edge_links!(H,d0,dN)
    E1=inner(psi',H,psi)
    @test E0 ≈ E1 atol = eps
    #
    #  Right->left sweep
    #
    ils,d0,dN=add_edge_links!(H)
    ms=matrix_state(ul,right)
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Should still be gauge fixed
    ac_orthogonalize!(H,ils,ms,eps)
    @test check_ortho(H,ms)
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Should still be gauge fixed
    qns && show_directions(H)
    #
    #  Expectation value check.
    #
    remove_edge_links!(H,d0,dN)
    E2=inner(psi',H,psi)
    @test E0 ≈ E2 atol = eps
end

@testset "Gauge transform rectangular W, qns=$qns, ul=$ul" for model in models, qns in [false,true], ul=[lower,upper]
    eps=1e-14
    
    N=5 #5 sites
    NNN=2 #Include 2nd nearest neighbour interactions
    sites = siteinds(model[2],N,conserve_qns=false)
    H=model[1](sites,NNN;ul=ul)
    pre_fixed=model[3] #Hamiltonian starts gauge fixed
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)
    
    ils,d0,dN=add_edge_links!(H)

    ms=matrix_state(ul,left)
    @test pre_fixed==is_gauge_fixed(H,ils,ms.ul,eps)

    H_lwl=deepcopy(H)
    @test pre_fixed==is_gauge_fixed(H_lwl,ils,ms.ul,eps)
    apply_Ls!(H_lwl,ils,ms)
    @test is_gauge_fixed(H_lwl,ils,ms.ul,eps,b=false)
    @test pre_fixed==is_gauge_fixed(H_lwl,ils,ms.ul,eps,c=false)
    H_g=deepcopy(H) 
    # #   
    # #  Left->Right sweep doing gauge c0==0 transforms
    # #
    ir=dag(ils[1])
    t=Matrix{Float64}(undef,1,1)
    for n in 1:N
        ic =ils[n+1]
        t=gauge_fix!(H[n],ir,ic,t,ms)
        @test norm(H_lwl[n]-H[n])<eps
        @test is_gauge_fixed(H[n],ir,ic,ms.ul,eps;b=false)    
        ir=dag(ic)
    end
    @test is_gauge_fixed(H,ils,ms.ul,eps,b=false)
    @test pre_fixed==is_gauge_fixed(H,ils,ms.ul,eps,c=false)
    #
    #  Check that the energy expectation is invariant.
    #   
    He=remove_edge_links(H,d0,dN)
    E1=inner(psi',He,psi)
    @test E0 ≈ E1 atol = eps
    #
    # Do a full gauge transform on Hg   
    #
    @test pre_fixed==is_gauge_fixed(H,ils,ms.ul,eps) #b0's not done yet
    @test pre_fixed==is_gauge_fixed(H_g,ils,ms.ul,eps,b=false) #only check the c0s
    @test pre_fixed==is_gauge_fixed(H_g,ils,ms.ul,eps,c=false) #only check the b0s
    gauge_fix!(H_g,ils,ms)
    @test pre_fixed==is_gauge_fixed(H,ils,ms.ul,eps) #deepcopy ensures we didn't just (inadvertently) gauge fix H as well
    @test is_gauge_fixed(H_g,ils,ms.ul,eps)
    #
    #  Sweep right to left abd gauge all the b0's==0 .
    #
    ms=matrix_state(ul,right)
    ic=ils[N+1]
    t=Matrix{Float64}(undef,1,1)
    for n in N:-1:1
        W=H[n]
        ir =dag(ils[n])
        t=gauge_fix!(W,ir,ic,t,ms)
        @test norm(H_g[n]-W)<eps
        @test is_gauge_fixed(H[n],ir,ic,ms.ul,eps;b=false)    
        @test is_gauge_fixed(H[n],ir,ic,ms.ul,eps;c=false)    
        ic=dag(ir)
    end
    @test is_gauge_fixed(H,ils,ms.ul,eps) #Now everything should be fixed
    
 
    remove_edge_links!(H,d0,dN)
    E2=inner(psi',H,psi)
    @test E0 ≈ E2 atol = eps
end

@testset "Extract blocks qns=$qns, ul=$ul" for qns in [false,true], ul=[lower,upper]
    eps=1e-15
    N=5 #5 sites
    NNN=2 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=qns)
    d=dim(inds(sites[1])[1])
    H=make_Hubbard_AutoMPO(sites,NNN;ul=ul)
    ils,d0,dN=add_dummy_links!(H)
    @test all(il->dir(il)==dir(ils[1]),ils) 
   

    ms= ul==lower ? matrix_state(ul,left) : matrix_state(ul,right)
    ir,ic=dag(ils[1]),ils[2]
    nr,nc=dim(ir),dim(ic)
    W=H[1]
    #pprint(W)
    rfb=extract_blocks(W,ir,ic,ms;all=true)
    @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
    @test isnothing(rfb.𝑨) 
    if ul==lower   
        @test isnothing(rfb.𝒃)
        norm(array(rfb.𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
        norm(array(rfb.𝒄)-array(W[ir=>nr:nr,ic=>2:nc-1]))<eps
    else
        @test isnothing(rfb.𝒄)
        norm(array(rfb.𝒅)-array(W[ir=>1:1,ic=>nc:nc]))<eps
        norm(array(rfb.𝒃)-array(W[ir=>1:1,ic=>2:nc-1]))<eps
    end
       
    W=H[N]
    ir,ic=dag(ils[N]),ils[N+1]
    nr,nc=dim(ir),dim(ic)
    rfb=extract_blocks(W,ir,ic,ms;all=true,fix_inds=true)
    @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
    @test isnothing(rfb.𝑨)    
    if ul==lower 
        @test isnothing(rfb.𝒄) 
        @test norm(array(rfb.𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
        @test norm(array(rfb.𝒃)-array(W[ir=>2:nr-1,ic=>1:1]))<eps
    else
        @test isnothing(rfb.𝒃) 
        @test norm(array(rfb.𝒅)-array(W[ir=>1:1,ic=>nc:nc]))<eps
        @test norm(array(rfb.𝒄)-array(W[ir=>2:nr-1,ic=>nc:nc]))<eps
    end
   
    W=H[2]
    ir,ic=dag(ils[2]),ils[3]
    nr,nc=dim(ir),dim(ic)
    rfb=extract_blocks(W,ir,ic,ms;all=true,fix_inds=true,Ac=true)
    if ul==lower
        @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
        @test norm(array(rfb.𝒅)-array(W[ir=>nr:nr,ic=>1:1]))<eps
        @test norm(array(rfb.𝒃)-array(W[ir=>2:nr-1,ic=>1:1]))<eps
        @test norm(array(rfb.𝒄)-array(W[ir=>nr:nr,ic=>2:nc-1]))<eps
        @test norm(array(rfb.𝑨)-array(W[ir=>2:nr-1,ic=>2:nc-1]))<eps
        @test norm(array(rfb.𝑨𝒄)-array(W[ir=>2:nr,ic=>2:nc-1]))<eps
    else
        @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
        @test norm(array(rfb.𝒅)-array(W[ir=>1:1,ic=>nc:nc]))<eps
        @test norm(array(rfb.𝒃)-array(W[ir=>1:1,ic=>2:nc-1]))<eps
        @test norm(array(rfb.𝒄)-array(W[ir=>2:nr-1,ic=>nc:nc]))<eps
        @test norm(array(rfb.𝑨)-array(W[ir=>2:nr-1,ic=>2:nc-1]))<eps
        @test norm(array(rfb.𝑨𝒄)-array(W[ir=>2:nr-1,ic=>2:nc]))<eps
    end

end

@testset "Calculate t's, L's and Linv's, qns=$qns" for qns in [false]
    N=10 #5 sites
    NNN=5 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=qns)
    H=make_Hubbard_AutoMPO(sites,NNN)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)

    ils,d0,dN=add_dummy_links!(H)
    ms=matrix_state(lower,left)
    apply_Ls!(H,ils,ms) #only gets the c0's, not the b0's
    #@test is_gauge_fixed(H,ils,ms.ul,1e-15) 
   
    remove_edge_links!(H,d0,dN)
    E2=inner(psi',H,psi)
    @test E0 ≈ E2 atol = 1e-15
end


nothing