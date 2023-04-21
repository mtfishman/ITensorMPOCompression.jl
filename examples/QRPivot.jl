using ITensors
using ITensorMPOCompression
using Test,Printf,Revise

import ITensors: tensor, Indices, AbstractMPS
import ITensorMPOCompression: @checkflux, mpoc_checkflux, insert_xblock, default_eps, is_regular_form

Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)

#-------------------------------------------------------------------------------
#
#  These functions are the dev. testing only.
#
function calculate_ts(H::MPO,ils::Vector{Index{T}},irs::Vector{Index{T}},ms::matrix_state) where {T}
    ts=Vector{Float64}[]
    il=ils[1]
    tprev=zeros(1)
    push!(ts,tprev)
    for n in eachindex(H)
        ir=irs[n]
        Wb=extract_blocks(H[n],il,ir,ms;all=true,fix_inds=true)
        𝕀,𝑨,𝐛̂,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝐛̂,Wb.𝒄,Wb.𝒅
        dh=d(Wb)
        nr,nc=dim(il),dim(ir)
        if nr==1 
            c0=𝒄*dag(𝕀)/dh
            t=vector_o2(c0)
        elseif nc==1
            t=zeros(1)
        else
            @assert hasinds(Wb.𝑨,Wb.irA,Wb.icA)
            ict=commonind(𝐛̂,(Wb.irA,Wb.icA))
            irt=commonind(𝒅,(Wb.irc,Wb.icc))
            tprevT=ITensor(tprev,irt,dag(ict))

            𝒄₀=𝒄*dag(𝕀)/dh
            𝑨₀=𝑨*dag(𝕀)/dh
            t=vector_o2(tprevT*𝑨₀+𝒄₀)
        end
        #@show t
        push!(ts,t)
        tprev=t
        il=dag(ir)
    end
    return ts
end

function calculate_Ls(H::MPO,ils::Vector{Index{T}},irs::Vector{Index{T}},ms::matrix_state) where {T}
    Ls=Matrix{Float64}[]
    Linvs=Matrix{Float64}[]
    ts=calculate_ts(H,ils,irs,ms)
    N=length(H)
    links=[ils...,dag(irs[N])]
    @assert length(ts)==length(links)
    for n in eachindex(links)
        ic=links[n]
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

function apply_Ls!(H::MPO,ils::Vector{Index{T}},irs::Vector{Index{T}},ms::matrix_state) where {T}
    Ls,Linvs=calculate_Ls(H,ils,irs,ms)
    il=ils[1]
    for n in eachindex(H)
        ir=irs[n]
        @assert hasinds(H[n],il,ir)
        is=noncommoninds(H[n],il,ir)
        LT=ITensor(Ls[n],il',il)
        LinvT=ITensor(Linvs[n+1],ir,ir')
        H[n]=noprime(LT*H[n]*LinvT,(ir',il')) #we have to be careful not unprime the site index.
        il=dag(ir)
    end
end

#-------------------------------------------------------------------------------
#
#  Make all ITensors in H of order 4.  This simplifies the code.
#
function add_edge_links!(H::MPO)
    N=length(H)
    irs=map(n->linkind(H,n),1:N-1) #right facing index, which can be thought of as a column index
    ils=dag.(irs) #left facing index, or row index.
    ts=ITensors.trivial_space(irs[1])
    T=eltype(H[1])
    il0=Index(ts;tags="Link,l=0",dir=dir(dag(irs[1])))
    ilN=Index(ts;tags="Link,l=$N",dir=dir(irs[1]))
    d0=onehot(T, il0 => 1)
    dN=onehot(T, ilN => 1)
    H[1]*=d0
    H[N]*=dN
    return [il0,ils...],[irs...,ilN],d0,dN
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

mutable struct reg_form_Op
    W::ITensor
    ileft::Index
    iright::Index
    ul::reg_form
    function reg_form_Op(W::ITensor,ileft::Index,iright::Index,ul::reg_form) 
        @assert hasinds(W,ileft,iright)
        @assert is_regular_form(W,ul)
        return new(W,ileft,iright,ul)
    end
end

mutable struct reg_form_MPO <: AbstractMPS
    data::Vector{reg_form_Op}
    llim::Int
    rlim::Int
    d0::ITensor
    dN::ITensor
    ul::reg_form
    function reg_form_MPO(H::MPO,ils::Indices,irs::Indices,d0::ITensor,dN::ITensor,ul::reg_form)
        N=length(H)
        @assert length(ils)==N
        @assert length(irs)==N
        data=Vector{reg_form_Op}(undef,N)
        for n in eachindex(H)
            data[n]=reg_form_Op(H[n],ils[n],irs[n],ul)
        end
        return new(data,H.llim,H.rlim,d0,dN,ul)
    end
    function reg_form_MPO(Ws::Vector{reg_form_Op},llim::Int64,rlim::Int64,d0::ITensor,dN::ITensor,ul::reg_form)
        return new(Ws,llim,rlim,d0,dN,ul)
    end
end


function reg_form_MPO(H::MPO,eps::Float64=1e-14)
    (bl,bu)=detect_regular_form(H,eps)
    if !(bl || bu)
        throw(ErrorException("MPO++(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
        @pprint(H[1])
    end
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    ils,irs,d0,dN=add_edge_links!(H)
    return reg_form_MPO(H,ils,irs,d0,dN,ul)
end

function ITensors.MPO(Hrf::reg_form_MPO)::MPO
    N=length(Hrf)
    H=MPO(Ws(Hrf))
    H[1]*=dag(Hrf.d0)
    H[N]*=dag(Hrf.dN)
    return H
end

data(H::reg_form_MPO)=H.data

function Ws(H::reg_form_MPO)
    return map(n-> H[n].W,1:length(H))
end

Base.length(H::reg_form_MPO) = length(H.data)
Base.reverse(H::reg_form_MPO) = reg_form_MPO(reverse(H.data),H.llim,H.rlim,H.d0,H.dN,H.ul)
Base.iterate(H::reg_form_MPO, args...) = iterate(H.data, args...)
Base.getindex(H::reg_form_MPO, args...) = getindex(H.data, args...)
Base.setindex!(H::reg_form_MPO, args...) = setindex!(H.data, args...)

function is_regular_form(W::reg_form_Op,eps::Float64=default_eps)::Bool
    i = W.ul==lower ? 1 : 2
    return detect_regular_form(W.W,eps)[i]
end

function is_regular_form(H::reg_form_MPO,eps::Float64=default_eps)::Bool
    for W in H
        !is_regular_form(W) && return false
    end
    return true
end

function ITensorMPOCompression.check_ortho(H::reg_form_MPO,lr::orth_type,eps::Float64=default_eps)::Bool
    ms=matrix_state(H.ul,lr)
    for n in sweep(H,ms.lr) #skip the edge row/col opertors
        !check_ortho(H[n].W,ms,eps) && return false
    end
    return true
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
    𝐛̂::Union{ITensor,Nothing}
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
b0(rfb::regform_blocks)::ITensor=rfb.𝐛̂*dag(rfb.𝕀)/d(rfb)
c0(rfb::regform_blocks)::ITensor=rfb.𝒄*dag(rfb.𝕀)/d(rfb)
A0(rfb::regform_blocks)::ITensor=rfb.𝑨*dag(rfb.𝕀)/d(rfb)

#
#  Transpose inds for upper, no-op for lower
#
function swap_ul(ileft::Index,iright::Index,ul::reg_form)
    return ul==lower ? (ileft,iright,dim(ileft),dim(iright)) :  (iright,ileft,dim(iright),dim(ileft))
end
#  Use recognizably distinct UTF symbols for operators, and op valued vectors and matrices: 𝕀 𝑨 𝒃 𝒄 𝒅 ⌃ c₀ 𝑨𝒄
# symbols from here: https://www.compart.com/en/unicode/block/U+1D400
extract_blocks(W::reg_form_Op,lr::orth_type;kwargs...)=extract_blocks(W.W,W.ileft,W.iright,matrix_state(W.ul,lr);kwargs...)

function extract_blocks(W::ITensor,ir::Index,ic::Index,ms::matrix_state;all=false,c=true,b=false,d=false,A=false,Ac=false,I=true,fix_inds=false,swap_bc=true)::regform_blocks
    @assert hasinds(W,ir,ic)
    @assert tags(ir)!=tags(ic)
    @assert plev(ir)==0
    @assert plev(ic)==0
    if dir(W,ic)!=dir(ic)
        ic=dag(ic)
    end
    if dir(W,ir)!=dir(ir)
        ir=dag(ir)
    end
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
    if !llur(ms) && swap_bc #not lower-left or upper-right
        b,c=c,b #swap flags
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
        if llur(ms)
            rfb.𝑨𝒄= nr>1 ? W[ir=>2:nr,ic=>2:nc-1] : W[ir=>1:1,ic=>2:nc-1]
        else
            rfb.𝑨𝒄= nc>1 ? W[ir=>2:nr-1,ic=>1:nc-1]  : W[ir=>2:nr-1,ic=>1:1]
        end
        rfb.irAc,=inds(rfb.𝑨𝒄,tags=tags(ir))
        rfb.icAc,=inds(rfb.𝑨𝒄,tags=tags(ic))
    end
    if b
        rfb.𝐛̂= W[ir=>2:nr-1,ic=>1:1]
        rfb.irb,=inds(rfb.𝐛̂,tags=tags(ir))
        rfb.icb,=inds(rfb.𝐛̂,tags=tags(ic))
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
            rfb.irc=rfb.ird
        end
        if !isnothing(rfb.𝐛̂)
            rfb.𝐛̂=replaceind(rfb.𝐛̂,rfb.icb,rfb.icd)
            rfb.icb=rfb.icd
        end
        if !isnothing(rfb.𝑨)
            rfb.𝑨=replaceinds(rfb.𝑨,[rfb.irA,rfb.icA],[rfb.irb,rfb.icc])
            rfb.irA,rfb.icA=rfb.irb,rfb.icc
        end
    end
    if !llur(ms) && swap_bc #not lower-left or upper-right
        rfb.𝐛̂,rfb.𝒄=rfb.𝒄,rfb.𝐛̂
        rfb.irb,rfb.irc=rfb.irc,rfb.irb
        rfb.icb,rfb.icc=rfb.icc,rfb.icb
    end
    if !isnothing(rfb.𝑨)
        @assert hasinds(rfb.𝑨,rfb.irA,rfb.icA)
    end
    return rfb
end

# lower left or upper right
llur(ms::matrix_state)=ms.lr==left&&ms.ul==lower || ms.lr==right&&ms.ul==upper


function set_𝐛̂_block!(W::ITensor,𝐛̂::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    W[i1=>2:n1-1,i2=>1:1]=𝐛̂
end

function set_𝒄_block!(W::ITensor,𝒄::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    W[i1=>n1:n1,i2=>2:n2-1]=𝒄
end
function set_𝐛̂𝒄_block!(W::ITensor,𝐛̂𝒄::ITensor,ileft::Index,iright::Index,ms::matrix_state)
    if llur(ms)
        set_𝐛̂_block!(W,𝐛̂𝒄,ileft,iright,ms.ul)
    else
        set_𝒄_block!(W,𝐛̂𝒄,ileft,iright,ms.ul)
    end
end

# noop versions for when b/c are empty.  Happens in edge ops of H.
function set_𝐛̂𝒄_block!(::ITensor,::Nothing,::Index,::Index,::matrix_state)
end
function set_𝐛̂_block!(::ITensor,::Nothing,::Index,::Index,::reg_form)
end
function set_𝒄_block!(::ITensor,::Nothing,::Index,::Index,::reg_form)
end

function set_𝒅_block!(W::ITensor,𝒅::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    W[i1=>n1:n1,i2=>1:1]=𝒅
end

function set_𝕀_block!(W::ITensor,𝕀::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    n1>1 && assign!(W,𝕀,i1=>1,i2=>1)
    n2>1 && assign!(W,𝕀,i1=>n1,i2=>n2)
end

function set_𝑨𝒄_block(W::ITensor,𝑨𝒄::ITensor,ileft::Index,iright::Index,ms::matrix_state)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ms.ul)
    if llur(ms) #lower left/upper right
        min1=Base.min(n1,2)
        W[i1=>min1:n1,i2=>2:n2-1]=𝑨𝒄
    else #lower right/upper left
        max2=Base.max(n2-1,1)
        W[i1=>2:n1-1,i2=>1:max2]=𝑨𝒄
    end
end
#-------------------------------------------------------------------------------
#
#  Gauge fixing functions.  In this conext gauge fixing means setting b₀=<𝐛̂,𝕀> && c₀=<𝒄,𝕀> to zero
#
function is_gauge_fixed(W::ITensor,il::Index{T},ir::Index{T},ul::reg_form,eps::Float64;b=true,c=true)::Bool where {T}
    igf=true
    ms=matrix_state(ul,left)
    Wb=extract_blocks(W,il,ir,ms;c=true,b=true)
    if b && !isnothing(Wb.𝐛̂) 
        igf=igf && norm(b0(Wb))<eps
    end
    if c && !isnothing(Wb.𝒄)
        igf=igf && norm(c0(Wb))<eps
    end
    return igf
end

function is_gauge_fixed(W::reg_form_Op,eps::Float64;b=true,c=true)::Bool where {T}
    igf=true
    Wb=extract_blocks(W,left;c=true,b=true)
    if b && dim(W.ileft)>1
        igf=igf && norm(b0(Wb))<eps
    end
    if c && dim(W.iright)>1
        igf=igf && norm(c0(Wb))<eps
    end
    return igf
end

function is_gauge_fixed(H::MPO,ils::Vector{Index{T}},irs::Vector{Index{T}},ul::reg_form,eps::Float64;kwargs...)::Bool where {T}
    igf=true
    il=ils[1]  #left facing index
    for n in eachindex(H)
        ir=irs[n]  #right facing index
        igf = igf && is_gauge_fixed(H[n],il,ir,ul,eps;kwargs...)
        if !igf
            break
        end
        il=dag(ir)
    end
    return igf
end
function is_gauge_fixed(Hrf::reg_form_MPO,eps::Float64;kwargs...)::Bool where {T}
    for W in Hrf
        !is_gauge_fixed(W,eps;kwargs...) && return false
    end
    return true
end
#
#  Find the first dim==1 index and remove it, then return a Vector.
#
function vector_o2(T::ITensor)
    @assert order(T)==2
    i1=inds(T)[findfirst(d->d==1,dims(T))]
    return vector(T*dag(onehot(i1=>1)))
end
#
# We store the tₙ₋₁ as Matrix (instead of ITensor) because the indices change after extract_blocks,
#  because of the way the current subtensor functions are implemented.
#
function gauge_fix!(W::ITensor,ileft::Index,iright::Index,tₙ₋₁::Vector{Float64},ms::matrix_state)
    @assert is_regular_form(W,ms.ul)
    Wb=extract_blocks(W,ileft,iright,ms;all=true,fix_inds=true)
    𝕀,𝑨,𝐛̂,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝐛̂,Wb.𝒄,Wb.𝒅 #for readability below.
    nr,nc=dim(ileft),dim(iright)
    nb,nf = ms.lr==left ? (nr,nc) : (nc,nr)
    #
    #  Make in ITensor with suitable indices from the 𝒕ₙ₋₁ vector.
    #
    if nb>1
        ibd,ibb = llur(ms) ?  (Wb.ird, Wb.irb) : (Wb.icd, Wb.icb)
        𝒕ₙ₋₁=ITensor(tₙ₋₁,dag(ibb),ibd)
    end
    𝒄⎖=nothing
    #
    #  First two if blocks are special handling for row and column vector at the edges of the MPO
    #
    if nb==1 #col/row at start of sweep.
        𝒕ₙ=c0(Wb) 
        𝒄⎖=𝒄-𝕀*𝒕ₙ
        𝒅⎖=𝒅
    elseif nf==1 ##col/row at the end of the sweep
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝐛̂
        𝒕ₙ=ITensor(1.0,Index(1),Index(1)) #Not used, but required for the return statement.
    else
        𝒕ₙ=𝒕ₙ₋₁*A0(Wb)+c0(Wb)
        𝒄⎖=𝒄+𝒕ₙ₋₁*𝑨-𝒕ₙ*𝕀
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝐛̂
    end
    
    set_𝒅_block!(W,𝒅⎖,ileft,iright,ms.ul)
    @assert is_regular_form(W,ms.ul)

    if !isnothing(𝒄⎖)
        if llur(ms)
            set_𝒄_block!(W,𝒄⎖,ileft,iright,ms.ul)
        else
            set_𝐛̂_block!(W,𝒄⎖,ileft,iright,ms.ul)
        end
    end
    @assert is_regular_form(W,ms.ul)

    # 𝒕ₙ is always a 1xN tensor so we need to remove that dim==1 index in order for vector(𝒕ₙ) to work.
    return vector_o2(𝒕ₙ)
end

function gauge_fix!(W::reg_form_Op,tₙ₋₁::Vector{Float64},lr::orth_type)
    @assert is_regular_form(W)
    Wb=extract_blocks(W,lr;all=true,fix_inds=true)
    𝕀,𝑨,𝐛̂,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝐛̂,Wb.𝒄,Wb.𝒅 #for readability below.
    nr,nc=dim(W.ileft),dim(W.iright)
    nb,nf = lr==left ? (nr,nc) : (nc,nr)
    #
    #  Make in ITensor with suitable indices from the 𝒕ₙ₋₁ vector.
    #
    if nb>1
        ibd,ibb = llur(matrix_state(W.ul,lr)) ?  (Wb.ird, Wb.irb) : (Wb.icd, Wb.icb)
        𝒕ₙ₋₁=ITensor(tₙ₋₁,dag(ibb),ibd)
    end
    𝒄⎖=nothing
    #
    #  First two if blocks are special handling for row and column vector at the edges of the MPO
    #
    if nb==1 #col/row at start of sweep.
        𝒕ₙ=c0(Wb) 
        𝒄⎖=𝒄-𝕀*𝒕ₙ
        𝒅⎖=𝒅
    elseif nf==1 ##col/row at the end of the sweep
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝐛̂
        𝒕ₙ=ITensor(1.0,Index(1),Index(1)) #Not used, but required for the return statement.
    else
        𝒕ₙ=𝒕ₙ₋₁*A0(Wb)+c0(Wb)
        𝒄⎖=𝒄+𝒕ₙ₋₁*𝑨-𝒕ₙ*𝕀
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝐛̂
    end
    
    set_𝒅_block!(W.W,𝒅⎖,W.ileft,W.iright,W.ul)
    @assert is_regular_form(W)

    if !isnothing(𝒄⎖)
        if llur(matrix_state(W.ul,lr))
            set_𝒄_block!(W.W,𝒄⎖,W.ileft,W.iright,W.ul)
        else
            set_𝐛̂_block!(W.W,𝒄⎖,W.ileft,W.iright,W.ul)
        end
    end
    @assert is_regular_form(W)

    # 𝒕ₙ is always a 1xN tensor so we need to remove that dim==1 index in order for vector(𝒕ₙ) to work.
    return vector_o2(𝒕ₙ)
end

function gauge_fix!(H::MPO,ils::Vector{Index{T}},irs::Vector{Index{T}},ms::matrix_state) where {T}
    N=length(H)
    tₙ=Vector{Float64}(undef,1)
    il=ils[1]
    for n in 1:N
        ir=irs[n]
        @assert hasinds(H[n],il,ir)
        tₙ=gauge_fix!(H[n],il,ir,tₙ,matrix_state(ms.ul,left))
        @assert is_regular_form(H[n],ms.ul)
        il=dag(ir)
    end
    #tₙ=Vector{Float64}(undef,1) end of sweep above already returns this.
    ir=irs[N]
    for n in N:-1:1
        il=ils[n]
        @assert hasinds(H[n],il,ir)
        tₙ=gauge_fix!(H[n],il,ir,tₙ,matrix_state(ms.ul,right))
        @assert is_regular_form(H[n],ms.ul)
        ir=dag(il)
    end
end

function gauge_fix!(H::reg_form_MPO) where {T}
    tₙ=Vector{Float64}(undef,1)
    for W in H
        tₙ=gauge_fix!(W,tₙ,left)
        @assert is_regular_form(W)
    end
    #tₙ=Vector{Float64}(undef,1) end of sweep above already returns this.
    for W in reverse(H)
        tₙ=gauge_fix!(W,tₙ,right)
        @assert is_regular_form(W)
    end
end

function equal_edge_blocks(i1::ITensors.QNIndex,i2::ITensors.QNIndex)::Bool
    qns1,qns2=space(i1),space(i2)
    qn11,qn1n=qns1[1],qns1[nblocks(qns1)]
    qn21,qn2n=qns2[1],qns2[nblocks(qns2)]
    return ITensors.have_same_qns(qn(qn11),qn(qn21)) && ITensors.have_same_qns(qn(qn1n),qn(qn2n))
end

function equal_edge_blocks(::Index,::Index)::Bool
     return true
end
#-------------------------------------------------------------------------------
#
#  block qx and orthogonalization of the vcat(𝑨,𝒄) and hcat(𝐛̂,𝑨) blocks. 𝐐
#
function redim1(iq::ITensors.QNIndex,pad1::Int64,pad2::Int64,qns::ITensors.QNBlocks)
    @assert pad1==blockdim(qns[1]) #Splitting blocks not supported
    @assert pad2==blockdim(qns[end]) #Splitting blocks not supported
    qnsp=[qns[1],space(iq)...,qns[end]] #creat the new space
    return Index(qnsp,tags=tags(iq),plev=plev(iq),dir=dir(iq)) #create new index.
end

function redim1(iq::Index,pad1::Int64,pad2::Int64,Dw::Int64)
    @assert dim(iq)+pad1+pad2<=Dw #Splitting blocks not supported
    return Index(dim(iq)+pad1+pad2,tags=tags(iq),plev=plev(iq),dir=dir(iq)) #create new index.
end


function insert_Q(Wb::regform_blocks,𝐐::ITensor,ileft::Index,ic::Index,iq::Index,ms::matrix_state)
    ilb,ilf =  ms.lr==left ? (ileft,ic) : (ic,ileft) #Backward and forward indices.
    @assert !isnothing(Wb.𝑨𝒄)
    is=noncommoninds(Wb.𝑨𝒄,Wb.irAc,Wb.icAc)
    @assert hasinds(𝐐,iq,is...)
#    @assert dir(ileft)==dir(dag(ic))

    #
    #  Build new index and MPO Tensor
    #
    iqp=redim1(iq,1,1,space(ilf))  #pad with 1 at the start and 1 and the end: iqp =(1,iq,1).
    Wp=ITensor(0.0,ilb,iqp,is)
    ileft,iright =  ms.lr==left ? (ilb,iqp) :  (iqp,ilb)
    set_𝐛̂𝒄_block!(Wp,Wb.𝐛̂,ileft,iright,ms) #preserve b or c block from old W
    set_𝒅_block!(Wp,Wb.𝒅,ileft,iright,ms.ul) #preserve d block from old W
    set_𝕀_block!(Wp,Wb.𝕀,ileft,iright,ms.ul) #init I blocks from old W
    set_𝑨𝒄_block(Wp,𝐐,ileft,iright,ms) #Insert new Qs form QR decomp
    return Wp,iqp
end

function ac_qx(W::ITensor,ileft::Index,iright::Index,ms::matrix_state;kwargs...)
    @checkflux(W)
    @assert hasinds(W,iright)
    @assert hasinds(W,ileft)
    Wb=extract_blocks(W,ileft,iright,ms;Ac=true,all=true)
    ilf_Ac = llur(ms) ?  Wb.icAc : Wb.irAc
    ilf =  ms.lr==left ? iright : ileft #Backward and forward indices.
    @checkflux(Wb.𝑨𝒄)
    if ms.lr==left
        Qinds=noncommoninds(Wb.𝑨𝒄,ilf_Ac) 
        Q,R,iq=qr(Wb.𝑨𝒄,Qinds;positive=true,cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf_Ac
        R,Q,iq=lq(Wb.𝑨𝒄,Rinds;positive=true,cutoff=1e-14,tags=tags(ilf))
    end
    @checkflux(Q)
    @checkflux(R)
    # Re-scale
    dh=d(Wb) #dimension of local Hilbert space.
    @assert abs(dh-round(dh))==0.0
    Q*=sqrt(dh)
    R/=sqrt(dh)

    Wp,iqp=insert_Q(Wb,Q,ileft,iright,iq,ms) 
    @assert equal_edge_blocks(ilf,iqp)
    @assert is_regular_form(Wp,ms.ul)
    R=prime(R,ilf_Ac) #both inds or R have the same tags, so we prime one of them so the grow function can distinguish.
    Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    return Wp,Rp,iqp
end

function ac_qx(W::reg_form_Op,lr::orth_type;kwargs...)
    @checkflux(W.W)
    #@assert dir(W.ileft)==dir(dag(W.iright))
    Wb=extract_blocks(W,lr;Ac=true,all=true)
    ilf_Ac = llur(matrix_state(W.ul,lr)) ?  Wb.icAc : Wb.irAc
    ilb,ilf =  lr==left ? (W.ileft,W.iright) : (W.iright,W.ileft) #Backward and forward indices.
    @checkflux(Wb.𝑨𝒄)
    if lr==left
        Qinds=noncommoninds(Wb.𝑨𝒄,ilf_Ac) 
        Q,R,iq=qr(Wb.𝑨𝒄,Qinds;positive=true,cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf_Ac
        R,Q,iq=lq(Wb.𝑨𝒄,Rinds;positive=true,cutoff=1e-14,tags=tags(ilf))
    end
    @checkflux(Q)
    @checkflux(R)
    # Re-scale
    dh=d(Wb) #dimension of local Hilbert space.
    @assert abs(dh-round(dh))==0.0
    Q*=sqrt(dh)
    R/=sqrt(dh)

    Wp,iqp=insert_Q(Wb,Q,W.ileft,W.iright,iq,matrix_state(W.ul,lr)) 
    Wprf=lr==left ? reg_form_Op(Wp,ilb,iqp,W.ul) : reg_form_Op(Wp,iqp,ilb,W.ul)
    @assert equal_edge_blocks(ilf,iqp)
    @assert is_regular_form(Wprf)
    R=prime(R,ilf_Ac) #both inds or R have the same tags, so we prime one of them so the grow function can distinguish.
    Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    return Wprf,Rp,iqp
end

function ac_orthogonalize!(H::MPO,ils::Vector{Index{T}},irs::Vector{Index{T}},ms::matrix_state,eps::Float64) where {T}
    if !is_gauge_fixed(H,ils,irs,ms.ul,eps)
        gauge_fix!(H,ils,irs,ms)
    end
    rng=sweep(H,ms.lr)
    if ms.lr==left
        il=ils[1]
        for n in rng
            nn=n+rng.step
            ir=irs[n]
            #@show n il ir
            H[n],R,iqp=ac_qx(H[n],il,ir,ms)
            H[nn]=R*H[nn]
            @assert is_regular_form(H[nn],ms.ul)
            irs[n]=dag(iqp)
            il=dag(iqp)
            ils[n+1]=iqp
        end
    else
        ir=irs[length(H)]
        for n in rng
            nn=n+rng.step
            il=ils[n]
            H[n],R,iqp=ac_qx(H[n],il,ir,ms)
            H[nn]=R*H[nn]
            @assert is_regular_form(H[nn],ms.ul)
            irs[n-1]=ir=dag(iqp)
            ils[n]=iqp
        end
    end
end

function ac_orthogonalize!(H::reg_form_MPO,lr::orth_type,eps::Float64) 
    if !is_gauge_fixed(H,eps)
        gauge_fix!(H)
    end
    rng=sweep(H,lr)
    if lr==left
        for n in rng
            nn=n+rng.step
            H[n],R,iqp=ac_qx(H[n],lr)
            @assert H[n].iright==iqp

            H[nn].W=R*H[nn].W
            H[nn].ileft=dag(iqp)
            @assert is_regular_form(H[nn])
        end
    else
        for n in rng
            nn=n+rng.step
            H[n],R,iqp=ac_qx(H[n],lr)
            @assert H[n].ileft==iqp
            @assert dir(H[n].ileft)==dir(iqp)

            H[nn].W=R*H[nn].W
            H[nn].iright=dag(iqp)
            @assert is_regular_form(H[nn])
        end
    end
end
verbose=false

@testset "Ac/Ab block respecting decomposition tests" begin
    models=[
        [make_transIsing_MPO,"S=1/2",true],
        [make_transIsing_AutoMPO,"S=1/2",true],
        [make_Heisenberg_AutoMPO,"S=1/2",true],
        [make_Heisenberg_AutoMPO,"S=1",true],
        [make_Hubbard_AutoMPO,"Electron",false],
        ]

    @testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false,true], ul=[lower,upper]
        eps=1e-14
        N=10 #5 sites
        NNN=5 #Include 2nd nearest neighbour interactions
        sites = siteinds(model[2],N,conserve_qns=qns);
        H=model[1](sites,NNN;ul=ul);
        Hrf=reg_form_MPO(model[1](sites,NNN;ul=ul))
        pre_fixed=model[3] #Hamiltonian starts gauge fixed
        # @show get_Dw(H)
        @assert is_regular_form(H,ul)
        state=[isodd(n) ? "Up" : "Dn" for n=1:N]
        psi=randomMPS(sites,state)
        E0=inner(psi',H,psi)

        ils,irs,d0,dN=add_edge_links!(H)
        @test all(il->dir(il)==dir(ils[1]),ils) 
        @test all(ir->dir(ir)==dir(irs[1]),irs) 
        @test is_regular_form(H,ul)
        @test is_regular_form(Hrf)
        #
        #  Left->right sweep
        #
        lr=left
        ms=matrix_state(ul,lr)
        @test pre_fixed == is_gauge_fixed(H,ils,irs,ms.ul,eps) 
        @test pre_fixed == is_gauge_fixed(Hrf,eps) 
        # verbose && qns && show_directions(H)
        # verbose && qns && show_directions(Hrf.H)
        ac_orthogonalize!(H,ils,irs,mirror(ms),eps)
        ac_orthogonalize!(Hrf,mirror(lr),eps)
        @test is_regular_form(Hrf)
        @test check_ortho(Hrf,mirror(lr))
        @test is_gauge_fixed(Hrf,eps) #Now everything should be fixed
        
        
        # verbose && qns && show_directions(H)
        # verbose && qns && show_directions(Hrf.H)
        ac_orthogonalize!(H,ils,irs,ms,eps)
        ac_orthogonalize!(Hrf,lr,eps)
        
        # verbose && qns && show_directions(H)
        # verbose && qns && show_directions(Hrf.H)
        @test is_regular_form(Hrf)
        @test check_ortho(Hrf,lr)
        @test is_gauge_fixed(Hrf,eps) #Now everything should be fixed
        #
        #  Expectation value check.
        #
        He1=MPO(Hrf)
        He=remove_edge_links(H,d0,dN)
        E1=inner(psi',He,psi)
        @test E0 ≈ E1 atol = eps
        E1=inner(psi',He1,psi)
        @test E0 ≈ E1 atol = eps
        #
        #  Right->left sweep
        #
        ms=matrix_state(ul,right)
        @test is_regular_form(Hrf)
        @test is_gauge_fixed(Hrf,eps) #Should still be gauge fixed
        ac_orthogonalize!(Hrf,right,eps)
        @test is_regular_form(Hrf)
        @test check_ortho(Hrf,right)
        @test is_gauge_fixed(Hrf,eps) #Should still be gauge fixed
        verbose && qns && show_directions(H)
        #
        # #  Expectation value check.
        # #
        He=MPO(Hrf)
        E2=inner(psi',He,psi)
        @test E0 ≈ E2 atol = eps
    end

    @testset "Gauge transform $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false], ul=[lower,upper]
        eps=1e-14
        
        N=5 #5 sites
        NNN=2 #Include 2nd nearest neighbour interactions
        sites = siteinds(model[2],N,conserve_qns=qns)
        H=model[1](sites,NNN;ul=ul)
        Hrf=reg_form_MPO(model[1](sites,NNN;ul=ul))
        pre_fixed=model[3] #Hamiltonian starts gauge fixed
        state=[isodd(n) ? "Up" : "Dn" for n=1:N]
        psi=randomMPS(sites,state)
        E0=inner(psi',H,psi)
        
        
        ils,irs,d0,dN=add_edge_links!(H)

        lr=left
        ms=matrix_state(ul,lr)
        @test pre_fixed==is_gauge_fixed(H,ils,irs,ms.ul,eps)
        @test pre_fixed==is_gauge_fixed(Hrf,eps)

        H_lwl=deepcopy(H)
        @test pre_fixed==is_gauge_fixed(H_lwl,ils,irs,ms.ul,eps)
        apply_Ls!(H_lwl,ils,irs,ms)
        @test is_gauge_fixed(H_lwl,ils,irs,ms.ul,eps,b=false)
        @test pre_fixed==is_gauge_fixed(H_lwl,ils,irs,ms.ul,eps,c=false)
        H_g=deepcopy(H) 
        #   
        #  Left->Right sweep doing gauge c0==0 transforms
        #
        il=ils[1]
        t=Vector{Float64}(undef,1)
        for n in 1:N
            ir =irs[n]
            t=gauge_fix!(H[n],il,ir,t,ms)
            @test norm(H_lwl[n]-H[n])<eps
            @test is_gauge_fixed(H[n],il,ir,ms.ul,eps;b=false)    
            il=dag(ir)
        end
        @test is_gauge_fixed(H,ils,irs,ms.ul,eps,b=false)
        @test pre_fixed==is_gauge_fixed(H,ils,irs,ms.ul,eps,c=false)
        #
        #  Check that the energy expectation is invariant.
        #   
        He=remove_edge_links(H,d0,dN)
        E1=inner(psi',He,psi)
        @test E0 ≈ E1 atol = eps
        # #
        # # Do a full gauge transform on Hg   
        # #
        @test pre_fixed==is_gauge_fixed(H,ils,irs,ms.ul,eps) #b0's not done yet
        @test pre_fixed==is_gauge_fixed(H_g,ils,irs,ms.ul,eps,b=false) #only check the c0s
        @test pre_fixed==is_gauge_fixed(H_g,ils,irs,ms.ul,eps,c=false) #only check the b0s
        gauge_fix!(H_g,ils,irs,ms)
        @test pre_fixed==is_gauge_fixed(H,ils,irs,ms.ul,eps) #deepcopy ensures we didn't just (inadvertently) gauge fix H as well
        @test is_gauge_fixed(H_g,ils,irs,ms.ul,eps)
        #
        #  Sweep right to left abd gauge all the b0's==0 .
        #
        ms=matrix_state(ul,right)
        ir=irs[N]
        t=Vector{Float64}(undef,1)
        for n in N:-1:1
            W=H[n]
            il =ils[n]
            t=gauge_fix!(W,il,ir,t,ms)
            @test norm(H_g[n]-W)<eps
            @test is_gauge_fixed(H[n],il,ir,ms.ul,eps;b=false)    
            @test is_gauge_fixed(H[n],il,ir,ms.ul,eps;c=false)    
            ir=dag(il)
        end
        @test is_gauge_fixed(H,ils,irs,ms.ul,eps) #Now everything should be fixed
    
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
        ils,irs,d0,dN=add_edge_links!(H)
        @test all(il->dir(il)==dir(ils[1]),ils) 
        @test all(ir->dir(ir)==dir(irs[1]),irs) 
    

        ms= ul==lower ? matrix_state(ul,left) : matrix_state(ul,right)
        il,ir =ils[1],irs[1]
        nr,nc=dim(il),dim(ir)
        W=H[1]
        #pprint(W)
        rfb=extract_blocks(W,il,ir,ms;all=true)
        @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
        @test isnothing(rfb.𝑨) 
        if ul==lower   
            @test isnothing(rfb.𝐛̂)
            norm(array(rfb.𝒅)-array(W[il=>nr:nr,ir=>1:1]))<eps
            norm(array(rfb.𝒄)-array(W[il=>nr:nr,ir=>2:nc-1]))<eps
        else
            @test isnothing(rfb.𝒄)
            norm(array(rfb.𝒅)-array(W[il=>1:1,ir=>nc:nc]))<eps
            norm(array(rfb.𝐛̂)-array(W[il=>1:1,ir=>2:nc-1]))<eps
        end
        
        W=H[N]
        il,ir =ils[N],irs[N]
        nr,nc=dim(il),dim(ir)
        rfb=extract_blocks(W,il,ir,ms;all=true,fix_inds=true)
        @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
        @test isnothing(rfb.𝑨)    
        if ul==lower 
            @test isnothing(rfb.𝒄) 
            @test norm(array(rfb.𝒅)-array(W[il=>nr:nr,ir=>1:1]))<eps
            @test norm(array(rfb.𝐛̂)-array(W[il=>2:nr-1,ir=>1:1]))<eps
        else
            @test isnothing(rfb.𝐛̂) 
            @test norm(array(rfb.𝒅)-array(W[il=>1:1,ir=>nc:nc]))<eps
            @test norm(array(rfb.𝒄)-array(W[il=>2:nr-1,ir=>nc:nc]))<eps
        end
    
        W=H[2]
        il,ir =ils[2],irs[2]
        nr,nc=dim(il),dim(ir)
        rfb=extract_blocks(W,il,ir,ms;all=true,fix_inds=true,Ac=true)
        if ul==lower
            @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
            @test norm(array(rfb.𝒅)-array(W[il=>nr:nr,ir=>1:1]))<eps
            @test norm(array(rfb.𝐛̂)-array(W[il=>2:nr-1,ir=>1:1]))<eps
            @test norm(array(rfb.𝒄)-array(W[il=>nr:nr,ir=>2:nc-1]))<eps
            @test norm(array(rfb.𝑨)-array(W[il=>2:nr-1,ir=>2:nc-1]))<eps
            @test norm(array(rfb.𝑨𝒄)-array(W[il=>2:nr,ir=>2:nc-1]))<eps
        else
            @test norm(matrix(rfb.𝕀)-1.0*Matrix(LinearAlgebra.I,d,d))<eps
            @test norm(array(rfb.𝒅)-array(W[il=>1:1,ir=>nc:nc]))<eps
            @test norm(array(rfb.𝐛̂)-array(W[il=>1:1,ir=>2:nc-1]))<eps
            @test norm(array(rfb.𝒄)-array(W[il=>2:nr-1,ir=>nc:nc]))<eps
            @test norm(array(rfb.𝑨)-array(W[il=>2:nr-1,ir=>2:nc-1]))<eps
            @test norm(array(rfb.𝑨𝒄)-array(W[il=>2:nr-1,ir=>2:nc]))<eps
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

        ils,irs,d0,dN=add_edge_links!(H)
        ms=matrix_state(lower,left)
        apply_Ls!(H,ils,irs,ms) #only gets the c0's, not the b0's
        @test is_gauge_fixed(H,ils,irs,ms.ul,1e-15;b=false) 
    
        remove_edge_links!(H,d0,dN)
        E2=inner(psi',H,psi)
        @test E0 ≈ E2 atol = 1e-15
    end

end

nothing