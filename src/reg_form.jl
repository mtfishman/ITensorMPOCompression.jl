


mutable struct reg_form_Op
    W::ITensor
    ileft::Index
    iright::Index
    ul::reg_form
    function reg_form_Op(W::ITensor,ileft::Index,iright::Index,ul::reg_form) 
        if !hasinds(W,ileft,iright)
            @show inds(W,tags="Link") ileft iright
        end
        @assert hasinds(W,ileft,iright)
        return new(W,ileft,iright,ul)
    end
    function reg_form_Op(W::ITensor,ul::reg_form) 
        return new(W,Index(1),Index(1),ul)
    end
end


function Base.getindex(Wrf::reg_form_Op,rleft::UnitRange,rright::UnitRange)
    return Wrf.W[Wrf.ileft=>rleft,Wrf.iright=>rright]
end

function detect_regular_form(Wrf::reg_form_Op,eps::Float64=default_eps)::Tuple{Bool,Bool}
    return is_regular_form(Wrf,lower,eps),is_regular_form(Wrf,upper,eps)
end

flip(ul::reg_form)= ul==lower ? upper : lower

is_regular_form(Wrf::reg_form_Op,eps::Float64=default_eps)=is_regular_form(Wrf,Wrf.ul,eps)

function is_regular_form(Wrf::reg_form_Op,ul::reg_form,eps::Float64=default_eps)::Bool
    ul_cache=Wrf.ul
    Wrf.ul=flip(ul)
    Wb=extract_blocks(Wrf,left;b=true,c=true,d=true)
    is=siteinds(Wrf)
    𝕀=delta(is) #We need to make our own, can't trust Wb.𝕀 is ul is wrong.
    dh=dim(is[1])
    nr,nc=dims(Wrf)
    if (nc==1 && ul==lower) || (nr==1 && ul==upper)
        i1=abs(scalar(dag(𝕀) * slice(Wrf.W,Wrf.ileft=>1,Wrf.iright=>1))-dh)<eps
        iN=bz=cz=dz=true
    end
    if (nr==1 && ul==lower) || (nc==1 && ul==upper)
        iN=abs(scalar(dag(𝕀) * slice(Wrf.W,Wrf.ileft=>nr,Wrf.iright=>nc))-dh)<eps
        i1=bz=cz=dz=true
    end
    if nr>1 && nc>1   
        i1=abs(scalar(dag(Wb.𝕀) * slice(Wrf.W,Wrf.ileft=>1,Wrf.iright=>1))-dh)<eps
        iN=abs(scalar(dag(Wb.𝕀) * slice(Wrf.W,Wrf.ileft=>nr,Wrf.iright=>nc))-dh)<eps
        bz=isnothing(Wb.𝒃) ? true : norm(Wb.𝒃)<eps
        cz=isnothing(Wb.𝒄) ? true : norm(Wb.𝒄)<eps
        dz=norm(Wb.𝒅)<eps
    end
    
    Wrf.ul=ul_cache
    # if !(i1 && iN && bz && cz && dz)
    #     pprint(Wrf.W)
    #     @show ul nr nc i1 iN bz cz dz
    #     @show dh scalar(dag(𝕀) * slice(Wrf.W,Wrf.ileft=>1,Wrf.iright=>1)) Wb.𝕀 𝕀
    # end


    return i1 && iN && bz && cz && dz
end

function Base.show(io::IO, Wrf::reg_form_Op)
    show(io,Wrf.ileft)
    show(io,Wrf.iright)
    show(io,Wrf.W)
end

ITensors.order(Wrf::reg_form_Op)=order(Wrf.W)

function check_ortho(Wrf::reg_form_Op,lr::orth_type,eps::Float64=default_eps)::Bool
    Wb=extract_blocks(Wrf,lr;V=true)
    DwDw=dim(Wb.irV)*dim(Wb.icV)
    ilf = llur(Wrf,lr) ? Wb.icV : Wb.irV
    
    Id=Wb.𝑽*prime(dag(Wb.𝑽),ilf)/d(Wb)
    if order(Id)==2
        is_can = norm(dense(Id)-delta(ilf,dag(ilf')))/sqrt(DwDw)<eps
        # if !is_can
        #     @show Id
        # end
    elseif order(Id)==0
        is_can = abs(scalar(Id)-d(Wb))<eps
    end
    return is_can
end

ITensors.dims(Wrf::reg_form_Op)=dim(Wrf.ileft),dim(Wrf.iright)
Base.getindex(Wrf::reg_form_Op,lr::orth_type)=lr==left ? Wrf.ileft : Wrf.iright 
function Base.setindex!(Wrf::reg_form_Op,il::Index,lr::orth_type)
    if lr==left 
         Wrf.ileft=il
    else
         Wrf.iright=il
    end
end
forward(Wrf::reg_form_Op,lr::orth_type)= Wrf[mirror(lr)]    
backward(Wrf::reg_form_Op,lr::orth_type)= Wrf[lr]   

ITensors.siteinds(Wrf::reg_form_Op)=noncommoninds(Wrf.W,Wrf.ileft,Wrf.iright)
ITensors.linkinds(Wrf::reg_form_Op)=Wrf.ileft,Wrf.iright
ITensors.linkinds(Wrf::reg_form_Op,lr::orth_type)=backward(Wrf,lr),forward(Wrf,lr)

function check(Wrf::reg_form_Op)
    @mpoc_assert order(Wrf.W)==4
    @mpoc_assert tags(Wrf.ileft)!=tags(Wrf.iright) || plev(Wrf.ileft)!=plev(Wrf.iright)
    @mpoc_assert hasinds(Wrf.W,Wrf.ileft)
    @mpoc_assert hasinds(Wrf.W,Wrf.iright)
    if hasqns(Wrf.W) 
        @mpoc_assert dir(Wrf.W,Wrf.ileft)==dir(Wrf.ileft)
        @mpoc_assert dir(Wrf.W,Wrf.iright)==dir(Wrf.iright)
    end
end


#-----------------------------------------------------------------------
#
#  Finite lattice with open BCs
#
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

function add_edge_links!(H::MPO)
    N=length(H)
    irs=map(n->ITensors.linkind(H,n),1:N-1) #right facing index, which can be thought of as a column index
    ils=dag.(irs) #left facing index, or row index.
    
    ts=ITensors.trivial_space(irs[1])
    T=eltype(H[1])
    il0=Index(ts;tags="Link,l=0",dir=dir(dag(irs[1])))
    ilN=Index(ts;tags="Link,l=$N",dir=dir(irs[1]))
    d0=onehot(T, il0 => 1)
    dN=onehot(T, ilN => 1)
    H[1]*=d0
    H[N]*=dN
    ils,irs=[il0,ils...],[irs...,ilN]
    for n in 1:N
        il,ir,W=ils[n],irs[n],H[n]
        #@show il ir inds(W,tags="Link")
        @assert !hasqns(il) || dir(W,il)==dir(il)
        @assert !hasqns(ir) || dir(W,ir)==dir(ir)
    end
    return ils,irs,d0,dN
end

function reg_form_MPO(H::MPO,eps::Float64=1e-14)
    (bl,bu)=detect_regular_form(H,eps)
    if !(bl || bu)
        throw(ErrorException("MPO++(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
        #@pprint(H[1])
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
Base.reverse(H::reg_form_MPO) = reg_form_MPO(Base.reverse(H.data),H.llim,H.rlim,H.d0,H.dN,H.ul)
Base.iterate(H::reg_form_MPO, args...) = iterate(H.data, args...)
Base.getindex(H::reg_form_MPO, args...) = getindex(H.data, args...)
Base.setindex!(H::reg_form_MPO, args...) = setindex!(H.data, args...)



function is_regular_form(H::reg_form_MPO,eps::Float64=default_eps)::Bool
    for W in H
        !is_regular_form(W,eps) && return false
    end
    return true
end


function check_ortho(H::reg_form_MPO,lr::orth_type,eps::Float64=default_eps)::Bool
    for n in sweep(H,lr) #skip the edge row/col opertors
        !check_ortho(H[n],lr,eps) && return false
    end
    return true
end


ITensors.inds(Wrf::reg_form_Op)=inds(Wrf.W)
function ITensors.setinds(Wrf::reg_form_Op,is)::reg_form_Op
    ITensors.setinds(Wrf.W,is)
    Wrf.ileft,Wrf.iright=parse_links(Wrf.W,left)
    return Wrf
end
#-----------------------------------------------------------------------
#
#  Infinite lattice with unit cell
#
function ITensorInfiniteMPS.translatecell(translator::Function, Wrf::reg_form_Op, n::Integer)
    new_inds=ITensorInfiniteMPS.translatecell(translator, inds(Wrf), n)
    W=ITensors.setinds(Wrf.W,new_inds)
    ileft,iright=parse_links(W)
    return reg_form_Op(W,ileft,iright,Wrf.ul)
end
  

mutable struct reg_form_iMPO <: AbstractInfiniteMPS
    data::CelledVector{reg_form_Op}
    llim::Int
    rlim::Int
    reverse::Bool
    ul::reg_form
    function reg_form_iMPO(H::InfiniteMPO,ul::reg_form)
        N=length(H)
        data=CelledVector{reg_form_Op}(undef,N)
        for n in eachindex(H)
            il,ir=parse_links(H[n])
            data[n]=reg_form_Op(H[n],il,ir,ul)
        end
        return new(data,H.llim,H.rlim,false,ul)
    end
    function reg_form_iMPO(Ws::CelledVector{reg_form_Op},llim::Int64,rlim::Int64,reverse::Bool,ul::reg_form)
        return new(Ws,llim,rlim,reverse,ul)
    end
end

data(H::reg_form_iMPO)=H.data

function Ws(H::reg_form_iMPO)
    return map(n-> H[n].W,1:length(H))
end

Base.length(H::reg_form_iMPO) = length(H.data)
Base.reverse(H::reg_form_iMPO) = reg_form_iMPO(Base.reverse(H.data),H.llim,H.rlim,H.reverse,H.ul)
Base.iterate(H::reg_form_iMPO, args...) = iterate(H.data, args...)
Base.getindex(H::reg_form_iMPO, n::Integer) = getindex(H.data,n)
Base.setindex!(H::reg_form_iMPO, W::reg_form_Op,n::Integer) = setindex!(H.data,W, n)
Base.copy(H::reg_form_iMPO)=reg_form_iMPO(copy(H.data),H.llim,H.rlim,H.reverse,H.ul)

function reg_form_iMPO(H::InfiniteMPO,eps::Float64=1e-14)
    (bl,bu)=detect_regular_form(H,eps)
    if !(bl || bu)
        throw(ErrorException("MPO++(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
        #@pprint(H[1])
    end
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    return reg_form_iMPO(H,ul)
end

function ITensorInfiniteMPS.InfiniteMPO(Hrf::reg_form_iMPO)::InfiniteMPO
    return InfiniteMPO(Ws(Hrf))
end

function to_openbc(Hrf::reg_form_iMPO)::reg_form_iMPO
    N=length(Hrf)
    if N>1
        l,r=get_lr(Hrf)    
        Hrf[1].W=l*prime(Hrf[1].W,Hrf[1].ileft)
        Hrf[N].W=prime(Hrf[N].W,Hrf[N].iright)*r
        @mpoc_assert length(inds(Hrf[1].W,tags="Link"))==1
        @mpoc_assert length(inds(Hrf[N].W,tags="Link"))==1
    end
    return Hrf
end

function get_lr(Hrf::reg_form_iMPO)::Tuple{ITensor,ITensor}
    N=length(Hrf)
    llink,rlink=linkinds(Hrf[1])
    l=ITensor(0.0,dag(llink'))
    r=ITensor(0.0,dag(rlink'))
    if Hrf.ul==lower
        l[llink'=>dim(llink)]=1.0
        r[rlink'=>1]=1.0
    else
        l[llink'=>1]=1.0
        r[rlink'=>dim(rlink)]=1.0
    end

    return l,r
end

function get_Dw(Hrf::reg_form_iMPO)
    return map(n->dim(Hrf[n].iright),eachindex(Hrf))
end

function check_ortho(H::reg_form_iMPO,lr::orth_type,eps::Float64=default_eps)::Bool
    for n in sweep(H,lr) #skip the edge row/col opertors
        !check_ortho(H[n],lr,eps) && return false
    end
    return true
end

function is_regular_form(H::reg_form_iMPO,eps::Float64=default_eps)::Bool
    for W in H
        !is_regular_form(W,eps) && return false
    end
    return true
end