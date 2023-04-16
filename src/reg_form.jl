


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

function Base.getindex(Wrf::reg_form_Op,rleft::UnitRange,rright::UnitRange)
    return Wrf.W[Wrf.ileft=>rleft,Wrf.iright=>rright]
end

function is_regular_form(W::reg_form_Op,eps::Float64=default_eps)::Bool
    i = W.ul==lower ? 1 : 2
    return detect_regular_form(W.W,eps)[i]
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
    ms=matrix_state(H.ul,lr)
    for n in sweep(H,ms.lr) #skip the edge row/col opertors
        !check_ortho(H[n].W,ms,eps) && return false
    end
    return true
end