module ITensorMPOCompression

using ITensors
using ITensors.NDTensors

import ITensors.BlockSparseTensor,ITensors.DenseTensor,ITensors.tensor,ITensors.orthogonalize!,ITensors.truncate!
                                                                    
export ql,lq,rq,assign!,getV,setV,growRL,to_openbc,set_scale!,block_qx,orthogonalize!,is_canonical,is_orthogonal
export reg_form,orth_type,matrix_state,upper,lower,none,left,right,mirror,parse_links
export is_regular_form,truncate,truncate!,getM,grow,detect_regular_form
export is_lower_regular_form,is_upper_regular_form,V_offsets
export detect_upper_lower,is_upper_lower,get_Dw,min,max,redim
export make_transIsing_MPO,make_Heisenberg_AutoMPO,make_transIsing_AutoMPO,fast_GS
export bond_spectrum,bond_spectrums,add_or_replace

default_eps=1e-14 #floats <default_eps are considered to be zero.

"""
    @enum reg_form  upper lower
    Indicates if an MPO matrix has either an `upper` or `lower` triangular form.
    This becomes non-trival for rectangular matrices.
    See also [`detect_upper_lower`](@ref)
"""
@enum reg_form  upper lower 

"""
    @enum orth_type none left right
    Indicates of an MPO matrix satisfies the conditions for `left` or `right` canonical form     
"""
@enum orth_type none left right

# """
#     mirror(lr::orth_type)::orth_type
#     returns this mirror of lr.  `left`->`right` and `right`->`left`
# """
function mirror(lr::orth_type)::orth_type
    if lr==left
        ret=right
    else #must be right
        ret=left
    end
    return ret
end

 """
     Indicates both the `orth_type` and `reg_form` of an MPO
     See also [`orth_type`](@ref) [`reg_form`](@ref)
 """
struct matrix_state
    ul::reg_form
    lr::orth_type
end

mirror(ms::matrix_state)::matrix_state=matrix_state(ms.ul,mirror(ms.lr))

"""
    V_offsets

A simple struct for encapsulating offsets for V-blocks

"""
struct V_offsets
    o1::Int64 #currently o1=o2 for std. Parker compression.  
    o2::Int64 #Leave them distinct for now until we know more
    #
    # The purpose of this struct is to ensure the asserts below
    #
    V_offsets(o1_::Int64, o2_::Int64) = begin
        @assert o1_==0 || o1_==1
        @assert o2_==0 || o2_==1 
        new(o1_,o2_)
    end 
end

"""
    V_offsets(ms::matrix_state)

Derives the correct V-block offsets for the given `matrix_state`
"""
V_offsets(ms::matrix_state) = begin
    if ms.lr==left
        if ms.ul ==lower
            o1_=1
            o2_=1
        else #upper
            o1_=0
            o2_=0
        end
    else #right
        if ms.ul ==lower
            o1_=0
            o2_=0
        else #upper
            o1_=1
            o2_=1
        end
    end
    V_offsets(o1_,o2_)
end


struct bond_spectrum
    spectrum::Vector{Float64}
    link_number::Int64
    bond_spectrum(s::ITensor,link::Int64) = begin
        @assert link>0
        @assert order(s)==2
        new(diag(array(s)),link)
    end
end

bond_spectrums = Vector{bond_spectrum} 

function max(s::bond_spectrum)::Float64 s.spectrum[1] end
function min(s::bond_spectrum)::Float64 s.spectrum[end] end
function max(ss::bond_spectrums)::Float64 
    ret=max(ss[1])
    for n in 2:length(ss)
        ms=max(ss[n]) 
        if ms>ret ret=ms end
    end
    return ret
end

function min(ss::bond_spectrums)::Float64 
    ret=min(ss[1])
    for n in 2:length(ss) 
        ms=min(ss[n])
        if ms<ret ret=ms end
    end
    return ret
end



function slice(A::ITensor,iv::IndexVal...)::ITensor
    iv_dagger=[dag(x.first)=>x.second for x in iv]
    return A*onehot(iv_dagger...)
end


"""
    assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)

Assign an operator to an element of the operator valued matrix W
    W[i1,i2]=op
"""
function assign!(W::ITensor,op::ITensor,ivs::IndexVal...)
    assign!(W,tensor(op),ivs...)
end 

function assign!(W::ITensor,op::DenseTensor{ElT,N},ivs::IndexVal...) where {ElT,N}
    iss=inds(op)
    for s in eachindval(iss)
        s2=[x.second for x in s]
        W[ivs...,s...]=op[s2...]
    end
end


function assign!(W::ITensor,op::BlockSparseTensor{ElT,N},ivs::IndexVal...) where {ElT,N}
    iss=inds(op)
    for b in eachnzblock(op)
        isv=[iss[i]=>b[i] for i in 1:length(b)]
        W[ivs...,isv...]=op[b][1] #not sure why we need [1] here
    end
end

"""
    function redim(i::Index,Dw::Int64)::Index
    
    Create an index with the same tags ans plev, but different dimension(s) and and id 
"""
function redim(i::Index,Dw::Int64...)::Index
    @assert length(Dw)==nblocks(i)
    if hasqns(i)
        j=0
        new_qns=[(j+=1;q.first=>Dw[j]) for q in space(i)]
        return Index(new_qns...;dir=dir(i),tags=tags(i),plev=plev(i))
    else
        return Index(Dw[1];tags=tags(i),plev=plev(i))
    end
end

"""
    set_scale!(RL::ITensor,Q::ITensor,off::V_offsets)

Fix the gauge freedom between QR/QL after a block respecting QR/QL decomposition. The
gauge fix is to ensure that either the top left or bottom right corner of `R` is 1.0. 
"""
function set_scale!(RL::ITensor,Q::ITensor,off::V_offsets)
    @assert order(RL)==2
    is=inds(RL)
    Dw1,Dw2=map(dim,is)
    i1= off.o1==0 ? 1 : Dw1
    i2= off.o2==0 ? 1 : Dw2
    scale=RL[is[1]=>i1,is[2]=>i2]
    @assert abs(scale)>1e-12
    RL./=scale
    Q.*=scale
end

include("util.jl")
include("hamiltonians.jl")
include("qx.jl")
include("characterization.jl") 
include("blocking.jl")
include("MPOpbc.jl")
include("orthogonalize.jl")
include("truncate.jl")


end
