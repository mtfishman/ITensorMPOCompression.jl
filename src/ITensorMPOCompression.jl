module ITensorMPOCompression

using ITensors
using ITensors.NDTensors
using ITensorInfiniteMPS

import ITensors: QNIndex, addqns, rq, AbstractMPS, isortho, orthocenter
import ITensors.BlockSparseTensor,ITensors.DenseTensor,ITensors.tensor

import ITensorInfiniteMPS: AbstractInfiniteMPS
import Base: similar

export block_qx #qx related
export slice,assign!,redim #operator handling
export getV,setV,growRL,V_offsets #blocking related
# lots of characterization functions
export reg_form,orth_type,matrix_state,upper,lower,left,right,mirror
export parse_links,parse_link,parse_site,is_regular_form,getM,grow,detect_regular_form
export is_lower_regular_form,is_upper_regular_form
export detect_upper_lower,is_upper_lower,sweep
export isortho, check_ortho
# Hamiltonian related
export make_transIsing_MPO,make_Heisenberg_AutoMPO,make_transIsing_AutoMPO,to_openbc,get_lr
export make_transIsing_iMPO,make_2body_AutoMPO
export fast_GS,make_3body_MPO,make_1body_op,make_2body_op,make_3body_op,add_ops,make_3body_AutoMPO,make_2body_sum,make_2body_MPO
# MPO and bond spectrum
export get_Dw,max_Dw,min,max
export bond_spectrums

export orthogonalize!,truncate,truncate! #the punchline
export @pprint,pprint
#  subtebsor related
export IndexRange, indices, range, ranges, getperm, permute, start

macro mpoc_assert(ex)
    esc(:($Base.@assert $ex))
end

default_eps=1e-14 #for characterization routines, floats abs()<default_eps are considered to be zero.

"""
    @enum reg_form  upper lower
    
Indicates that an MPO or operator-valued matrix has either an `upper` or `lower` regular form.
This becomes non-trival for rectangular matrices.
See also [`detect_regular_form`](@ref) and related functions
"""
@enum reg_form  upper lower 

"""
    @enum orth_type left right

Indicates that an MPO matrix satisfies the conditions for `left` or `right` canonical form     
"""
@enum orth_type left right

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
        @mpoc_assert o1_==0 || o1_==1
        @mpoc_assert o2_==0 || o2_==1 
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

bond_spectrums = Vector{Spectrum}

function max(s::Spectrum)::Float64 sqrt(eigs(s)[1]) end
function min(s::Spectrum)::Float64 sqrt(eigs(s)[end]) end
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
function redim(i::Index,Dw::Int64,offset::Int64=0)::Index
    if hasqns(i)
        qns=copy(space(i))
        if Dw>dim(i)
            #
            # We need grow the space.  If there are multiple QNs, where to add the space?
            # Lets add to the end for now.
            #
            @mpoc_assert offset==0 #not ready to handle this case yet.
            delta=Dw-dim(i)
            dq=qns[end].second #dim of space for last QN
            qns[end]=qns[end].first=>dq+delta
            return Index(qns;dir=dir(i),tags=tags(i),plev=plev(i))
        else
            start_offset=offset
            end_offset=dim(i)-Dw-offset
            #
            #  Set spaces from 1<=d<=1+offset and Dw<d<D to zero 
            #
            for n in eachindex(qns)
                dq=qns[n].second #dim of space
                d_remain=Base.max(0,dq-start_offset) #How much space to leave
                qns[n]=qns[n].first=>d_remain #update dim of QN
                start_offset-=(dq-d_remain) #decrement start_offset 
                if start_offset==0 break end #are we done?
                @mpoc_assert start_offset>0 #sanity check
            end

            for n in reverse(eachindex(qns))
                dq=qns[n].second
                d_remain=Base.max(0,dq-end_offset) #How much space to leave
                qns[n]=qns[n].first=>d_remain #update dim of QN
                end_offset-=(dq-d_remain) #decrement end_offset 
                if end_offset==0 break end #are we done?
                @mpoc_assert end_offset>0 #sanity check
            end
        
            #
            #  now trim out all the QNs with dim(q)==0
            #
            qns_trim=Pair{QN, Int64}[]
            for q in qns
                if q.second>0
                    append!(qns_trim,[q])
                end
            end
            @mpoc_assert Dw==sum(map((q)->q.second,qns_trim))
            return Index(qns_trim;dir=dir(i),tags=tags(i),plev=plev(i))
        end #if Dw>dim(i)
    else
        return Index(Dw;tags=tags(i),plev=plev(i))
    end
end

"""
    set_scale!(RL::ITensor,Q::ITensor,off::V_offsets)

Fix the gauge freedom between QR/QL after a block respecting QR/QL decomposition. The
gauge fix is to ensure that either the top left or bottom right corner of `R` is 1.0. 
"""
function set_scale!(RL::ITensor,Q::ITensor,off::V_offsets)
    @mpoc_assert order(RL)==2
    is=inds(RL)
    Dw1,Dw2=map(dim,is)
    i1= off.o1==0 ? 1 : Dw1
    i2= off.o2==0 ? 1 : Dw2
    scale=RL[is[1]=>i1,is[2]=>i2]
    @mpoc_assert abs(scale)>1e-12
    RL./=scale
    Q.*=scale
end

include("subtensor.jl")
include("util.jl")
include("hamiltonians.jl")
include("hamiltonians_AutoMPO.jl")
include("hamiltonians_infinite.jl")
include("qx.jl")
include("characterization.jl") 
include("blocking.jl")
include("orthogonalize.jl")
include("truncate.jl")


end
