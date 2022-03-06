module ITensorMPOCompression

using ITensors

export ql,lq,rq,assign!,getV,setV,growRL,to_openbc,set_scale!,block_qx,orthogonalize!,is_canonical
export tri_type,orth_type,matrix_state,upper,lower,none,left,right,mirror,parse_links
export has_pbc,is_regular_form,truncate!,getM,grow
export is_lower_regular_form,is_upper_regular_form,V_offsets
export detect_upper_lower,is_upper_lower

"""
    @enum tri_type  upper lower
    Indicates if an MPO matrix has either an `upper` or `lower` triangular form.
    This becomes non-trival for rectangular matrices.
    See also [`detect_upper_lower`](@ref)
"""
@enum tri_type  upper lower 

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
     Indicates both the `orth_type` and `tri_type` of an MPO
     See also [`orth_type`](@ref) [`tri_type`](@ref)
 """
struct matrix_state
    ul::tri_type
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

"""
    assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)

Assign an operator to an element of the operator valued matrix W
    W[i1,i2]=op
"""
function assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)
    is=inds(op)
    for s in eachindval(is)
        W[i1,i2,s...]=op[s...]
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
include("qx.jl")
include("characterization.jl") 
include("blocking.jl")
include("MPOpbc.jl")
include("canonical.jl")
include("compress.jl")


end
