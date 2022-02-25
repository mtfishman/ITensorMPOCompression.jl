module ITensorMPOCompression

using ITensors

export ql,lq,rq,assign!,getV,setV!,growRL,to_openbc,set_scale!,block_qx!,block_qx,canonical!,is_canonical
export tri_type,orth_type,matrix_state,upper,lower,none,left,right,parse_links
export has_pbc,is_regular_form,compress,compress!,getM,grow
export is_lower_regular_form,is_upper_regular_form,V_offsets
export detect_upper_lower,is_upper_lower

@enum tri_type  upper lower
@enum orth_type none left right

struct matrix_state
    ul::tri_type
    lr::orth_type
end

#
#  simple struct for encapsulating offsets for V-blocks
#
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

#
#  This is essentially table 2 in the notes.
#
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


function assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)
    is=inds(op)
    for s in eachindval(is)
        W[i1,i2,s...]=op[s...]
    end
end

set_scale!(RL::ITensor,Q::ITensor,o1::Int64,o2::Int64)=set_scale!(RL,Q,V_offsets(o1,o2))

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
