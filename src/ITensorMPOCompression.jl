module ITensorMPOCompression

using ITensors

export ql,lq,assign!,getV,setV!,growRL,to_openbc,set_scale!,block_qx!,block_qx,canonical!,is_canonical
export tri_type,orth_type,matrix_state,full,upper,lower,none,left,right,parse_links
export detect_upper_lower,has_pbc,is_regular_form,compress,getM,grow
export is_lower_regular_form,is_upper_regular_form

@enum tri_type  full upper lower diagonal
@enum orth_type none left right

struct matrix_state
    ul::tri_type
    lr::orth_type
end


function assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)
    is=inds(op)
    for s in eachindval(is)
        W[i1,i2,s...]=op[s...]
    end
end


function set_scale!(RL::ITensor,Q::ITensor,o1::Int64,o2::Int64)
    @assert o1==0 || o1==1
    @assert o2==0 || o2==1
    @assert order(RL)==2
    is=inds(RL)
    Dw1,Dw2=map(dim,is)
    i1= o1==0 ? 1 : Dw1
    i2= o2==0 ? 1 : Dw2
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
