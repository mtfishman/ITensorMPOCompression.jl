module ITensorMPOCompression

using ITensors
include("util.jl")
include("qx.jl")
# Write your package code here.

function assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)
    is=inds(op)
    for s in eachindval(is)
        W[i1,i2,s...]=op[s...]
    end
end


export ql,assign!


end
