module ITensorMPOCompression

using ITensors
using NDTensors

import ITensors: addqns, isortho, orthocenter, setinds , linkind, data, permute, checkflux
import ITensors: dim, dims, trivial_space, eachindval, eachval, getindex, setindex!
import ITensors: truncate!, truncate, orthogonalize, orthogonalize!

import ITensors: QNIndex, QNBlocks, Indices, AbstractMPS, DenseTensor, BlockSparseTensor,DiagTensor, tensor

import NDTensors: getperm, BlockDim, blockstart, blockend
  

import Base: similar, reverse, transpose
 
# reg_form and orth_type values and functions
export upper, lower, left, right, mirror, flip
# lots of characterization functions
export is_regular_form, isortho, check_ortho, detect_regular_form, @checkflux
# MPO bond dimensions and bond spectrum
export get_Dw, min, max
export bond_spectrums
# primary operations
#export orthogonalize!, truncate, truncate!
# Display helpers
export @pprint, pprint, show_directions
# Wrapped MPO types
export reg_form_MPO

macro mpoc_assert(ex)
  esc(:($Base.@assert $ex))
end

function mpoc_checkflux(::Union{DenseTensor,DiagTensor})
  # No-op
end
function mpoc_checkflux(T::Union{BlockSparseTensor,DiagBlockSparseTensor})
  return checkflux(T)
end

macro checkflux(T)
  return esc(:(ITensorMPOCompression.mpoc_checkflux(NDTensors.tensor($T))))
end

default_eps = 1e-14 #for characterization routines, floats abs()<default_eps are considered to be zero.

@doc """
    @enum reg_form  upper lower
    
Indicates that an MPO or operator-valued matrix has either an `upper` or `lower` regular form.
This becomes non-trival for rectangular matrices.
See also [`detect_regular_form`](@ref) and related functions
"""
@enum reg_form upper lower

@doc """
    @enum orth_type left right

Indicates that an MPO matrix satisfies the conditions for `left` or `right` canonical form     
"""
@enum orth_type left right

# """
#     mirror(lr::orth_type)::orth_type
#     returns this mirror of lr.  `left`->`right` and `right`->`left`
# """
function mirror(lr::orth_type)::orth_type
  if lr == left
    ret = right
  else #must be right
    ret = left
  end
  return ret
end

mirror(ul::reg_form) = ul == lower ? upper : lower

bond_spectrums = Vector{Spectrum}

function Base.max(s::Spectrum)::Float64
  return sqrt(eigs(s)[1])
end
function Base.min(s::Spectrum)::Float64
  return sqrt(eigs(s)[end])
end
function Base.max(ss::bond_spectrums)::Float64
  ret = max(ss[1])
  for n in 2:length(ss)
    ms = max(ss[n])
    if ms > ret
      ret = ms
    end
  end
  return ret
end

function Base.min(ss::bond_spectrums)::Float64
  ret = min(ss[1])
  for n in 2:length(ss)
    ms = min(ss[n])
    if ms < ret
      ret = ms
    end
  end
  return ret
end

function slice(A::ITensor, iv::IndexVal...)::ITensor
  iv_dagger = [dag(x.first) => x.second for x in iv]
  return A * onehot(iv_dagger...)
end

"""
    assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)

Assign an operator to an element of the operator valued matrix W
    W[i1,i2]=op
"""
function assign!(W::ITensor, op::ITensor, ivs::IndexVal...)
  is=commoninds(W,inds(op))
  op_sort = permute(op, is...; allow_alias=true)
  return assign!(W, tensor(op_sort), ivs...)
end

function assign!(W::ITensor, op::DenseTensor, ivs::IndexVal...)
  iss = inds(op)
  for s in eachindval(iss)
    s2 = [x.second for x in s]
    W[ivs..., s...] = op[s2...]
  end
end

function assign!(W::ITensor, op::BlockSparseTensor, ivs::IndexVal...) 
  # @show typeof(storage(W))
  if isempty(W)
    iss = inds(op)
    for b in eachnzblock(op)
      isv = [iss[i] => b[i] for i in 1:length(b)]
      W[ivs..., isv...] = op[b][1] #not sure why we need [1] here
    end
  else
    assign!(tensor(W),op,ivs...)
  end
    
end

function assign!(Wt::BlockSparseTensor, op::BlockSparseTensor, ivs::IndexVal...) 
  N=ndims(Wt)
  # @show nzblocks(Wt)
  iss = inds(op)
  for b in eachnzblock(op)
    bs= NDTensors.blockstart(op,b)
    isv = [iss[i] => bs[i] for i in 1:length(b)]
    is=(ivs..., isv...)
    p = NDTensors.getperm(inds(Wt), ntuple(n -> ind(@inbounds is[n]), Val(N)))
    Is=NDTensors.permute(ntuple(n -> val(@inbounds is[n]), Val(N)), p)
    _, wb=blockindex(Wt, Is...)
    # @show bs isv is Is wb 
    blockW = blockview(Wt, wb)
    if isnothing(blockW)
      insertblock!(Wt, wb)
      blockW = blockview(Wt, wb)
    end
    # @show blockW blockview(op,b)
    blockW.= blockview(op,b)
   
  end
end


"""
    function redim(i::Index,Dw::Int64)::Index
    
    Create an index with the same tags ans plev, but different dimension(s) and and id 
"""
#
# Build and increased QN space with padding at the begining and end.
# Use the sample qns argument get the correct blocks for padding.
# Ability to split blocks is not needed, therefore not supported.
#
function redim(iq::QNIndex, pad1::Int64, pad2::Int64, qns::QNBlocks)
  @assert pad1 == blockdim(qns[1]) #Splitting blocks not supported
  @assert pad2 == blockdim(qns[end]) #Splitting blocks not supported
  qnsp = [qns[1], space(iq)..., qns[end]] #creat the new space
  return Index(qnsp; tags=tags(iq), plev=plev(iq), dir=dir(iq)) #create new index.
end

function redim(i::Index, pad1::Int64, pad2::Int64, ::Int64)
  #@assert dim(i) + pad1 + pad2 <= Dw 
  return Index(dim(i) + pad1 + pad2; tags=tags(i), plev=plev(i), dir=dir(i)) #create new index.
end

#
#  Build a reduced QN space from offset->Dw+offset, possibly splitting QNBlocks at the
#  begining and end of the space.
#
function redim(iq::QNIndex, Dw::Int64, offset::Int64)
  # println("---------------------")
  # @show iq Dw offset
  @assert dim(iq) - offset >= Dw 
  qns=copy(space(iq))
  qns1=QNBlocks()
  
  is=1
  for i in 0:length(qns) #starting at 0 quickly dispenses with the offset==0 case.
    _Dw=dim(qns[1:i])
    if _Dw==offset
      is=i+1
      break
    elseif _Dw>offset
      excess = _Dw-offset 
      if excess==blockdim(qns[i])
        is=i #no need to split the block.  Add this block below.
      else  #we need to split this block and add it here.
        qn_split=qn(qns[i])=>excess
        #push!(qns1,qn_split)
        qns[i]=qn_split
        is=i
      end
      break
    end
  end
  # @show qns1 is
  
  for i in is:length(qns)
    _Dw=dim(qns[is:i])
    if _Dw==Dw #no need to split .. and we are done.
      push!(qns1,qns[i])
      break;
    elseif _Dw>Dw #Need to split
      excess = _Dw-Dw #How much space to leave
      bdim=blockdim(qns[i])
      qn_split=qn(qns[i])=>bdim-excess
      push!(qns1,qn_split)
      break 
    else
      push!(qns1,qns[i])
    end
  end
  # @show qns1
  @mpoc_assert Dw==dim(qns1)
  return Index(qns1; tags=tags(iq), plev=plev(iq), dir=dir(iq)) #create new index.
end

function redim(i::Index, Dw::Int64, ::Int64)
  return Index(Dw; tags=tags(i), plev=plev(i)) #create new index.
end

function Base.reverse(i::QNIndex)
  return Index(Base.reverse(space(i)); dir=dir(i), tags=tags(i), plev=plev(i))
end
function Base.reverse(i::Index)
  return Index(space(i); tags=tags(i), plev=plev(i))
end


function G_transpose(i::Index, iu::Index)
  D = dim(i)
  @mpoc_assert D == dim(iu)
  G = ITensor(0.0, dag(i), iu)
  for n in 1:D
    G[i => n, iu => D + 1 - n] = 1.0
  end
  return G
end


include("subtensor.jl")
include("reg_form_Op.jl")
include("blocking.jl")
include("reg_form_MPO.jl")
include("util.jl")
include("gauge_fix.jl")
include("qx.jl")
include("characterization.jl")
include("orthogonalize.jl")
include("truncate.jl")

end
