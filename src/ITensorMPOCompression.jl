module ITensorMPOCompression

using ITensors
using ITensors.NDTensors
using ITensorInfiniteMPS

import ITensors: QNIndex, addqns, rq, AbstractMPS, isortho, orthocenter, Indices, linkind
import ITensors.BlockSparseTensor,
  ITensors.DenseTensor, ITensors.DiagTensor, ITensors.tensor

import ITensorInfiniteMPS: AbstractInfiniteMPS, translatecell
import Base: similar, reverse, transpose
 

export block_qx #qx related
export slice, assign!  #operator handling
export getV, setV, growRL, V_offsets #blocking related
export my_similar
# lots of characterization functions
export reg_form, orth_type, upper, lower, left, right, mirror, flip
export parse_links,
  parse_link, parse_site, is_regular_form, build_R⎖, grow, detect_regular_form
export is_lower_regular_form, is_upper_regular_form
export detect_upper_lower, is_upper_lower, sweep
export isortho, check_ortho
# Hamiltonian related
export make_transIsing_MPO,
  make_Heisenberg_AutoMPO, make_transIsing_AutoMPO, to_openbc, get_lr
export make_transIsing_iMPO, make_2body_AutoMPO, make_Hubbard_AutoMPO, make_Heisenberg_MPO
export fast_GS,
  make_3body_MPO,
  make_1body_op,
  make_2body_op,
  make_3body_op,
  add_ops,
  make_3body_AutoMPO,
  make_2body_sum,
  make_2body_MPO
export make_transIsing_AutoiMPO, make_Heisenberg_AutoiMPO, make_Hubbard_AutoiMPO
export to_upper!, make_AutoiMPO
# MPO and bond spectrum
export get_Dw, maxlinkdim, min, max
export bond_spectrums

export orthogonalize!, truncate, truncate! #the punchline
export @pprint, pprint, @mpoc_assert, show_directions
#  subtebsor related
export IndexRange, indices, range, ranges, getperm, permute, start
#
#  New ac_qx
#
export reg_form_MPO, extract_blocks, is_gauge_fixed, gauge_fix!, ac_qx, ac_orthogonalize!
export reg_form_iMPO, transpose, check, check_ortho
export InfiniteCanonicalMPO

macro mpoc_assert(ex)
  esc(:($Base.@assert $ex))
end

function mpoc_checkflux(::Union{DenseTensor,DiagTensor})
  # No-op
end
function mpoc_checkflux(T::Union{BlockSparseTensor,DiagBlockSparseTensor})
  return ITensors.checkflux(T)
end

macro checkflux(T)
  return esc(:(mpoc_checkflux(tensor($T))))
end

default_eps = 1e-14 #for characterization routines, floats abs()<default_eps are considered to be zero.

"""
    @enum reg_form  upper lower
    
Indicates that an MPO or operator-valued matrix has either an `upper` or `lower` regular form.
This becomes non-trival for rectangular matrices.
See also [`detect_regular_form`](@ref) and related functions
"""
@enum reg_form upper lower

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
  if lr == left
    ret = right
  else #must be right
    ret = left
  end
  return ret
end

flip(ul::reg_form) = ul == lower ? upper : lower

bond_spectrums = Vector{Spectrum}

function max(s::Spectrum)::Float64
  return sqrt(eigs(s)[1])
end
function min(s::Spectrum)::Float64
  return sqrt(eigs(s)[end])
end
function max(ss::bond_spectrums)::Float64
  ret = max(ss[1])
  for n in 2:length(ss)
    ms = max(ss[n])
    if ms > ret
      ret = ms
    end
  end
  return ret
end

function min(ss::bond_spectrums)::Float64
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
  return assign!(W, tensor(op), ivs...)
end

function assign!(W::ITensor, op::DenseTensor{ElT,N}, ivs::IndexVal...) where {ElT,N}
  iss = inds(op)
  for s in eachindval(iss)
    s2 = [x.second for x in s]
    W[ivs..., s...] = op[s2...]
  end
end

function assign!(W::ITensor, op::BlockSparseTensor{ElT,N}, ivs::IndexVal...) where {ElT,N}
  iss = inds(op)
  for b in eachnzblock(op)
    isv = [iss[i] => b[i] for i in 1:length(b)]
    W[ivs..., isv...] = op[b][1] #not sure why we need [1] here
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
function redim(iq::ITensors.QNIndex, pad1::Int64, pad2::Int64, qns::ITensors.QNBlocks)
  @assert pad1 == blockdim(qns[1]) #Splitting blocks not supported
  @assert pad2 == blockdim(qns[end]) #Splitting blocks not supported
  qnsp = [qns[1], space(iq)..., qns[end]] #creat the new space
  return Index(qnsp; tags=tags(iq), plev=plev(iq), dir=dir(iq)) #create new index.
end

function redim(i::Index, pad1::Int64, pad2::Int64, Dw::Int64)
  @assert dim(i) + pad1 + pad2 <= Dw 
  return Index(dim(i) + pad1 + pad2; tags=tags(i), plev=plev(i), dir=dir(i)) #create new index.
end

#
#  Build a reduced QN space from offset->Dw+offset, possibly splitting QNBlocks at the
#  begining and end of the space.
#
function redim(iq::ITensors.QNIndex, Dw::Int64, offset::Int64)
  # println("---------------------")
  # @show iq Dw offset
  @assert dim(iq) - offset >= Dw 
  qns=copy(space(iq))
  qns1=ITensors.QNBlocks()
  
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

function redim(i::Index, Dw::Int64, offset::Int64=0)
  return Index(Dw; tags=tags(i), plev=plev(i)) #create new index.
end


include("subtensor.jl")
include("util.jl")
include("reg_form.jl")
include("blocking.jl")
include("gauge_fix.jl")
include("hamiltonians.jl")
include("hamiltonians_AutoMPO.jl")
include("hamiltonians_infinite.jl")
include("qx.jl")
include("characterization.jl")
include("orthogonalize.jl")
include("truncate.jl")
include("infinite_canonical_mpo.jl")

end
