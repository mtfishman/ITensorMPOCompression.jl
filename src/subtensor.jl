using StaticArrays
import Base.range

struct IndexRange
  index::Index
  range::UnitRange{Int64}
  function IndexRange(i::Index, r::UnitRange)
    return new(i, r)
  end
end
irPair{T} = Pair{<:Index{T},UnitRange{Int64}} where {T}
irPairU{T} = Union{irPair{T},IndexRange} where {T}
IndexRange(ir::irPair{T}) where {T} = IndexRange(ir.first, ir.second)
IndexRange(ir::IndexRange) = IndexRange(ir.index, ir.range)

start(ir::IndexRange) = range(ir).start
starts(irs::Tuple{Vararg{UnitRange{Int64}}}) = map((ir) -> ir.start, irs)
stops(irs::Tuple{Vararg{UnitRange{Int64}}}) = map((ir) -> ir.stop, irs)
starts(irs::SVector{N,UnitRange{Int64}}) where {N} = map((ir) -> ir.start, irs)
stops(irs::SVector{N,UnitRange{Int64}}) where {N} = map((ir) -> ir.stop, irs)

range(ir::IndexRange) = ir.range
range(i::Index) = 1:dim(i)
ranges(irs::Tuple) = ntuple(i -> range(irs[i]), Val(length(irs)))
ranges(starts::SVector{N,Int64},stops::SVector{N,Int64}) where {N} = map(i -> starts[i]:stops[i], 1:N)
indices(irs::Tuple{Vararg{IndexRange}}) = map((ir) -> ir.index, irs)
indranges(ips::Tuple{Vararg{irPairU}}) = map((ip) -> IndexRange(ip), ips)

dim(ir::IndexRange) = dim(range(ir))
dim(r::UnitRange{Int64}) = r.stop - r.start + 1
dims(irs::Tuple{Vararg{IndexRange}}) = map((ir) -> dim(ir), irs)

redim(ip::irPair) = redim(IndexRange(ip))
redim(ir::IndexRange) = redim(ir.index, dim(ir), start(ir) - 1)
redim(irs::Tuple{Vararg{IndexRange}}) = map((ir) -> redim(ir), irs)

eachval(ir::IndexRange) = range(ir)
eachval(irs::Tuple{Vararg{IndexRange}}) = CartesianIndices(ranges(irs))

function eachindval(irs::Tuple{Vararg{IndexRange}})
  return (indices(irs) .=> Tuple(ns) for ns in eachval(irs))
end

#--------------------------------------------------------------------------------------
#
#  NDTensor level code which distinguishes between Dense and BlockSparse storage
#
function in_range(
  block_start::NTuple{N,Int64}, block_end::NTuple{N,Int64}, rs::UnitRange{Int64}...
) where {N}
  for i in eachindex(rs)
     (block_start[i] > rs[i].stop || block_end[i] < rs[i].start)  &&  return false
  end
  return true
end


function fix_ranges(
  dest_block_start::SVector{N,Int64},
  dest_block_end::SVector{N,Int64},
  rs::SVector{N,UnitRange{Int64}}
) where {N}
  return ranges(max.(0,starts(rs)-dest_block_start).+1,min.(dest_block_end,stops(rs))-dest_block_start.+1)
end

function get_subtensor(
  T::BlockSparseTensor{ElT,N}, new_inds, rs::UnitRange{Int64}...
) where {ElT,N}
  Ds = Vector{DenseTensor{ElT,N}}()
  bs = Vector{Block{N}}()

  for b in eachnzblock(T)
      blockT = blockview(T, b)
      if in_range(blockstart(T,b),blockend(T,b),rs...)# && !isnothing(blockT)
          rs1=fix_ranges(SVector(blockstart(T,b)),SVector(blockend(T,b)),SVector(rs))
          _,tb1=blockindex(T,starts(rs)...)
          tb1=CartesianIndex(ntuple(i->tb1[i]-1,N))
          push!(Ds,blockT[rs1...])
          push!(bs,CartesianIndex(b)-tb1)
      end
  end
  if length(Ds) == 0
    return BlockSparseTensor(new_inds)
  end
  T_sub = BlockSparseTensor(ElT, undef, bs, new_inds)
  for ib in eachindex(Ds)
    blockT_sub = bs[ib]
    blockview(T_sub, blockT_sub) .= Ds[ib]
  end
  return T_sub
end

function set_subtensor(
  T::BlockSparseTensor{ElT,N}, A::BlockSparseTensor{ElT,N}, rs::UnitRange{Int64}...
) where {ElT,N}
  insert = false
  rsa = ntuple(i -> 1:dim(inds(A)[i]), N)
  for ab in eachnzblock(A)
    blockA = blockview(A, ab)
    if in_range(blockstart(A, ab), blockend(A, ab), rsa...)
      iA = blockstart(A, ab)
      iT = ntuple(i -> iA[i] + rs[i].start - 1, N)
      index_within_block, tb = blockindex(T, Tuple(iT)...)
      blockT = blockview(T, tb)
      if blockT == nothing
        insertblock!(T, tb)
        blockT = blockview(T, tb)
        insert = true
      end
      dA = [dims(blockA)...]
      dT = [blockend(T, tb)...] - [index_within_block...] + fill(1, N)
      rs1 = ntuple(
        i -> index_within_block[i]:(index_within_block[i] + Base.min(dA[i], dT[i]) - 1), N
      )
      if length(findall(dA .> dT)) == 0
        blockT[rs1...] = blockA #Dense assignment for each block
      else
        rsa1 = ntuple(i -> (rsa[i].start):(rsa[i].start + Base.min(dA[i], dT[i]) - 1), N)
        blockT[rs1...] = blockA[rsa1...] #partial block Dense assignment for each block
        @error "Incomplete bloc transfer."
        @assert false
      end
    end
  end
end


function set_subtensor(
  T::DiagBlockSparseTensor{ElT,N}, A::DiagBlockSparseTensor{ElT,N}, rs::UnitRange{Int64}...
) where {ElT,N}
  _,tb1=blockindex(T,starts(rs)...)
  tb1=CartesianIndex(ntuple(i->tb1[i]-1,N))
  rsa = ntuple(i -> (rs[i].start - tb1[i]):(rs[i].stop - tb1[i]), N)
  for ab in eachnzblock(A)
    if in_range(blockstart(A, ab), blockend(A, ab), rsa...)
      it = blockstart(A, ab)
      it = ntuple(i -> it[i] + tb1[i], N)
      _, tb = blockindex(T, Tuple(it)...)
      blockT = blockview(T, tb)
      blockA = blockview(A, ab)
      rs1 = fix_ranges(SVector(blockstart(T,tb)),SVector(blockend(T,tb)),SVector(rs))
      blockT[rs1...] = blockA #Diag assignment for each block
    end
  end
end

function set_subtensor(
  T::DiagTensor{ElT}, A::DiagTensor{ElT}, rs::UnitRange{Int64}...
) where {ElT}
  if !all(y -> y == rs[1], rs)
    @error("set_subtensor(DiagTensor): All ranges must be the same, rs=$(rs).")
  end
  N = length(rs)
  #only assign along the diagonal.
  for i in rs[1]
    is = ntuple(i1 -> i, N)
    js = ntuple(j1 -> i - rs[1].start + 1, N)
    T[is...] = A[js...]
  end
end

function setindex!(
  T::BlockSparseTensor{ElT,N}, A::BlockSparseTensor{ElT,N}, irs::Vararg{UnitRange{Int64},N}
) where {ElT,N}
  return set_subtensor(T, A, irs...)
end
function setindex!(
  T::DiagTensor{ElT,N}, A::DiagTensor{ElT,N}, irs::Vararg{UnitRange{Int64},N}
) where {ElT,N}
  return set_subtensor(T, A, irs...)
end
function setindex!(
  T::DiagBlockSparseTensor{ElT,N},
  A::DiagBlockSparseTensor{ElT,N},
  irs::Vararg{UnitRange{Int64},N},
) where {ElT,N}
  return set_subtensor(T, A, irs...)
end

#------------------------------------------------------------------------------------
#
#  ITensor level wrappers which allows us to handle the indices in a different manner
#  depending on dense/block-sparse
#
function get_subtensor_wrapper(
  T::DenseTensor{ElT,N}, new_inds, rs::UnitRange{Int64}...
) where {ElT,N}
  return ITensor(T[rs...], new_inds)
end

function get_subtensor_wrapper(
  T::BlockSparseTensor{ElT,N}, new_inds, rs::UnitRange{Int64}...
) where {ElT,N}
  return ITensor(get_subtensor(T, new_inds, rs...))
end

function permute(indsT::T, irs::IndexRange...) where {T<:(Tuple{Vararg{T,N}} where {N,T})}
  ispec = indices(irs) #indices caller specified ranges for
  inot = Tuple(noncommoninds(indsT, ispec)) #indices not specified by caller
  isort = ispec..., inot... #all indices sorted so user specified ones are first.
  isort_sub = redim(irs)..., inot... #all indices for subtensor
  p = getperm(indsT, ntuple(n -> isort[n], length(isort)))
  if !isperm(p)
    @show p ispec inot indsT isort
  end
  return NDTensors.permute(isort_sub, p), NDTensors.permute((ranges(irs)..., ranges(inot)...), p)
end
#
#  Use NDTensors T[3:4,1:3,1:6...] syntax to extract the subtensor.
#
function get_subtensor_ND(T::ITensor, irs::IndexRange...)
  isub, rsub = permute(inds(T), irs...) #get indices and ranges for the subtensor
  return get_subtensor_wrapper(tensor(T), isub, rsub...) #virtual dispatch based on Dense or BlockSparse
end

function match_tagplev(i1::Index, i2s::Index...)
  for i in eachindex(i2s)
    #@show i  tags(i1) tags(i2s[i]) plev(i1) plev(i2s[i])
    if tags(i1) == tags(i2s[i]) && plev(i1) == plev(i2s[i])
      return i
    end
  end
  @error("match_tagplev: unable to find tag/plev for $i1 in Index set $i2s.")
  return nothing
end

function getperm_tagplev(s1, s2)
  N = length(s1)
  r = Vector{Int}(undef, N)
  return map!(i -> match_tagplev(s1[i], s2...), r, 1:length(s1))
end
#
#
#  Permute is1 to be in the order of is2. 
#  BUT: match based on tags and plevs instread of IDs
#
function permute_tagplev(is1, is2)
  length(is1) != length(is2) && throw(
    ArgumentError(
      "length of first index set, $(length(is1)) does not match length of second index set, $(length(is2))",
    ),
  )
  perm = getperm_tagplev(is1, is2)
  #@show perm is1 is2
  return is1[invperm(perm)]
end

#  This operation is non trivial because we need to
#   1) Establish that the non-requested (not in irs) indices are identical (same ID) between T & A
#   2) Establish that requested (in irs) indices have the same tags and 
#       prime levels (dims are different so they cannot possibly have the same IDs)
#   3) Use a combination of tags/primes (for irs indices) or IDs (for non irs indices) to permut 
#      the indices of A prior to conversion to a Tensor.
#
function set_subtensor_ND(T::ITensor, A::ITensor, irs::IndexRange...)
  ireqT = indices(irs) #indices caller requested ranges for
  inotT = Tuple(noncommoninds(inds(T), ireqT)) #indices not requested by caller
  ireqA = Tuple(noncommoninds(inds(A), inotT)) #Get the requested indices for a
  inotA = Tuple(noncommoninds(inds(A), ireqA))
  p=getperm(inotA,ntuple(n -> inotT[n], length(inotT)))
  inotA_sorted=NDTensors.permute(inotA,p)
  if length(ireqT) != length(ireqA) || inotA_sorted != inotT
    @show inotA inotT ireqT ireqA inotA != inotT length(ireqT) length(ireqA)
    @error(
      "subtensor assign, incompatable indices\ndestination inds=$(inds(T)),\n source inds=$(inds(A))."
    )
    @assert(false)
  end
  isortT = ireqT..., inotT... #all indices sorted so user specified ones are first.
  p = getperm(inds(T), ntuple(n -> isortT[n], length(isortT))) # find p such that isort[p]==inds(T)
  rsortT = NDTensors.permute((ranges(irs)..., ranges(inotT)...), p) #sorted ranges for T
  ireqAp = permute_tagplev(ireqA, ireqT) #match based on tags & plev, NOT IDs since dims are different.
  isortA = NDTensors.permute((ireqAp..., inotA...), p) #inotA is the same inotT, using inotA here for a less confusing read.
  Ap = permute(A, isortA...; allow_alias=true)
  return tensor(T)[rsortT...] = tensor(Ap)
end
function set_subtensor_ND(T::ITensor, v::Number, irs::IndexRange...)
  ireqT = indices(irs) #indices caller requested ranges for
  inotT = Tuple(noncommoninds(inds(T), ireqT)) #indices not requested by caller
  isortT = ireqT..., inotT... #all indices sorted so user specified ones are first.
  p = getperm(inds(T), ntuple(n -> isortT[n], length(isortT))) # find p such that isort[p]==inds(T)
  rsortT = permute((ranges(irs)..., ranges(inotT)...), p) #sorted ranges for T
  return tensor(T)[rsortT...] .= v
end

getindex(T::ITensor, irs::Vararg{IndexRange,N}) where {N} = get_subtensor_ND(T, irs...)
function getindex(T::ITensor, irs::Vararg{irPairU,N}) where {N}
  return get_subtensor_ND(T, indranges(irs)...)
end
function setindex!(T::ITensor, A::ITensor, irs::Vararg{IndexRange,N}) where {N}
  return set_subtensor_ND(T, A, irs...)
end
function setindex!(T::ITensor, A::ITensor, irs::Vararg{irPairU,N}) where {N}
  return set_subtensor_ND(T, A, indranges(irs)...)
end
function setindex!(T::ITensor, v::Number, irs::Vararg{IndexRange,N}) where {N}
  return set_subtensor_ND(T, v, irs...)
end
function setindex!(T::ITensor, v::Number, irs::Vararg{irPairU,N}) where {N}
  return set_subtensor_ND(T, v, indranges(irs)...)
end
