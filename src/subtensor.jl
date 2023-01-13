import ITensors: dim, dims, DenseTensor, eachindval, eachval, getindex, setindex!
import NDTensors: getperm, permute, BlockDim
import ITensorMPOCompression: redim
import Base.range


struct IndexRange
    index::Index
    range::UnitRange{Int64}
    function IndexRange(i::Index,r::UnitRange)
        return new(i,r)
    end
end
irPair{T}=Pair{<:Index{T},UnitRange{Int64}} where {T}
irPairU{T}=Union{irPair{T},IndexRange} where {T}
IndexRange(ir::irPair{T}) where {T} =IndexRange(ir.first,ir.second)
IndexRange(ir::IndexRange)=IndexRange(ir.index,ir.range)

start(ir::IndexRange)=range(ir).start
range(ir::IndexRange)=ir.range
range(i::Index)=1:dim(i)
ranges(irs::Tuple) = ntuple(i -> range(irs[i]), Val(length(irs)))
indices(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ir.index ,irs)
indranges(ips::Tuple{Vararg{irPairU}}) = map((ip)->IndexRange(ip) ,ips)

dim(ir::IndexRange)=dim(range(ir))
dim(r::UnitRange{Int64})=r.stop-r.start+1
dims(irs::Tuple{Vararg{IndexRange}})=map((ir)->dim(ir),irs)
redim(irs::Tuple{Vararg{IndexRange}}) = map((ir)->redim(ir.index,dim(ir)) ,irs)


eachval(ir::IndexRange) = range(ir)
eachval(irs::Tuple{Vararg{IndexRange}}) = CartesianIndices(ranges(irs))

eachindval(irs::Tuple{Vararg{IndexRange}}) = (indices(irs).=> Tuple(ns) for ns in eachval(irs))


#--------------------------------------------------------------------------------------
#
#  NDTensor level code which distinguishes between Dense and BlockSparse storage
#

function fix_ranges(ds::NTuple{N, Int64},rs::UnitRange{Int64}...) where {N}
    rs1=Vector{UnitRange{Int64}}(undef,N)
    for i in eachindex(rs1)
        if ds[i]==1
            rs1[i]=1:1 
        else
            rs1[i]=rs[i]
        end 
    end
    return Tuple(rs1)
end

function get_subtensor(T::BlockSparseTensor{ElT,N},new_inds,rs::UnitRange{Int64}...) where {ElT,N}
    Ds = Vector{DenseTensor{ElT,N}}(undef, nnzblocks(T))
    for (jj, b) in enumerate(eachnzblock(T))
        blockT = blockview(T, b)
        rs1=fix_ranges(dims(blockT),rs...)
        Ds[jj]=blockT[rs1...] #dense subtensor
    end
    #
    #  JR: All attempts at building the new indices here at the NDTensors level failed.
    #  The only thing I could make work was to pass the new indices down from the ITensors
    #  level and use those. 
    #
    T_sub = BlockSparseTensor(ElT, undef, nzblocks(T), new_inds)
    for ib in eachindex(Ds)
        blockT_sub = nzblocks(T_sub)[ib]
        blockview(T_sub, blockT_sub) .= Ds[ib]
    end
    return T_sub
end

function set_subtensor(T::BlockSparseTensor{ElT,N},A::BlockSparseTensor{ElT,N},rs::UnitRange{Int64}...) where {ElT,N}
    @assert nzblocks(T)==nzblocks(A)
    for (tb,ab) in zip(eachnzblock(T),eachnzblock(A))
        blockT = blockview(T, tb)
        blockA = blockview(A, ab)
        rs1=fix_ranges(dims(blockT),rs...)
        blockT[rs1...]=blockA #Dense assignment for each block
    end
end
setindex!(T::BlockSparseTensor{ElT,N},A::BlockSparseTensor{ElT,N},irs::Vararg{UnitRange{Int64},N}) where {ElT,N} = set_subtensor(T,A,irs...)


#------------------------------------------------------------------------------------
#
#  ITensor level wrappers which allows us to handle the indices in a different manner
#  depending on dense/block-sparse
#
function get_subtensor_wrapper(T::DenseTensor{ElT,N},new_inds,rs::UnitRange{Int64}...) where {ElT,N}
    return ITensor(T[rs...],new_inds)
end

function get_subtensor_wrapper(T::BlockSparseTensor{ElT,N},new_inds,rs::UnitRange{Int64}...) where {ElT,N}
    return ITensor(get_subtensor(T,new_inds,rs...))
end


function permute(indsT::T,irs::IndexRange...) where T<:(Tuple{Vararg{T, N}} where {N, T})
    ispec=indices(irs) #indices caller specified ranges for
    inot=Tuple(noncommoninds(indsT,ispec)) #indices not specified by caller
    isort=ispec...,inot... #all indices sorted so user specified ones are first.
    isort_sub=redim(irs)...,inot... #all indices for subtensor
    p=getperm(indsT, ntuple(n -> isort[n], length(isort)))
    #@show p
    return permute(isort_sub,p),permute((ranges(irs)...,ranges(inot)...),p)
end
#
#  Use NDTensors T[3:4,1:3,1:6...] syntax to extract the subtensor.
#
function get_subtensor_ND(T::ITensor,irs::IndexRange...)
    isub,rsub=permute(inds(T),irs...) #get indices and ranges for the subtensor
    return get_subtensor_wrapper(tensor(T),isub,rsub...) #virtual dispatch based on Dense or BlockSparse
end

function match_tagplev(i1::Index,i2s::Index...)
    for i in eachindex(i2s)
        #@show i  tags(i1) tags(i2s[i]) plev(i1) plev(i2s[i])
        if tags(i1)==tags(i2s[i]) && plev(i1)==plev(i2s[i])
            return i
        end
    end
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
function permute_tagplev(is1,is2)
    length(is1) != length(is2) && throw(
        ArgumentError(
        "length of first index set, $(length(is1)) does not match length of second index set, $(length(is2))",
        ),
    ) 
    perm=getperm_tagplev(is1,is2)
    return is1[invperm(perm)]
end
#  This operation is non trivial because we need to
#   1) Establish that the non-requested (not in irs) indices are identical (same ID) between T & A
#   2) Establish that requested (in irs) indices have the same tags and 
#       prime levels (dims are different so they cannot possibly have the same IDs)
#   3) Use a combination of tags/primes (for irs indices) or IDs (for non irs indices) to permut 
#      the indices of A prior to conversion to a Tensor.
#
function set_subtensor_ND(T::ITensor, A::ITensor,irs::IndexRange...)
    ireqT=indices(irs) #indices caller requested ranges for
    inotT=Tuple(noncommoninds(inds(T),ireqT)) #indices not requested by caller
    ireqA=Tuple(noncommoninds(inds(A),inotT)) #Get the requested indices for a
    inotA=Tuple(noncommoninds(inds(A),ireqA))
    if length(ireqT)!=length(ireqA) || inotA!=inotT
        @error("subtensor assign, incompatable indices\ndestination inds=$(inds(T)),\n source inds=$(inds(A)).")
    end
    isortT=ireqT...,inotT... #all indices sorted so user specified ones are first.
    p=getperm(inds(T), ntuple(n -> isortT[n], length(isortT))) # find p such that isort[p]==inds(T)
    rsortT=permute((ranges(irs)...,ranges(inotT)...),p) #sorted ranges for T
    #@show isortT rsortT
    ireqAp=permute_tagplev(ireqA,ireqT) #match based on tags & plev, NOT IDs since dims are different.
    # @show ireqAp inotT inds(T)
    #@show ireqAp p
    isortA=permute((ireqAp...,inotA...),p) #inotA is the same inotT, using inotA here for a less confusing read.
    #@show inds(A) isortA
    # @show typeof(isortA)
    Ap=ITensors.permute(A,isortA...;allow_alias=true)
    #@show inds(Ap) typeof(tensor(T)) typeof(tensor(A)) typeof(tensor(Ap)) A Ap
    tensor(T)[rsortT...]=tensor(Ap)
    #@show T
end


getindex(T::ITensor, irs::Vararg{IndexRange,N}) where {N} = get_subtensor_ND(T,irs...)
getindex(T::ITensor, irs::Vararg{irPairU,N}) where {N} = get_subtensor_ND(T,indranges(irs)...)
setindex!(T::ITensor, A::ITensor,irs::Vararg{IndexRange,N}) where {N} = set_subtensor_ND(T,A,irs...)
setindex!(T::ITensor, A::ITensor,irs::Vararg{irPairU,N}) where {N} = set_subtensor_ND(T,A,indranges(irs)...)


