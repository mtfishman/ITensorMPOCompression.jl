using ITensors
using ITensors.NDTensors
using ITensorMPOCompression
using Printf

import ITensors: dims
import ITensorMPOCompression: redim
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

struct IndexRange
    index::Index
    range::UnitRange{Int64}
    function IndexRange(i::Index,r::UnitRange)
        return new(i,r)
    end
end

range(ir::IndexRange)=ir.range
range(i::Index)=1:ITensors.dim(i)
dim(ir::IndexRange)=dim(range(ir))
dim(r::UnitRange{Int64})=r.stop-r.start+1
dims(irs::Tuple{Vararg{IndexRange}})=map((ir)->dim(ir),irs)
eachval(ir::IndexRange) = range(ir)

ranges(irs::Tuple) = ntuple(i -> range(irs[i]), Val(length(irs)))
indices(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ir.index ,irs)
ITensorMPOCompression.redim(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ITensorMPOCompression.redim(ir.index,dim(ir)) ,irs)


eachval(irs::IndexRange...) = eachval(irs)
eachval(irs::Tuple{Vararg{IndexRange}}) = CartesianIndices(ranges(irs))

eachindval(irs::IndexRange...) = eachindval(irs)
eachindval(irs::Tuple{Vararg{IndexRange}}) = (indices(irs).=> Tuple(ns) for ns in eachval(irs))

#Itensors uses this signature:
#function _getindex(T::Tensor, ivs::Vararg{<:Any,N}) where {N}

#
#  Use loops over ITensor indices to perform and subtensor extraction
#
function subtensor_I(T::ITensor,irs::IndexRange...)
    is=indices(irs)
    iothers=Tuple(noncommoninds(T,is))
    iso=is...,iothers...
    N=length(iso)
    p=NDTensors.getperm(inds(T), ntuple(n -> iso[n], Val(N)))
    is_sub=redim(irs) #get re-dimensied Indices
    iso_sub=is_sub...,iothers...
    iso_subp=NDTensors.permute(iso_sub,p)
    T_sub=ITensor(eltype(T),iso_subp)
    for (i1,i) in zip(ITensors.eachindval(is_sub),eachindval(irs))
        for io in ITensors.eachindval(iothers)
            e=T[i...,io...]
            T_sub[i1...,io...]=e
        end
    end
    return T_sub
end

#
#  Use NDTensors T[3:4,1:3,1:6...] syntax to extract the subtensor.
#
function subtensor_ND(T::ITensor,irs::IndexRange...)
    is=indices(irs) #indices caller specified ranges for
    iothers=Tuple(noncommoninds(T,is)) #indices not specified by caller
    iso=is...,iothers... #all indices
    rs=ranges(irs)...,ranges(iothers)... #all ranges
    is_sub=redim(irs) #get re-dimensied Indices
    iso_sub=is_sub...,iothers... #all indices for subtensor
    # Now permute all subtensor indices and ranges
    p=NDTensors.getperm(inds(T), ntuple(n -> iso[n], length(iso)))
    iso_subp=NDTensors.permute(iso_sub,p)
    rsp=NDTensors.permute(rs,p)
#--- begin pure NDTensors level code
    Tt=tensor(T) #extract NDTensor from T
    Tt_sub=Tt[rsp...] #use ranges to exctract the requested ND subtensor.
#--- end pure NDTensors level code

    return ITensor(Tt_sub,iso_subp) #Rebuild I subtensor with full indices.
end

ITensors.getindex(T::ITensor, irs::Vararg{IndexRange,N}) where {N} = subtensor_ND(T,irs...)

ii=Index(5,"i")
ij=Index(6,"j")
ik=Index(2,"k")
T=randomITensor(ik,ij,ii)

iri=IndexRange(ii,2:4)
irj=IndexRange(ij,3:5)

irij=iri,irj
Ts1=T[irij...]
Ts2=subtensor_I(T,irij...)
@show  norm(tensor(Ts2)-tensor(Ts1))==0.0

