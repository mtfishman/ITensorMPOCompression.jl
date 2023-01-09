using ITensors
using ITensors.NDTensors
using ITensorMPOCompression
using Printf
using Test

import ITensors: dims, DenseTensor
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
ranges(irs::Tuple) = ntuple(i -> range(irs[i]), Val(length(irs)))
indices(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ir.index ,irs)

dim(ir::IndexRange)=dim(range(ir))
dim(r::UnitRange{Int64})=r.stop-r.start+1
dims(irs::Tuple{Vararg{IndexRange}})=map((ir)->dim(ir),irs)
ITensorMPOCompression.redim(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ITensorMPOCompression.redim(ir.index,dim(ir)) ,irs)


eachval(ir::IndexRange) = range(ir)
eachval(irs::IndexRange...) = eachval(irs)
eachval(irs::Tuple{Vararg{IndexRange}}) = CartesianIndices(ranges(irs))

eachindval(irs::IndexRange...) = eachindval(irs)
eachindval(irs::Tuple{Vararg{IndexRange}}) = (indices(irs).=> Tuple(ns) for ns in eachval(irs))


#
#  NDTensor level code which distinguishes between Dense and BlockSparse storage
#
function subtensor(T::DenseTensor{ElT,N,IndsT,StoreT},rs::UnitRange{Int64}...) where {ElT,N,IndsT,StoreT<:Dense}
    return T[rs...]
end

function fix_ranges(ds::NTuple{N, Int64},c::CartesianIndex{N},rs::UnitRange{Int64}...) where {N}
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

function subtensor(T::BlockSparseTensor{ElT,N},rs::UnitRange{Int64}...) where {ElT,N}
    nnzblocksT = nnzblocks(T)
    nzblocksT = nzblocks(T)
    Ds = Vector{DenseTensor{ElT,N}}(undef, nnzblocksT)
    for (jj, b) in enumerate(eachnzblock(T))
        blockT = blockview(T, b)
        rs1=fix_ranges(dims(blockT),CartesianIndex(b),rs...)
        Ds[jj]=subtensor(blockT,rs1...)
    end
    Tds=dims(T)
    Dds=dims(Ds[1])
    T_sub_inds=Vector{Index}(undef,N)
    for i in 1:N
        if Dds[i]==1 && Tds[i]>1
            T_sub_inds[i]=inds(T)[i]
        else
            T_sub_inds[i]=redim(inds(T)[i],Dds[i])
        end
    end
    T_sub = BlockSparseTensor(ElT, undef, nzblocksT, Tuple(T_sub_inds))
    for ib in eachindex(Ds)
        blockT_sub = nzblocks(T_sub)[ib]
        blockview(T_sub, blockT_sub) .= Ds[ib]
    end
    return T_sub
end
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
    #iso_sub=is_sub...,iothers... #all indices for subtensor
    # Now permute all subtensor indices and ranges
    p=NDTensors.getperm(inds(T), ntuple(n -> iso[n], length(iso)))
    #iso_subp=NDTensors.permute(iso_sub,p)
    rsp=NDTensors.permute(rs,p)
    Tt_sub=subtensor(tensor(T),rsp...) #virtual dispatch based on Dense or BlockSparse
    return ITensor(Tt_sub)
end

ITensors.getindex(T::ITensor, irs::Vararg{IndexRange,N}) where {N} = subtensor_ND(T,irs...)


@testset "subtensor with dense storage" begin
    # setup a random 3 indes ITensor
    ii=Index(5,"i")
    ij=Index(6,"j")
    ik=Index(2,"k")
    T=randomITensor(ik,ij,ii)
    #
    # Choose ranges for two of the indices.
    #
    iri=IndexRange(ii,2:4)
    irj=IndexRange(ij,3:5)
    #
    #  Extract subvector using two different methods. Ts1 and Ts2 will have different i,j indices
    #  so we can't subtract them at the ITensor level.
    #
    Ts1=T[iri,irj] #production version the uses NDTensors. 
    Ts2=subtensor_I(T,iri,irj) #development version does everying at the ITensors level
    @test  norm(tensor(Ts2)-tensor(Ts1))==0.0 #should be the same index permuations are correct.
    #
    #  Now swap the order of the ranges, should have no effect
    #
    Ts3=T[irj,iri]
    Ts4=subtensor_I(T,irj,iri)
    @test  norm(tensor(Ts3)-tensor(Ts4))==0.0
    @test  norm(tensor(Ts1)-tensor(Ts3))==0.0
end

@testset "subtensor with block sparse storage" begin
    N,NNN=10,3
    sites = siteinds("SpinHalf", N;conserve_qns=true)
    H=make_transIsing_MPO(sites,NNN)
    W=H[5] 
    pprint(W)
    il=inds(W,tags="Link")
    Dw=ITensors.dim(il[1])
    i1,i2=IndexRange(il[1],2:Dw),IndexRange(il[2],2:Dw)
    V=W[i1,i2]
    pprint(V)
end
