using ITensors 
using ITensors.NDTensors
using ITensorMPOCompression
using Printf
using Test
using Revise

import ITensors: dim, dims, DenseTensor, eachindval, eachval, getindex, setindex!
import NDTensors: getperm, permute, BlockDim
import ITensorMPOCompression: redim

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

struct IndexRange
    index::Index
    range::UnitRange{Int64}
    function IndexRange(i::Index,r::UnitRange)
        return new(i,r)
    end
end

start(ir::IndexRange)=range(ir).start
range(ir::IndexRange)=ir.range
range(i::Index)=1:dim(i)
ranges(irs::Tuple) = ntuple(i -> range(irs[i]), Val(length(irs)))
indices(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ir.index ,irs)

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


function permute(indsT,irs::IndexRange...)
    ispec=indices(irs) #indices caller specified ranges for
    inot=Tuple(noncommoninds(indsT,ispec)) #indices not specified by caller
    isort=ispec...,inot... #all indices sorted so user specified ones are first.
    isort_sub=redim(irs)...,inot... #all indices for subtensor
    p=getperm(indsT, ntuple(n -> isort[n], length(isort)))
    return permute(isort_sub,p),permute((ranges(irs)...,ranges(inot)...),p)
end
#
#  Use NDTensors T[3:4,1:3,1:6...] syntax to extract the subtensor.
#
function get_subtensor_ND(T::ITensor,irs::IndexRange...)
    isub,rsub=permute(inds(T),irs...) #get indices and ranges for the subtensor
    return get_subtensor_wrapper(tensor(T),isub,rsub...) #virtual dispatch based on Dense or BlockSparse
end

function set_subtensor_ND(T::ITensor, A::ITensor,irs::IndexRange...)
    _,rsub=permute(inds(T),irs...) #get ranges for the subtensor
    tensor(T)[rsub...]=tensor(A)
end

getindex(T::ITensor, irs::Vararg{IndexRange,N}) where {N} = get_subtensor_ND(T,irs...)
setindex!(T::ITensor, A::ITensor,irs::Vararg{IndexRange,N}) where {N} = set_subtensor_ND(T,A,irs...)



#-----------------------------------------------------------------------
#
#  Test helper Use loops over ITensor indices to perform and subtensor extraction
#
function get_subtensor_I(T::ITensor,irs::IndexRange...)
    is=indices(irs)
    iothers=Tuple(noncommoninds(T,is))
    iso=is...,iothers...
    p=getperm(inds(T), ntuple(n -> iso[n], length(iso)))
    is_sub=redim(irs) #get re-dimensied Indices
    iso_subp=permute((is_sub...,iothers...),p)
    
    T_sub=ITensor(eltype(T),iso_subp)
    for (i1,i) in zip(eachindval(is_sub),eachindval(irs))
        assign!(T_sub,slice(T,i...),i1...)
    end
    return T_sub
end

@testset "subtensor with dense storage" begin
    # setup a random 3 index ITensor
    i=Index(5,"i")
    j=Index(6,"j")
    k=Index(2,"k")
    T=randomITensor(k,j,i)
    #
    # Choose ranges for two of the indices.
    #
    ir=IndexRange(i,2:4)
    jr=IndexRange(j,3:5)
    #
    #  Extract subvector using two different methods. Ts1 and Ts2 will have different i,j indices
    #  so we can't subtract them at the ITensor level.
    #
    Ts1=T[ir,jr] #production version NDTensors. 
    Ts2=get_subtensor_I(T,ir,jr) #development version does everying at the ITensors level
    @test  norm(tensor(Ts2)-tensor(Ts1))==0.0 #should be the same index permuations are correct.
    #
    #  Now swap the order of the ranges, should have no effect
    #
    Ts3=T[jr,ir]
    Ts4=get_subtensor_I(T,jr,ir)
    @test  norm(tensor(Ts3)-tensor(Ts4))==0.0
    @test  norm(tensor(Ts1)-tensor(Ts3))==0.0
    #
    #  Test assignment back into T
    #
    Ts1.*=2.121 #change the data
    T[ir,jr]=Ts1 #bulk assignment
    is1=inds(Ts1,tags="i")[1] #find the new inds of Ts1 
    js1=inds(Ts1,tags="j")[1] #k should be the same
    for iv in eachindval(is1)
        for jv in eachindval(js1)
            for kv in eachindval(k)
                @test Ts1[iv,jv,kv]==T[i=>(val(iv)+start(ir)-1),j=>(val(jv)+start(jr)-1),k=>val(kv)]
            end
        end
    end

end

@testset "subtensor with block sparse storage" begin
    N,NNN=10,3
    sites = siteinds("SpinHalf", N;conserve_qns=true)
    H=make_transIsing_MPO(sites,NNN)
    W=H[5] 

    il=inds(W,tags="Link")
    Dw=dim(il[1])
    #
    #  extract sub tensors
    #
    i1,i2=IndexRange(il[1],1:Dw-1),IndexRange(il[2],1:Dw-1)
    V=W[i1,i2]
    V1=get_subtensor_I(W,i1,i2)
    @test  norm(tensor(dense(V))-tensor(dense(V1)))==0.0

    i1,i2=IndexRange(il[1],2:Dw),IndexRange(il[2],2:Dw)
    V=W[i1,i2]
    V1=get_subtensor_I(W,i1,i2)
    @test  norm(tensor(dense(V))-tensor(dense(V1)))==0.0
    #
    #  Assign sub tensor
    #
    V.*=2.5345
    W[i1,i2]=V

    iv1,iv2=inds(V,tags="Link")
    for iiv1 in eachindval(iv1)
        for iiv2 in eachindval(iv2)
            opV=slice(V,iiv1,iiv2)
            opW=slice(W,il[1]=>(val(iiv1)+start(i1)-1),il[2]=>(val(iiv2)+start(i2)-1))
            @test opV==opW
        end
    end

    #make another version of W, but with indexes re-ordered
    iW=inds(W)
    iW1=(iW[3],iW[1],iW[4],iW[2]) #interleave blocked and un-blocked indices.
    W1=ITensor(QN("Sz",0),iW1)
    V=W1[i1,i2]
    V1=get_subtensor_I(W1,i1,i2)
    @test  norm(tensor(dense(V))-tensor(dense(V1)))==0.0
 end
