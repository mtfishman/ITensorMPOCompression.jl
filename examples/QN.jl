using ITensors
using ITensorMPOCompression
import ITensors.BlockSparseTensor,ITensors.DenseTensor,ITensors.tensor
import ITensors.blockview

function trim(
  L::NDTensors.BlockSparseTensor{Elt,N,IndsT},
  Q::NDTensors.BlockSparseTensor{Elt,N,IndsT},
  eps::Float64)  where {Elt,N,IndsT}
  nb=nnzblocks(L) # number of nonzero blocks in R
  @assert nb==nnzblocks(Q) # they should have the same blocks
  # Qs = Vector{DenseTensor{Elt,NL+1,NTuple{NL+1, Int64},<:NDTensors.Dense}}(undef, nb)
  # Ls = Vector{DenseTensor{Elt,NR+1,NTuple{NR+1, Int64},<:NDTensors.Dense}}(undef, nb)
  for (n, b) in enumerate(eachnzblock(L))
    blockL = blockview(L, b)
    bq=nzblocks(Q)[n]
    blockQ = blockview(Q,bq)
    LT,QT=trim(blockL,blockQ,eps)
    @show LT QT
  end
end

function ql1(A::ITensor, Linds...; kwargs...)
    tags::TagSet = get(kwargs, :tags, "Link,ql")
    Lis = commoninds(A, ITensors.indices(Linds)) #indices narrows type and handles empty case?
    Ris = uniqueinds(A, Lis)
    Lpos, Rpos = NDTensors.getperms(inds(A), Lis, Ris) #gets hard index ordering used at the tensor level
    QT, LT = ql1(ITensors.tensor(A), Lpos, Rpos; kwargs...) #virtual dispatch to handle dense and block sparse cases.
    Q, L = itensor(QT), itensor(LT)
    q::Index = ITensors.commonind(Q, L,tags="Link") #L,Q might have common site indices if blocking is used.
    settags!(Q, tags, q)
    settags!(L, tags, q)
    q = settags(q, tags)
    #
    #  Do row removal for rank revealing LQ
    #
    epsrr::Float64 = get(kwargs, :epsrr , 1e-14)
    if epsrr>0.0 L,Q=trim1(L,Q,epsrr) end
    return Q, L, q
end


# ql decomposition of an order-n block sparse tensor according to 
# positions Lpos and Rpos 
function ql1(
    T::NDTensors.BlockSparseTensor{Elt,N,IndsT}, 
    Lpos::NTuple{NL,Int}, 
    Rpos::NTuple{NR,Int}; 
    kwargs...
  ) where {Elt,N,IndsT,NL,NR}
  
    nb=nnzblocks(T) # number of nonzero blocks in T
    Qs = Vector{DenseTensor{Elt,NL+1,NTuple{NL+1, Int64},<:NDTensors.Dense}}(undef, nb)
    Ls = Vector{DenseTensor{Elt,NR+1,NTuple{NR+1, Int64},<:NDTensors.Dense}}(undef, nb)
    
    #@show Lpos Rpos inds(T) dim(T,1) dim(T,2)
    #
    @show "in ql block sparse" Lpos Rpos
    for (n, b) in enumerate(eachnzblock(T))
      blockT = blockview(T, b) #this is a DenseTensor so we call the dense ql for each block.
      Qs[n],Ls[n] = ql(blockT,Lpos,Rpos;kwargs...) #All reshaping is taken care of in this call.
#      @show blockT Qs[n] Ls[n]
    end
    #
    #  Now we need to build an ITensor index that goes between Q and L
    #
    qdim=dim(ind(Ls[1],1)) #first get the dimension
    @assert dim(inds(Qs[1])[end])==qdim #sanity check
    qind = sim(ind(T, 1))
    if dim(qind) != qdim
      resize!(qind, qdim)
    end
    #@show qind
    # indsQ = setindex(inds(T), dag(qind), 2)  #return inds(T) but with the 2nd index replaced with dag(qind)
    # indsL = setindex(inds(T), dag(qind), 1)  #return inds(T) but with the 1st index replaced with dag(qind)
    iq1=ind(T,Lpos[1])
    iq2=ind(T,Lpos[2])
    iq3=ind(T,Lpos[3])
    il1=ind(T,Rpos[1])
    indsQ = (iq1,iq2,iq3,dag(qind))
    indsL = (qind,il1,iq2,iq3)
    #@show indsQ indsL
    nzblocksQ = Vector{Block{4}}(undef, nb)
    nzblocksL = Vector{Block{4}}(undef, nb)
    for n in 1:nb
       blockT=nzblocks(T)[n]
       nzblocksQ[n]=(blockT[Lpos[1]],blockT[Lpos[2]],blockT[Lpos[3]],UInt(1))
       nzblocksL[n]=(UInt(1)        ,blockT[Rpos[1]],blockT[Lpos[2]],blockT[Lpos[3]])
    end
    Q = BlockSparseTensor(Elt, undef, nzblocksQ, indsQ)
    L = BlockSparseTensor(Elt, undef, nzblocksL, indsL)
    for (n, b) in enumerate(eachnzblock(Q))
      blockview(Q, b).=Qs[n]
    end
    for (n, b) in enumerate(eachnzblock(L))
      blockview(L, b).=Ls[n]
    end
    return Q,L
end

function find_zero_rows1(R::ITensor,iq::Index,eps::Float64)::Array{Bool}
  others=noncommoninds(R,iq)
  iq1,others1 = NDTensors.getperms(inds(R), [iq],others) #gets hard index ordering used at the tensor level
  return find_zero_rows1(tensor(R),iq1[1],eps)
end

function find_zero_rows1(R::DenseTensor{ElT,N},rind::Int64,eps::Float64)::Array{Bool} where {ElT,N}
  r=ind(R,rind)
  zeros=falses(dim(r))
  for ir in 1:r #choose a row
    s=0.0
    #here we need to sweep over all other indices.  Currently this is a terrible way to do it!.
    for I in eachindex(R) #sweep over all indices
      if I[rind]==ir #filter out row ir ... yuck!!!!
        s+=abs(R[I])
      end
    end
    zeros[ir]= (s<=eps)
  end
  return zeros
end

function find_zero_rows1(R::BlockSparseTensor{ElT,N},rind::Int64,eps::Float64)::Array{Bool} where {ElT,N}
  r=ind(R,rind)
  zeros=trues(dim(r))
  for (n, b) in enumerate(eachnzblock(R))
    zeros.=zeros .&& find_zero_rows1(blockview(R, b),rind,eps)
  end
  return zeros
end

# function remove_rows(source::ITensor,dest::ITensor,rows::Array{Bool})::ITensor

# end

function trim1(R::ITensor,Q::ITensor,eps::Float64)
  iq=commonind(R,Q)
  zeros=find_zero_rows1(R,iq,eps)
  @show zeros
  if sum(zeros)==0
    return R,Q
  end
  #@printf "Rank Reveal removing %4i rows with epsrr=%.1e\n" sum(zeros) eps
  nq=dim(iq)-sum(zeros)
  iRo=noncommoninds(R,iq)
  iQo=noncommoninds(Q,iq)
  iRl=noncommoninds(R,iq,tags="Link")
  iQl=noncommoninds(Q,iq,tags="Link")
#  iqn=Index(nq,tags(iq))
  iqn=redim(iq,nq)
  T=eltype(R)
  @show nblocks(Q) nblocks(R)
  Rn=ITensor(T(0.0),iqn,iRo)
  Qn=ITensor(T(0.0),iQo,dag(iqn))
#  @show inds(Rn)
  @show nblocks(Qn) nblocks(Rn)

  ivqn=1
  for ivq in eachindval(iq)
    if zeros[ivq.second]==false
      for ivRl in eachindval(iRl...)
        op=slice(R,ivq,ivRl)
        #@show op
        assign!(Rn,op,iqn=>ivqn,ivRl)
        #@show nblocks(Rn)
      end #for ivRo
      for ivQl in eachindval(iQl...)
        op=slice(Q,dag(ivq),ivQl)
        assign!(Qn,op,iqn=>ivqn,ivQl)
      end #for ivQo
      ivqn+=1
    end #if zero
  end #for ivq
  return Rn,Qn
end


N = 10
NNN = 2
hx=0.0 #can't make and sx op with QNs in play
ms=matrix_state(upper,left)
sites = siteinds("S=1/2",N;conserve_qns=true)
H=make_transIsing_MPO(sites,NNN,hx,ms.ul)
V=getV(H[2],V_offsets(ms))
@show nblocks(V)
Rinds=filterinds(V,tags="l=2")
Linds=noncommoninds(inds(V),Rinds)
# CL = combiner(Linds...)
# CR = combiner(Rinds...)
# AC = V * CR * CL
# cL = combinedind(CL)
# cR = combinedind(CR)
# if inds(AC) != (cL, cR)
#     AC = permute(AC, cL, cR)
# end
# U,s,V=ITensors.NDTensors.svd(tensor(AC))   
# U

Lpos, Rpos = NDTensors.getperms(inds(V), Linds, Rinds) #((1, 3, 4), (2,))
nblocks(V) #(1, 1, 2, 2)
qind = sim(ind(V, 2)) #(dim=4|id=825|"Link,l=2") <Out> 1: QN("Sz",0) => 4
#setindex(inds(V), qind, 2) 
#Vt=tensor(V)
#nzblocks(V)
#pprint(V)
#find_zero_rows1(V,Linds[1],1e-14)

Q,R=qr(V,Linds;positive=true) #,orth=left,epsrr=1e-10)
@show inds(Q)
@show inds(R)
V1=Q*R
@show V-V1
# @show nblocks(Q) nblocks(RL)
# @show V1
#pprint(V1)
nothing