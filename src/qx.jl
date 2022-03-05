using ITensorMPOCompression
using LinearAlgebra
using ITensors

"""
  `block_qx(W_::ITensor,ms::matrix_state)::Tuple{ITensor,ITensor,Index}`

Perform a block respecting QX decomposition of the operator valued matrix `W`. 
The appropriate decomposition, QR, RQ, QL, LQ is selected based on the `matrix_state` `ms`.
The new internal `Index` between Q and R/L is modified so that the tags are "Link,qx" instead
"Link,qr" etc. returned by the qr/rq/ql/lq routines.  Q and R are also gauge fixed so that 
the corner element of R is 1.0 and Q†Q=d x 𝕀 where d in the dimensionality of the local
Hilbert space.

# Arguments
- `W` Opertor valued matrix for decomposition.
- `ms` upper/lower state of `W` and the left/right state of the returned `Q` 

# Keywords
- `dir::orth_type = right` : choose `left` or `right` canonical form
- `epsrr::Foat64 = 1e-14` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[:,j]))<epsrr are considered zero and removed. 

# Returns a Tuple containing
- `Q` with orthonormal columns or rows depending on `ms.lr`, dimensions: (χ+1)x(χ\'+1)
- `R` or `L` depending on `ms.ul`, dimensions: (χ+2)x(χ\'+2)
- `iq` the new internal link index between `Q` and `R`/`L`.  tags="Link,qx"
"""
function block_qx(W_::ITensor,ms::matrix_state)::Tuple{ITensor,ITensor,Index}
  W=copy(W_)
  use_rr=true #use rank revealing QX to clear out zero pivots.
  d,n,r,c=parse_links(W)
  #
  #  decide some strings and variables based on lr.
  #
  (tln,cr)= ms.lr==left ? ("l=$n",c) : ("l=$(n-1)",r)
  
  ilw=filterinds(inds(W),tags=tln)[1] #get the link to the next site
  offset=V_offsets(ms)
  V=getV(W,offset) #extract the V block
  il=filterinds(inds(V),tags=tln)[1] #link to next site 
  iothers=noncommoninds(inds(V),il) #group all other indices for QX factorization

  if ms.ul==lower
    if ms.lr==left
      Q,RL=ql(V,iothers;positive=true,rank=use_rr) #block respecting QL decomposition
    else #right
      RL,Q=lq(V,iothers;positive=true,rank=use_rr) #block respecting LQ decomposition
    end
  else #upper
    if ms.lr==left
      Q,RL=ITensorMPOCompression.qr(V,iothers;positive=true,rank=use_rr) #block respecting QR decomposition
    else #right
      RL,Q=rq(V,iothers;positive=true,rank=use_rr) #block respecting RQ decomposition
    end
  end
  set_scale!(RL,Q,offset) #rescale so the L(n,n)==1.0
  @assert norm(V-RL*Q)<1e-12 #make sure decomp worked
  qx=String(tags(commonind(RL,Q))[2]) #should be "ql","lq","qr" os "rq"
  replacetags!(Q ,qx,"qx") #releive client code from the burden dealing ql,lq,qr,rq tags
  replacetags!(RL,qx,"qx")
  W=setV(W,Q,ms) #Q is the new V, stuff Q into W. THis can resize W
  # @show ms
  # @show "RL="
  # if ms.lr==left
  #   pprint(commonind(RL,Q),RL,il,1e-14)
  # else
  #   pprint(il,RL,commonind(RL,Q),1e-14)
  # end
  #@show inds(W) ilw inds(RL)
  RLplus,iqx=growRL(RL,ilw,offset) #Now make a full size version of RL
  #@show inds(RLplus) iqx
  # @show "RLplus="
  # if ms.lr==left
  #   #@show inds(RLplus) iqx ilw
  #   pprint(iqx,RLplus,ilw,1e-14)
  # else
  #   pprint(ilw,RLplus,iqx,1e-14)
  # end  
  ilw=filterinds(W,tags=tln)[1]
  replaceind!(W,ilw,iqx)  
  @assert hastags(W,"qx")
  return W,RLplus,iqx
end



function ql!(A::StridedMatrix{<:LAPACK.BlasFloat}, ::NoPivot; blocksize=36)
  tau=similar(A, min(size(A)...))
  x=LAPACK.geqlf!(A, tau)
  #save L from the lower portion of A, before orgql! mangles it!
  nr,nc=size(A)
  mn=min(nr,nc)
  L=similar(A,(mn,nc))
  for r in 1:mn
    for c in 1:r+nc-mn
      L[r,c]=A[r+nr-mn,c]
    end
    for c in r+1+nc-mn:nc
      L[r,c]=0.0
    end
  end
  # Now we need shift the orth vectors from the right side of Q over the left side, before
  if (mn<nc)
      for r in 1:nr
        for c in 1:mn
            A[r,c]=A[r,c+nc-mn];
        end
      end
  end
# A may now have extra columns from mn+1:nc, but they get chopped of in the outer functions
  LAPACK.orgql!(A,tau)
  return  A,L
end

function lq!(A::StridedMatrix{<:LAPACK.BlasFloat}, ::NoPivot; blocksize=36)
  tau=similar(A, min(size(A)...))
  x=LAPACK.gelqf!(A, tau)
  #save L from the lower portion of A, before orgql! mangles it!
  nr,nc=size(A)
  mn=min(nr,nc)
  L=similar(A,(nr,mn))
  for c in 1:mn
    for r in 1:c-1
      L[r,c]=0.0
    end
    for r in c:nr
      L[r,c]=A[r,c]
    end
  end
  
  LAPACK.orglq!(A,tau)
  # now A is Q but it may nave extra rows that need to be chopped off
  Q=A[1:mn,:] 
  return  L,Q
end

function rq!(A::StridedMatrix{<:LAPACK.BlasFloat}, ::NoPivot; blocksize=36)
  tau=similar(A, min(size(A)...))
  x=LAPACK.gerqf!(A, tau)
  #save R from the lower portion of A, before orgql! mangles it!
  nr,nc=size(A)
  mn=min(nr,nc)
  R=similar(A,(nr,mn))
  for c in 1:mn
    for r in 1:c+nr-mn
      R[r,c]=A[r,c+nc-mn]
    end
    for r in c+1+nr-mn:nr
      R[r,c]=0.0
    end
  end
  #
  # If nr>nc we need shift the orth vectors from the bottom of Q up to top before
  # unpacking the reflectors.
  #
  if mn<nr
    for c in 1:nc
      for r in 1:mn
        A[r,c]=A[r+nr-mn,c]
      end
    end
    A=A[1:mn,:] #whack the extra rows in A or orgrq! will complain
  end
  LAPACK.orgrq!(A,tau)
  return  R,A
end


ql!(A::AbstractMatrix) = ql!(A, NoPivot())
lq!(A::AbstractMatrix) = lq!(A, NoPivot())
rq!(A::AbstractMatrix) = rq!(A, NoPivot())


function ql(A::AbstractMatrix{T}, arg...; kwargs...) where T
    Base.require_one_based_indexing(A)
    AA = similar(A, LinearAlgebra._qreltype(T), size(A))
    copyto!(AA, A)
    return ql!(AA, arg...; kwargs...)
end

function lq(A::AbstractMatrix{T}, arg...; kwargs...) where T
  Base.require_one_based_indexing(A)
  AA = similar(A, LinearAlgebra._qreltype(T), size(A))
  copyto!(AA, A)
  return lq!(AA, arg...; kwargs...)
end

function rq(A::AbstractMatrix{T}, arg...; kwargs...) where T
  Base.require_one_based_indexing(A)
  AA = similar(A, LinearAlgebra._qreltype(T), size(A))
  copyto!(AA, A)
  return rq!(AA, arg...; kwargs...)
end

function ql_positive(M::AbstractMatrix)
  sparseQ, L = ql(M)
  Q = convert(Matrix, sparseQ)
  nr,nc = size(Q)
  dc=nc>nr ? nc-nr : 0 #diag is shifted over by dc if nc>nr
  for c in 1:nc
    if c<=nr && real(L[c, c+dc]) < 0.0
      L[c, 1:c+dc] *= -1
      Q[:,c] *= -1
    end
  end
  return (Q, L)
end

function lq_positive(M::AbstractMatrix)
  L, sparseQ = lq(M)
  Q = convert(Matrix, sparseQ)
  nr,nc = size(Q)
  for r in 1:nr
    if r<=nc && real(L[r, r]) < 0.0
      L[r:end, r] *= -1 
      Q[r,:] *= -1
    end
  end
  return (L, Q)
end

function rq_positive(M::AbstractMatrix)
  R, sparseQ = rq(M)
  Q = convert(Matrix, sparseQ)
  nr,nc = size(R)
  dr=nr>nc ? nr-nc : 0 #diag is shifted down by dr if nr>nc
  for r in 1:nr
    if r<=nc && real(R[r+dr, r]) < 0.0
      R[1:r+dr, r] *= -1 
      Q[r,:] *= -1
    end
  end
  return (R, Q)
end

function ql(T::NDTensors.DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
    positive = get(kwargs, :positive, false)
    # TODO: just call qr on T directly (make sure
    # that is fast)
    if positive
      QM, LM = ql_positive(matrix(T))
    else
      QM, LM = ql(matrix(T))
    end
    # Make the new indices to go onto Q and L
    q, l = inds(T)
    q = dim(q) < dim(l) ? sim(q) : sim(l)
    Qinds = IndsT((ind(T, 1), q))
    Linds = IndsT((q, ind(T, 2)))
    Q = NDTensors.tensor(NDTensors.Dense(vec(Matrix(QM))), Qinds)
    L = NDTensors.tensor(NDTensors.Dense(vec(LM)), Linds)
    return Q, L
  end

  function lq(T::NDTensors.DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
    positive = get(kwargs, :positive, false)
    # TODO: just call qr on T directly (make sure
    # that is fast)
    if positive
      LM, QM = lq_positive(matrix(T))
    else
      LM, QM = lq(matrix(T))
    end
    
    # Make the new indices to go onto Q and L
    l, q = inds(T)
    #@show inds(T)
    q = dim(q) < dim(l) ? sim(q) : sim(l)
    Qinds = IndsT((q,ind(T, 2)))
    Linds = IndsT((ind(T, 1),q))
    #@show Linds Qinds
    Q = NDTensors.tensor(NDTensors.Dense(vec(Matrix(QM))), Qinds)
    L = NDTensors.tensor(NDTensors.Dense(vec(LM)), Linds)
    return L, Q
  end

  function rq(T::NDTensors.DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
    positive = get(kwargs, :positive, false)
    # TODO: just call qr on T directly (make sure
    # that is fast)
    if positive
      RM, QM = rq_positive(matrix(T))
    else
      RM, QM = rq(matrix(T))
    end
    
    # Make the new indices to go onto Q and L
    #@show inds(T)
    l, q = inds(T)
    q = dim(q) < dim(l) ? sim(q) : sim(l)
    Qinds = IndsT((q,ind(T, 2)))
    Linds = IndsT((ind(T, 1),q))
    #@show Linds Qinds
    Q = NDTensors.tensor(NDTensors.Dense(vec(Matrix(QM))), Qinds)
    R = NDTensors.tensor(NDTensors.Dense(vec(RM)), Linds)
    return R, Q
  end

# ql decomposition of an order-n tensor according to 
# positions Lpos and Rpos 
function ql(
  T::NDTensors.DenseTensor{<:Number,N,IndsT}, 
  Lpos::NTuple{NL,Int}, 
  Rpos::NTuple{NR,Int}; 
  kwargs...
) where {N,IndsT,NL,NR}
  M = NDTensors.permute_reshape(T, Lpos, Rpos)
  QM, LM = ql(M; kwargs...)
  l = ind(LM, 1)
  # TODO: simplify this by permuting inds(T) by (Lpos,Rpos)
  # then grab Linds,Rinds
  Linds = NDTensors.similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Qinds = NDTensors.push(Linds, l)
  Q = reshape(QM, Qinds)
  Rinds = NDTensors.similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Rinds = NDTensors.pushfirst(Rinds, l)
  L = NDTensors.reshape(LM, Rinds)
  return Q, L
end

# lq decomposition of an order-n tensor according to 
# positions Lpos and Rpos 
function lq(
  T::NDTensors.DenseTensor{<:Number,N,IndsT}, 
  Lpos::NTuple{NL,Int}, 
  Rpos::NTuple{NR,Int}; 
  kwargs...
) where {N,IndsT,NL,NR}
  M = NDTensors.permute_reshape(T, Lpos, Rpos)
  LM, QM = lq(M; kwargs...)
  l = ind(LM, 2)
  # TODO: simplify this by permuting inds(T) by (Lpos,Rpos)
  # then grab Linds,Rinds
  Rinds = NDTensors.similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Qinds = NDTensors.pushfirst(Rinds, l)
  Q = reshape(QM, Qinds)
  Linds = NDTensors.similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Linds = NDTensors.push(Linds, l)
  L = NDTensors.reshape(LM, Linds)
  return L, Q
end

# lq decomposition of an order-n tensor according to 
# positions Lpos and Rpos 
function rq(
  T::NDTensors.DenseTensor{<:Number,N,IndsT}, 
  Lpos::NTuple{NL,Int}, 
  Rpos::NTuple{NR,Int}; 
  kwargs...
) where {N,IndsT,NL,NR}
  M = NDTensors.permute_reshape(T, Lpos, Rpos)
  RM, QM = rq(M; kwargs...)
  l = ind(RM, 2)
  # TODO: simplify this by permuting inds(T) by (Lpos,Rpos)
  # then grab Linds,Rinds
  Rinds = NDTensors.similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Qinds = NDTensors.pushfirst(Rinds, l)
  Q = reshape(QM, Qinds)
  Linds = NDTensors.similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Linds = NDTensors.push(Linds, l)
  R = NDTensors.reshape(RM, Linds)
  return R, Q
end
  

function ql(A::ITensor, Linds...; kwargs...)
    tags::TagSet = get(kwargs, :tags, "Link,ql")
    Lis = commoninds(A, ITensors.indices(Linds))
    Ris = uniqueinds(A, Lis)
    Lpos, Rpos = NDTensors.getperms(inds(A), Lis, Ris)
    QT, LT = ql(ITensors.tensor(A), Lpos, Rpos; kwargs...)
    Q, L = itensor(QT), itensor(LT)
    q::Index = ITensors.commonind(Q, L)
    settags!(Q, tags, q)
    settags!(L, tags, q)
    q = settags(q, tags)
    #
    #  Do row removal for rank revealing LQ
    #
    rr::Bool     = get(kwargs, :rank, false)
    eps::Float64 = get(kwargs, :eps , 1e-14)
    if rr L,Q=trim(L,Q,eps) end
    return Q, L, q
end

function lq(A::ITensor, Rinds...; kwargs...)
  tags::TagSet = get(kwargs, :tags, "Link,lq")
  Ris = commoninds(A, ITensors.indices(Rinds))
  Lis = uniqueinds(A, Ris)
  Lpos, Rpos = NDTensors.getperms(inds(A), Lis, Ris)
  LT, QT = lq(ITensors.tensor(A), Lpos, Rpos; kwargs...)
  Q, L = itensor(QT), itensor(LT)
  q::Index = ITensors.commonind(Q, L)
  settags!(Q, tags, q)
  settags!(L, tags, q)
  q = settags(q, tags)
  #
  #  Do row removal for rank revealing LQ
  #
  rr::Bool     = get(kwargs, :rank, false)
  eps::Float64 = get(kwargs, :eps , 1e-14)
  if rr L,Q=trim(L,Q,eps) end
  return L, Q, q
end

function rq(A::ITensor, Rinds...; kwargs...)
  tags::TagSet = get(kwargs, :tags, "Link,rq")
  Ris = commoninds(A, ITensors.indices(Rinds))
  Lis = uniqueinds(A, Ris)
  Lpos, Rpos = NDTensors.getperms(inds(A), Lis, Ris)
  RT, QT = rq(ITensors.tensor(A), Lpos, Rpos; kwargs...)
  Q, R = itensor(QT), itensor(RT)
  q::Index = ITensors.commonind(Q, R)
  settags!(Q, tags, q)
  settags!(R, tags, q)
  q = settags(q, tags)
  #
  #  Do row removal for rank revealing RQ
  #
  rr::Bool     = get(kwargs, :rank, false)
  eps::Float64 = get(kwargs, :eps , 1e-14)
  if rr R,Q=trim(R,Q,eps) end
  return R, Q, q
end

function qr(A::ITensor, Rinds...; kwargs...)
  Q,R,q=ITensors.qr(A,Rinds...;kwargs...)
  #
  #  Do row removal for rank revealing RQ
  #
  rr::Bool     = get(kwargs, :rank, false)
  eps::Float64 = get(kwargs, :eps , 1e-14)
  if rr R,Q=trim(R,Q,eps) end
  return Q, R, q
end

function trim(R::ITensor,Q::ITensor,eps::Float64)
  iq=commonind(R,Q)
  zeros=find_zero_rows(R,iq,eps)
  nq=dim(iq)-sum(zeros)
  iRo=noncommoninds(R,iq)
  iQo=noncommoninds(Q,iq)
  iqn=Index(nq,tags(iq))
  Rn=ITensor(0.0,iqn,iRo)
  Qn=ITensor(0.0,iQo,iqn)
  ivqn=1
  for ivq in eachindval(iq)
    if zeros[ivq.second]==false
      for ivRo in eachindval(iRo...)
        Rn[iqn=>ivqn,ivRo]=R[ivq,ivRo]
      end #for ivRo
      for ivQo in eachindval(iQo...)
        Qn[iqn=>ivqn,ivQo...]=Q[ivq,ivQo...]
      end #for ivQo
      ivqn+=1
    end #if zero
  end #for ivq
  return Rn,Qn
end

function find_zero_rows(R::ITensor,iq::Index,eps::Float64)::Array{Bool}
  zeros=falses(dim(iq))
  others=noncommoninds(R,iq)
  for iqv in eachindval(iq)
    s=0.0
    for io in eachindval(others...)
      s+=abs(R[iqv,io])
    end
    zeros[iqv.second]= (s<=eps)
  end
  return zeros
end
#=
is=Index(2,"Site,n=1")
ir=Index(4,"Link,l=0")
ic=Index(4,"Link,l=1")
iq=Index(4,"Link,qx")
#AQ=[i*1.0 for i in 1:dim(ic)*dim(iq)*dim(is)*dim(is)]
#Q=ITensor(eltype(AQ),AQ,ic,iq,is,is')
AQ=[i*1.0 for i in 1:dim(ic)*dim(iq)]
Q=ITensor(eltype(AQ),AQ,ic,iq)

A=[1:4;;5:8;;9:12;;13:16]
A[2,:].=0.0
R=ITensor(eltype(A),A,iq,ir)
z=find_zero_rows(R,iq,0.0)
Rn,Qn=trim(R,Q,1e-14)
@show Rn
@show Qn
=#
