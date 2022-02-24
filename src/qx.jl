using LinearAlgebra
using ITensors

function block_qx!(W::ITensor,lr::orth_type)::ITensor
  d,n,r,c=parse_links(W)
  if lr==left
      V=getV(W,1,1) #extract the V block
      il=filterinds(inds(V),tags="l=$n")[1] #link to next site to the right
  elseif lr==right
      V=getV(W,0,0) #extract the V block
      il=filterinds(inds(V),tags="l=$(n-1)")[1] #link to next site to left
  else
      assert(false)
  end

  iothers=noncommoninds(inds(V),il)
  if lr==left
      Q,L=ql(V,iothers;positive=true) #block respecting QL decomposition
      set_scale!(L,Q,1,1) #rescale so the L(n,n)==1.0
      @assert norm(V-Q*L)<1e-12 
      setV!(W,Q,1,1) #Q is the new V, stuff Q into W
      iWl=filterinds(inds(W),tags="l=$n")[1]
      Lplus,iqx=growRL(L,iWl,1,1) #Now make a full size version of L
      replaceind!(W,c,iqx)
      replacetags!(W    ,"ql","qx")
      replacetags!(Lplus,"ql","qx")
  elseif lr==right
      @assert detect_upper_lower(V,1e-14)==lower
      L,Q=lq(V,iothers;positive=true) #block respecting QL decomposition
      set_scale!(L,Q,0,0) #rescale so the L(n,n)==1.0
      @assert norm(V-L*Q)<1e-12 
      setV!(W,Q,0,0) #Q is the new V, stuff Q into W
      @assert detect_upper_lower(W,1e-14)==lower
      iWl=filterinds(inds(W),tags="l=$(n-1)")[1]
      Lplus,iqx=growRL(L,iWl,0,0) #Now make a full size version of L
      replaceind!(W,r,iqx)
      replacetags!(W    ,"lq","qx")
      replacetags!(Lplus,"lq","qx")
  else
      assert(false)
  end
  return Lplus   
end

function block_qx(W_::ITensor,lr::orth_type)::Tuple{ITensor,ITensor,Index}
  W=copy(W_)
  d,n,r,c=parse_links(W)
  #
  #  decide some strings and variables based on lr.
  #
  (tln,o1,o2,lql,cr)= lr==left ? ("l=$n",1,1,"ql",c) : ("l=$(n-1)",0,0,"lq",r)
 
  V=getV(W,o1,o2) #extract the V block
  il=filterinds(inds(V),tags=tln)[1] #link to next site 
  iothers=noncommoninds(inds(V),il) #group all other indices for QX factorization

  if lr==left
      Q,L=ql(V,iothers;positive=true) #block respecting QL decomposition
  elseif lr==right
      L,Q=lq(V,iothers;positive=true) #block respecting LQ decomposition
  else
      assert(false)
  end
  set_scale!(L,Q,o1,o2) #rescale so the L(n,n)==1.0
  @assert norm(V-L*Q)<1e-12 #make decomp worked
  replacetags!(Q,lql,"qx") #releive client code from the burden dealing ql,lq,qr,rq tags
  replacetags!(L,lql,"qx")
  setV!(W,Q,o1,o2) #Q is the new V, stuff Q into W
  #@assert detect_upper_lower(W,1e-14)==lower
  iln=filterinds(inds(W),tags=tln)[1]
  Lplus,iqx=growRL(L,iln,o1,o2) #Now make a full size version of L
  replaceind!(W,cr,iqx)  
  return W,Lplus,iqx
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


ql!(A::AbstractMatrix) = ql!(A, NoPivot())
lq!(A::AbstractMatrix) = lq!(A, NoPivot())


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
    q = dim(q) < dim(l) ? sim(q) : sim(l)
    Qinds = IndsT((q,ind(T, 2)))
    Linds = IndsT((ind(T, 1),q))
    Q = NDTensors.tensor(NDTensors.Dense(vec(Matrix(QM))), Qinds)
    L = NDTensors.tensor(NDTensors.Dense(vec(LM)), Linds)
    return L, Q
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
  q = ind(QM, 2)
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
  q = ind(QM, 1)
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
  return L, Q, q
end

# function qr_test()
#   A1 = [3.0 -6.0; 4.0 -8.0; 0.0 1.0]
#   M,N=size(A1)
#   ir=Index(M,"row")
#   ic=Index(N,"col")
#   A=ITensor(ir,ic,)
#   for irc=eachindval(inds(A))
#       A[irc...]=A1[irc[1].second,irc[2].second]
#   end
#   Q,R=qr(A,ir)
#   A2=Q*R
#   norm(A-A2)<1e-15
# end

function ql_test()
  #A1 = [2.0 0.5 1.0; -1.0 0.0 2.0]
  #A1 = [2.0 0.5; 1.0 1.0; 0.0 -2.0]
  A1 = [1.0 0.0; 2.0 1.0]
  M,N=size(A1)
  ir=Index(M,"row")
  ic=Index(N,"col")
  A=ITensor(ir,ic) 
  for irc=eachindval(inds(A))
      A[irc...]=A1[irc[1].second,irc[2].second]
  end
  Q,L=ql(A,ir;positive=true)
  @show Q,L
  norm(A-Q*L)
end

function lq_test()
  #A1 = [2.0 0.5 1.0; -1.0 0.0 2.0]
  A1 = [2.0 0.5; 1.0 1.0; 0.0 -2.0]
  #A1 = [1.0 0.0; 2.0 1.0]
  M,N=size(A1)
  ir=Index(M,"row")
  ic=Index(N,"col")
  A=ITensor(ir,ic)
  for irc=eachindval(inds(A))
      A[irc...]=A1[irc[1].second,irc[2].second]
  end
  L,Q=lq(A,ic;positive=true)
  @show L,Q
  norm(A-L*Q)
end

#println("--------------------start--------------------")
#@show lq_test()
