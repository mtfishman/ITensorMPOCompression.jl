using LinearAlgebra
@doc """
  `block_qx(W::ITensor,ul::reg_form)::Tuple{ITensor,ITensor,Index}`

Perform a block respecting QX decomposition of the operator valued matrix `W`. 
The appropriate decomposition, QR, RQ, QL, LQ is selected based on the `reg_form` `ul` 
and the `orth` keyword argument.
The new internal `Index` between Q and R/L is modified so that the tags are "Link,qx" instead
"Link,qr" etc. returned by the qr/rq/ql/lq routines.  Q and R/L are also gauge fixed so that 
the corner element of R/L is 1.0 and Q⁺Q=d𝕀 where d is the dimensionality of the local
Hilbert space.

# Arguments
- `W` Operator valued matrix for decomposition.
- `ul` upper/lower regular form of `W`. We can auto detect here, but is more efficient if this is done by the higher level calling routines.

# Keywords
- `orth::orth_type = right` : choose `left` or `right` orthogonal (canonical) form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed.  rr_cutoff==-1.0 indicates no rank reduction.

# Returns a Tuple containing
- `Q` with orthonormal columns or rows depending on orth=left/right, dimensions: (χ+1)x(χq+1)
- `R` or `L` depending on `ul` with dimensions: (χq+2)x(χ\'+2)
- `iq` the new internal link index between `Q` and `R`/`L` with dimensions χq+2 and tags="Link,qx"

# Example
```julia
julia>using ITensors
julia>using ITensorMPOCompression
julia>N=5; #5 sites
julia>NNN=2; #Include 2nd nearest neighbour interactions
julia>sites = siteinds("S=1/2",N);
#
#  Make a Hamiltonian directly, i.e. not using autoMPO
#
julia>H=make_transIsing_MPO(sites,NNN);
#
#  Use pprint to see the structure for site #2. I = unit operator and S = any 
#  other operator
#
julia>pprint(H[2]) #H[1] is a row vector, so let's see what H[2] looks like
I 0 0 0 0 
S 0 0 0 0 
S 0 0 0 0 
0 0 I 0 0 
0 S 0 S I 
#
#  Now do a block respecting QX decomposition. QL decomposition is chosen because
#  H[2] is in lower regular form and the default ortho direction is left.
#
julia>Q,L,iq=block_qx(H[2];rr_cutoff=1e-14); #Block respecting QL
#
#  The first column of Q is unchanged because it is outside the V-block.
#  Also one column was removed because set rr_cutoff to enable rank revealing QX.
#
julia>pprint(Q)
I 0 0 0 
S 0 0 0 
S 0 0 0 
0 I 0 0 
0 0 S I 
#
#  Similarly L is missing one row due to the rank revealing QX algorithm
#
julia>pprint(L,iq) #we need to tell pprint the iq is the row index.
I 0 0 0 0 
0 0 I 0 0 
0 S 0 S 0 
0 0 0 0 I 
```
"""
function block_qx(W::ITensor,ul::reg_form=lower;kwargs...)::Tuple{ITensor,ITensor,Index}
  lr::orth_type=get(kwargs, :orth, left)
  f,r=parse_links(W,lr)
  return block_qx(W,f,ul;kwargs...)
end

#
#  try to use this one for internal development.
#
function block_qx(W_::ITensor,forward::Index,ul::reg_form=lower;kwargs...)::Tuple{ITensor,ITensor,Index}
  #
  # Copy so that we don't mess up the original MPO
  #
  W=copy(W_) 
  #
  # settle the left/right && upper/lower question
  #
  lr::orth_type=get(kwargs, :orth, left)
  qtags=ts"Link,qx"
  ms=matrix_state(ul,lr)
  ilw=copy(forward) #get the link to the next site. 
  offset=V_offsets(ms)
  V,qn=getV(W,offset) #extract the V block and the QN for excluded row/column
  ind_on_V=filterinds(inds(V),tags=tags(ilw))[1] #link to next site 
  inds_on_Q=noncommoninds(inds(V),ind_on_V) #group all other indices for QX factorization
  @checkflux(W)
  @checkflux(V)
  if ul==lower
    if lr==left
      Q,X,iq=ql(V,inds_on_Q;positive=true,tags=qtags,kwargs...) #block respecting QL decomposition
      p=get_firstrow_perm(X)
    else #right
      X,Q,iq=lq(V,ind_on_V;positive=true,tags=qtags,kwargs...) #block respecting LQ decomposition
      p=get_firstcol_perm(X)
    end
  else #upper
    if lr==left
      Q,X,iq=qr(V,inds_on_Q;positive=true,tags=qtags,kwargs...) #block respecting QR decomposition
      p=get_firstcol_perm(X)
    else #right
      X,Q,iq=rq(V,ind_on_V;positive=true,tags=qtags,kwargs...) #block respecting RQ decomposition
      p=get_firstrow_perm(X)
    end
  end
  Q,X=restore_triform!(Q,X,iq,p) #Block sparse decomp does not gaurantee preservation of tri form for R or L.
  set_scale!(X,Q,offset) #rescale so the L(n,n)==1.0
  
  ITensors.@debug_check begin
    err=norm(V-X*Q)
    if  err>1e-11
      @warn "Loss of precision in block_qx, norm(V-RL*Q)=$err"
    end
  end
  Xplus,iqx=growRL(X,ilw,offset,qn) #Now make a full size version of X=R/L
  W=setV(W,Q,iqx,ms) #Q is the new V, stuff Q into W. This can resize W
  @mpoc_assert dir(W,iqx)==dir(iqx) #Check and QN directions are consistent
  @mpoc_assert dir(Xplus,iqx)==dir(dag(iqx)) #Check and QN directions are consistent
  @mpoc_assert hastags(W,qtags)
  @mpoc_assert hastags(Xplus,qtags)
  return W,Xplus,iqx
end

function restore_triform!(Q::ITensor,RL::ITensor,iq::Index,p::Vector{Int64})
  if hasqns(RL) 
    @checkflux(Q)
    @checkflux(RL)
    #@show dense(RL) p iq
    iqsp=NDTensors.permuteblocks(splitblocks(iq),p)
    G=G_perm(dag(iqsp'),iq,p) #gauge transform to permute rows of R into upper tri-form
    RL=noprime(RL*G,tags="Link,qx") #permute rows of R into upper tri-form
    Q=noprime(Q*dag(G),tags="Link,qx") #permute cols of Q accordingly.
    @checkflux(Q)
    @checkflux(RL)
    #@show dense(RL)
  end
  return Q,RL
end

function get_firstcol_perm(R::ITensor)
  @assert order(R)==2
  r,=inds(R,tags="Link,qx")
  c=noncommonind(R,r)
  first_cols=zeros(Int64,dim(r))
  for ir in eachindval(r)
    for ic in eachindval(c)
      #@show ir.second ic.second R[ir,ic]
      if abs(R[ir,ic])>0.0 #fortunately the off tri elements should be rigouresly zero, so they are easy to detect.
        first_cols[ir.second]=ic.second
        break
      end
    end
    if first_cols[ir.second]==0
      @warn "Zero row detected r=$(ir.second), please make sure rr_cutoff is set for qr routines."
      @show first_cols
    end
  end
  p=sortperm(first_cols)
  return p
end

function get_firstrow_perm(R::ITensor)
  @assert order(R)==2
  c,=inds(R,tags="Link,qx")
  r=noncommonind(R,c)
  first_rows=zeros(Int64,dim(c))
  for ic in eachindval(c)
    for ir in Iterators.reverse(eachindval(r)) #We need to sweep *up* to find the first non-zero row.
      if abs(R[ir,ic])>0.0 #fortunately the off tri elements should be rigouresly zero, so they are easy to detect.
        first_rows[ic.second]=ir.second
        break
      end
    end
    if first_rows[ic.second]==0
      @warn "Zero column detected c=$(ic.second), please make sure rr_cutoff is set for qr routines."
    end
  end
  #@show first_rows
  p=sortperm(first_rows)
  return p
end


function G_perm(i1::Index, i2::Index,perm::Vector)
  d = emptyITensor(i1, i2)
  for n in 1:Base.min(dim(i1), dim(i2))
    d[n, perm[n]] = 1.0
  end
  return d
end











function equal_edge_blocks(i1::ITensors.QNIndex,i2::ITensors.QNIndex)::Bool
  qns1,qns2=space(i1),space(i2)
  qn11,qn1n=qns1[1],qns1[nblocks(qns1)]
  qn21,qn2n=qns2[1],qns2[nblocks(qns2)]
  return ITensors.have_same_qns(qn(qn11),qn(qn21)) && ITensors.have_same_qns(qn(qn1n),qn(qn2n))
end

function equal_edge_blocks(::Index,::Index)::Bool
   return true
end

function redim1(iq::ITensors.QNIndex,pad1::Int64,pad2::Int64,qns::ITensors.QNBlocks)
  @assert pad1==blockdim(qns[1]) #Splitting blocks not supported
  @assert pad2==blockdim(qns[end]) #Splitting blocks not supported
  qnsp=[qns[1],space(iq)...,qns[end]] #creat the new space
  return Index(qnsp,tags=tags(iq),plev=plev(iq),dir=dir(iq)) #create new index.
end

function redim1(iq::Index,pad1::Int64,pad2::Int64,Dw::Int64)
  @assert dim(iq)+pad1+pad2<=Dw #Splitting blocks not supported
  return Index(dim(iq)+pad1+pad2,tags=tags(iq),plev=plev(iq),dir=dir(iq)) #create new index.
end


function insert_Q(Wb::regform_blocks,𝐐::ITensor,ileft::Index,ic::Index,iq::Index,ms::matrix_state)
  ilb,ilf =  ms.lr==left ? (ileft,ic) : (ic,ileft) #Backward and forward indices.
  @assert !isnothing(Wb.𝑨𝒄)
  is=noncommoninds(Wb.𝑨𝒄,Wb.irAc,Wb.icAc)
  @assert hasinds(𝐐,iq,is...)
#    @assert dir(ileft)==dir(dag(ic))

  #
  #  Build new index and MPO Tensor
  #
  iqp=redim1(iq,1,1,space(ilf))  #pad with 1 at the start and 1 and the end: iqp =(1,iq,1).
  Wp=ITensor(0.0,ilb,iqp,is)
  ileft,iright =  ms.lr==left ? (ilb,iqp) :  (iqp,ilb)
  set_𝒃𝒄_block!(Wp,Wb.𝒃,ileft,iright,ms) #preserve b or c block from old W
  set_𝒅_block!(Wp,Wb.𝒅,ileft,iright,ms.ul) #preserve d block from old W
  set_𝕀_block!(Wp,Wb.𝕀,ileft,iright,ms.ul) #init I blocks from old W
  set_𝑨𝒄_block(Wp,𝐐,ileft,iright,ms) #Insert new Qs form QR decomp
  return Wp,iqp
end

function ac_qx(W::reg_form_Op,lr::orth_type;kwargs...)
  @checkflux(W.W)
  #@assert dir(W.ileft)==dir(dag(W.iright))
  Wb=extract_blocks(W,lr;Ac=true,all=true)
  ilf_Ac = llur(matrix_state(W.ul,lr)) ?  Wb.icAc : Wb.irAc
  ilb,ilf =  lr==left ? (W.ileft,W.iright) : (W.iright,W.ileft) #Backward and forward indices.
  @checkflux(Wb.𝑨𝒄)
  if lr==left
      Qinds=noncommoninds(Wb.𝑨𝒄,ilf_Ac) 
      Q,R,iq=qr(Wb.𝑨𝒄,Qinds;positive=true,cutoff=1e-14,tags=tags(ilf))
  else
      Rinds=ilf_Ac
      R,Q,iq=lq(Wb.𝑨𝒄,Rinds;positive=true,cutoff=1e-14,tags=tags(ilf))
  end
  @checkflux(Q)
  @checkflux(R)
  # Re-scale
  dh=d(Wb) #dimension of local Hilbert space.
  @assert abs(dh-round(dh))==0.0
  Q*=sqrt(dh)
  R/=sqrt(dh)

  Wp,iqp=insert_Q(Wb,Q,W.ileft,W.iright,iq,matrix_state(W.ul,lr)) 
  Wprf=lr==left ? reg_form_Op(Wp,ilb,iqp,W.ul) : reg_form_Op(Wp,iqp,ilb,W.ul)
  @assert equal_edge_blocks(ilf,iqp)
  @assert is_regular_form(Wprf)
  R=prime(R,ilf_Ac) #both inds or R have the same tags, so we prime one of them so the grow function can distinguish.
  Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
  return Wprf,Rp,iqp
end
