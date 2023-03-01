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

function δ_split(i1::Index, i2::Index,perm::Vector)
  d = emptyITensor(i1, i2)
  for n in 1:Base.min(dim(i1), dim(i2))
    d[n, perm[n]] = 1
  end
  return d
end


# function permute1(Q::ITensor,RL::ITensor,p::Vector)
#   iq=commonind(Q,RL)
#   @show p
#   iqp=NDTensors.permuteblocks(iq,p)
#   Qp=ITensor(iqp,noncommoninds(Q,iq))
#   RLp=ITensor(0.0,dag(iqp),noncommoninds(RL,iq))
#   @show RL RLp
#   #@show inds(Q) inds(Qp) inds(RL) inds(RLp)
#   for i in 1:dim(iq)
#     @show i p[i]
#     #op=slice(Q,iq=>i)
#     #@show op space(iq)[i] space(iqp)[p[i]]
#     #op2=slice(Qp,iqp=>p[i])
#     #@show op2 flux(op) flux(op2)
#     #assign!(Qp,op,iqp=>p[i])
#     op=slice(RL,dag(iq)=>i)
#     @show op
#     assign!(RLp,op,iqp=>p[i])
#   end
#   return Qp,RLp
# end

function getperm1(RL::ITensor,eps::Float64=1e-14)
  @mpoc_assert order(RL)==2
  c,=inds(RL,tags="Link,qx")
  r=noncommonind(RL,c)
  first_rows=zeros(Int64,dim(c))
  for ic in eachindval(c)
    for ir in eachindval(r)
      if abs(RL[ir,ic])>=eps
        first_rows[ic.second]=ir.second
        break
      end
    end
  end
  p=sortperm(first_rows)
  return p
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
  ms=matrix_state(ul,lr)
  ilw=copy(forward) #get the link to the next site. 
  offset=V_offsets(ms)
  V=getV(W,offset) #extract the V block
  ind_on_V=filterinds(inds(V),tags=tags(ilw))[1] #link to next site 
  inds_on_Q=noncommoninds(inds(V),ind_on_V) #group all other indices for QX factorization
  #@show get(kwargs,:rr_cutoff,-999.0)
  if ul==lower
    if lr==left
      #@show inds(V) inds_on_Q
      Q,RL,iq=ql(V,inds_on_Q;positive=true,tags="Link,qx",kwargs...) #block respecting QL decomposition
    else #right
      RL,Q,iq=lq(V,ind_on_V;positive=true,tags="Link,qx",kwargs...) #block respecting LQ decomposition
    end
  else #upper
    if lr==left
      Q,RL,iq=qr(V,inds_on_Q;positive=true,tags="Link,qx",kwargs...) #block respecting QR decomposition
    else #right
      RL,Q,iq=rq(V,ind_on_V;positive=true,tags="Link,qx",kwargs...) #block respecting RQ decomposition
    end
  end
  #@show RL flux(Q) flux(RL)
  if hasqns(iq)
    iqs=splitblocks(iq)
    p=getperm1(RL)
    #@show inds(RL,tags="Link") dense(RL) p iqs
    iqsp=NDTensors.permuteblocks(iqs,p)
    d=δ_split(dag(iqsp'),iq,p)
    #@show flux(d)
    RL=noprime(RL*d)
    Q=noprime(Q*dag(d),tags="Link,qx")
    #@show flux(Q) flux(RL)
    #@show norm(dense(V)-dense(RL*Q)) dense(RL)
  end
  set_scale!(RL,Q,offset) #rescale so the L(n,n)==1.0
  
  ITensors.@debug_check begin
    err=norm(V-RL*Q)
    if  err>1e-11
      @warn "Loss of precision in block_qx, norm(V-RL*Q)=$err"
    end
  end
  RLplus,iqx=growRL(RL,ilw,offset) #Now make a full size version of RL
  #@pprint Q
  #@pprint W
  W=setV(W,Q,iqx,ms) #Q is the new V, stuff Q into W. This can resize W
  #@show flux(RLplus) flux(W)
  #@pprint W
  @mpoc_assert dir(W,iqx)==dir(iqx) #Check and QN directions are consistent
  @mpoc_assert hastags(W,"qx")
  @mpoc_assert hastags(RLplus,"qx")
  return W,RLplus,iqx
end
