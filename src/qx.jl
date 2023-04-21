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

#   Q̂

  
function insert_Q(Ŵrf::reg_form_Op,Q̂::ITensor,iq::Index,lr::orth_type)
  #
  #  Create new index by growing iq.
  #
  ilb,ilf = linkinds(Ŵrf,lr) #Backward and forward indices.
  iq⎖=redim1(iq,1,1,space(ilf))  #pad with 1 at the start and 1 and the end: iqp =(1,iq,1).
  ileft,iright =  lr==left ? (ilb,iq⎖) :  (iq⎖,ilb)
  #
  #  Create a new reg form tensor
  #
  Ŵ=ITensor(0.0,ileft,iright,siteinds(Ŵrf))
  Ŵrf⎖=reg_form_Op(Ŵ,ileft,iright,Ŵrf.ul)
  #
  #  Preserve b,c,d blocks and insert Q
  #
  Wb=extract_blocks(Ŵrf,lr;b=true,c=true,d=true)
  set_𝐛̂𝐜̂_block!(Ŵrf⎖,Wb.𝐛̂,lr) #preserve b or c block from old W
  set_𝐝̂_block!(Ŵrf⎖,Wb.𝐝̂) #preserve d block from old W
  set_𝕀_block!(Ŵrf⎖,Wb.𝕀) #init I blocks from old W
  set_𝐀̂𝐜̂_block(Ŵrf⎖,Q̂,lr) #Insert new Qs form QR decomp

  return Ŵrf⎖,iq⎖
end

function ac_qx(Ŵrf::reg_form_Op,lr::orth_type;qprime=false,verbose=false, kwargs...)
  @checkflux(Ŵrf.W)
  Wb=extract_blocks(Ŵrf,lr;Ac=true)
  ilf_Ac = llur(Ŵrf,lr) ?  Wb.icAc : Wb.irAc
  ilf =  forward(Ŵrf,lr) #Backward and forward indices.
  @checkflux(Wb.𝐀̂𝐜̂)
  if lr==left
      Qinds=noncommoninds(Wb.𝐀̂𝐜̂,ilf_Ac) 
      Q̂,R,iq,p=qr(Wb.𝐀̂𝐜̂,Qinds;verbose=verbose,positive=true,cutoff=1e-14,tags=tags(ilf))
  else
      Rinds=ilf_Ac
      R,Q̂,iq,p=lq(Wb.𝐀̂𝐜̂,Rinds;verbose=verbose,positive=true,cutoff=1e-14,tags=tags(ilf))
  end
  @checkflux(Q̂)
  @checkflux(R)
  # Re-scale
  dh=d(Wb) #dimension of local Hilbert space.
  @assert abs(dh-round(dh))==0.0 #better be an integer!
  Q̂*=sqrt(dh)
  R/=sqrt(dh)

  Ŵrf⎖,iq⎖=insert_Q(Ŵrf,Q̂,iq,lr) #create a new W with Q.  The size may change.
  @assert equal_edge_blocks(ilf,iq⎖)
  
  #both inds or R have the same tags, so we prime one of them so the grow function can distinguish.
  R⎖=grow(prime(R,iq),dag(iq⎖)',ilf)
  p=add_edges(p) #grow p so we can apply it to Rp.
  if qprime
    iq⎖=prime(iq⎖)
  else
    R⎖=noprime(R⎖)
  end
  return Ŵrf⎖,R⎖,iq⎖,p
end

function add_edges(p::Vector{Int64})
  Dw=length(p)+2
  return [1,(p.+1)...,Dw]
end

#
#  This assumes the edge D=1 blocks appear first in the block list
#  If this fails we need to return a dict{Block,Vector{Int64}} so we can 
#  associate block with perm vectors
#
function add_edges(p::Vector{Vector{Int64}})
    return [[1],[1],p...]
end