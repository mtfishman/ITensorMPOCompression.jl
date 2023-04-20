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


function insert_Q(Wb::regform_blocks,𝐐::ITensor,ileft::Index,ic::Index,iq::Index,ms::matrix_state)
  ilb,ilf =  ms.lr==left ? (ileft,ic) : (ic,ileft) #Backward and forward indices.
  @assert !isnothing(Wb.𝑨𝒄)
  is=noncommoninds(Wb.𝑨𝒄,Wb.irAc,Wb.icAc)
  @assert hasinds(𝐐,iq,is...)
  #
  #  Build new index and MPO Tensor
  #
  iqp=redim1(iq,1,1,space(ilf))  #pad with 1 at the start and 1 and the end: iqp =(1,iq,1).
  Wp=ITensor(0.0,ilb,iqp,is)
  ileft,iright =  ms.lr==left ? (ilb,iqp) :  (iqp,ilb)
  Wrfp=reg_form_Op(Wp,ileft,iright,ms.ul)
  set_𝒃𝒄_block!(Wrfp,Wb.𝒃,ms.lr) #preserve b or c block from old W
  set_𝒅_block!(Wrfp,Wb.𝒅) #preserve d block from old W
  set_𝕀_block!(Wrfp,Wb.𝕀) #init I blocks from old W
  set_𝑨𝒄_block(Wrfp,𝐐,ms.lr) #Insert new Qs form QR decomp
  return Wrfp.W,iqp
end

function ac_qx(Wrf::reg_form_Op,lr::orth_type;verbose=false, kwargs...)
  @checkflux(Wrf.W)
  Wb=extract_blocks(Wrf,lr;Ac=true,all=true)
  ilf_Ac = llur(Wrf,lr) ?  Wb.icAc : Wb.irAc
  ilb,ilf =  lr==left ? (Wrf.ileft,Wrf.iright) : (Wrf.iright,Wrf.ileft) #Backward and forward indices.
  @checkflux(Wb.𝑨𝒄)
  if lr==left
      Qinds=noncommoninds(Wb.𝑨𝒄,ilf_Ac) 
      Q,R,iq,p=qr(Wb.𝑨𝒄,Qinds;verbose=verbose,positive=true,cutoff=1e-14,tags=tags(ilf))
  else
      Rinds=ilf_Ac
      R,Q,iq,p=lq(Wb.𝑨𝒄,Rinds;verbose=verbose,positive=true,cutoff=1e-14,tags=tags(ilf))
  end
  @checkflux(Q)
  @checkflux(R)
  # Re-scale
  dh=d(Wb) #dimension of local Hilbert space.
  @assert abs(dh-round(dh))==0.0
  Q*=sqrt(dh)
  R/=sqrt(dh)

  Wp,iqp=insert_Q(Wb,Q,Wrf.ileft,Wrf.iright,iq,matrix_state(Wrf.ul,lr)) 
  Wprf=lr==left ? reg_form_Op(Wp,ilb,iqp,Wrf.ul) : reg_form_Op(Wp,iqp,ilb,Wrf.ul)
  @assert equal_edge_blocks(ilf,iqp)
  @assert is_regular_form(Wprf)
  R=prime(R,ilf_Ac) #both inds or R have the same tags, so we prime one of them so the grow function can distinguish.
  Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
  p=add_edges(p)
  return Wprf,Rp,iqp,p
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