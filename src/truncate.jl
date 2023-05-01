
@doc """
    truncate!(H::MPO)

Compress an MPO using block respecting SVD techniques as described in 
> *Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147*

# Arguments
- `H` MPO for decomposition. If `H` is not already in the correct canonical form for compression, it will automatically be put into the correct form prior to compression.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form for the final output. 
- `cutoff::Float64 = 0.0` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
- `maxdim::Int64` : If the number of singular values exceeds `maxdim`, only the largest `maxdim` will be retained.
- `mindim::Int64` : At least `mindim` singular values will be retained, even if some fall below the cutoff

# Example
```julia
julia> using ITensors
julia> using ITensorMPOCompression
julia> N=10; #10 sites
julia> NNN=7; #Include up to 7th nearest neighbour interactions
julia> sites = siteinds("S=1/2",N);
#
# This makes H directly, bypassing autoMPO.  (AutoMPO is too smart for this
# demo, it makes maximally reduced MPOs right out of the box!)
#
julia> H=transIsing_MPO(sites,NNN);
#
#  Make sure we have a regular form or truncate! won't work.
#
julia> is_lower_regular_form(H)==true
true

#
#  Now we can truncate with defaults of left orthogonal cutoff=1e-14.
#  truncate! returns the spectrum of singular values at each bond.  The largest
#  singular values are remaining well under control.  i.e. no sign of divergences.
#
julia> @show truncate!(H);
site  Ns   max(s)     min(s)    Entropy  Tr. Error
   1    1  0.30739   3.07e-01   0.22292  0.00e+00
   2    2  0.35392   3.49e-02   0.26838  0.00e+00
   3    3  0.37473   2.06e-02   0.29133  0.00e+00
   4    4  0.38473   1.77e-02   0.30255  0.00e+00
   5    5  0.38773   7.25e-04   0.30588  0.00e+00
   6    4  0.38473   1.77e-02   0.30255  0.00e+00
   7    3  0.37473   2.06e-02   0.29133  0.00e+00
   8    2  0.35392   3.49e-02   0.26838  0.00e+00
   9    1  0.30739   3.07e-01   0.22292  0.00e+00

julia> pprint(H[2])
I 0 0 0 
S S S 0 
0 S S I 

#
#  We can see that bond dimensions have been drastically reduced.
#
julia> get_Dw(H)
9-element Vector{Int64}: 3 4 5 6 7 6 5 4 3

julia> is_lower_regular_form(H)==true
true

julia> isortho(H,left)==true
true

```
"""
function truncate(
  Ŵrf::reg_form_Op, lr::orth_type; kwargs...
)::Tuple{reg_form_Op,ITensor,Spectrum}
  @mpoc_assert Ŵrf.ul==lower
  ilf = forward(Ŵrf, lr)
  #   l=n-1   l=n        l=n-1  l=n  l=n
  #   ------W----   -->  -----Q-----R-----
  #           ilf               iqx   ilf
  Q̂, R, iqx = ac_qx(Ŵrf, lr; qprime=true, kwargs...) #left Q[r,qx], R[qx,c] - right R[r,qx] Q[qx,c]
  @checkflux(Q̂.W)
  @checkflux(R)
  if dim(ilf)>dim(iqx)
    @warn "Truncate bail out, dim(ilf)=$(dim(ilf)), dim(iqx)=$(dim(iqx))"
    return  noprime(Q̂), noprime(R), Spectrum(nothing,0.0)
  end
  @mpoc_assert dim(ilf) == dim(iqx) #Rectanuglar not allowed
  #
  #  Factor RL=M*L' (left/lower) = L'*M (right/lower) = M*R' (left/upper) = R'*M (right/upper)
  #  M will be returned as a Dw-2 X Dw-2 interior matrix.  M_sans in the Parker paper.
  #
  Dw = dim(iqx)
  M = R[dag(iqx) => 2:(Dw - 1), ilf => 2:(Dw - 1)]
  M = replacetags(M, tags(ilf), "Link,m"; plev=0)
  #  l=n'    l=n           l=n'   m      l=n
  #  ------R----   --->   -----M------R'----
  #  iqx'    ilf           iqx'   im     ilf
  R⎖, im = build_R⎖(R, iqx, ilf) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
  #  
  #  svd decomp M. 
  #    
  isvd = inds(M; tags=tags(iqx))[1] #decide the backward index for svd.  Works for both sweep directions.
  U, s, V, spectrum, iu, iv = svd(M, isvd; kwargs...) # ns sing. values survive compression
  #@show diag(array(s))
  #
  #  Now recontrsuct R, and W in the truncated space.
  #
  iup = redim(iu, 1, 1, space(iqx))
  R = grow(s * V, iup, im) * R⎖ #RL[l=n,u] dim ns+2 x Dw2
  Uplus = grow(U, dag(iqx), dag(iup))
  Uplus = noprime(Uplus, iqx)
  Ŵrf = Q̂ * Uplus #W[l=n-1,u]

  R = replacetags(R, "Link,u", tags(ilf)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
  Ŵrf = replacetags(Ŵrf, "Link,u", tags(ilf)) #W[l=n-1,l=n]
  check(Ŵrf)
  return Ŵrf, R, spectrum
end

function truncate!(
  H::reg_form_MPO, lr::orth_type; eps=1e-14, kwargs...
)::bond_spectrums
  #Two sweeps are essential for avoiding rectangular R in site truncate.
  if !isortho(H)
    orthogonalize!(H, lr; eps=eps, kwargs...)
    orthogonalize!(H, mirror(lr); eps=eps, kwargs...)
  end
  gauge_fix!(H)
  ss = bond_spectrums(undef, 0)
  rng = sweep(H, lr)
  for n in rng
    nn = n + rng.step
    H[n], R, s = truncate(H[n], lr; kwargs...)
    H[nn] *= R
    push!(ss, s)
  end
  H.rlim = rng.stop + rng.step + 1
  H.llim = rng.stop + rng.step - 1
  return ss
end
