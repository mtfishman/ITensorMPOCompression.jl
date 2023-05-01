#-----------------------------------------------------------------
#
#   Ac block respecting orthogonalization.
#
@doc """
    orthogonalize!(H::MPO)

Bring an MPO into left or right canonical form using block respecting QR decomposition
 as described in:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=`1.0 indicates no rank reduction.

# Examples
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
#  Make sure we have a regular form or orhtogonalize! won't work.
#
julia> is_lower_regular_form(H)==true
true
#
#  Let's see what the second site for the MPO looks like.
#  I = unit operator, and S = any other operator
#
julia> pprint(H[2])
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
.
.
.
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 I 0 0 
0 S 0 S 0 0 S 0 0 0 S 0 0 0 0 S 0 0 0 0 0 S 0 0 0 0 0 0 S I 
#
#  Now we can orthogonalize or bring it into canonical form.
#  Defaults are left orthogonal with rank reduction.
#
julia> orthogonalize!(H;rr_cutoff=1e-14)
#
#  Wahoo .. rank reduction knocked the size of H way down, and we haven't
#  tried compressing yet!
#
julia> pprint(H[2])
I 0 0 0 
S I 0 0 
0 0 S I 
#
#  What do all the bond dimensions of H look like?  We will need compression 
#  (truncation) in order to further bang down the size of H
#
julia> get_Dw(H)
9-element Vector{Int64}: 3 4 5 6 7 6 5 4 3
#
#  wrap up with two more checks on the structure of H
#
julia> is_lower_regular_form(H)==true
true
julia> isortho(H,left)==true
true


```

"""
function ITensors.orthogonalize!(H::reg_form_MPO, lr::orth_type; kwargs...)
  gauge_fix!(H;kwargs...)
  rng = sweep(H, lr)
  for n in rng
    nn = n + rng.step
    H[n], R, iqp = ac_qx(H[n], lr;kwargs...)
    H[nn] *= R
    check(H[n])
    check(H[nn])
  end
  H.rlim = rng.stop + rng.step + 1
  H.llim = rng.stop + rng.step - 1
  return
end

function ITensors.orthogonalize!(H::reg_form_MPO, j::Int64; kwargs...)
  if !isortho(H)
    orthogonalize!(H,right;kwargs...)
  end
  oc=orthocenter(H)
  rng= oc<=j ? (oc:1:j-1) : (oc-1:-1:j)
  for n in rng
    nn = n + rng.step
    H[n], R, iqp = ac_qx(H[n], left;kwargs...)
    H[nn] *= R
    check(H[n])
    check(H[nn])
  end
  H.rlim = j + 1
  H.llim = j - 1
  return 
end

function ITensors.orthogonalize!(H::MPO, lr::orth_type; kwargs...)
  Hrf = reg_form_MPO(H)
  orthogonalize!(Hrf, lr; kwargs)
  copy!(H,Hrf)
end

function copy!(H::MPO,Hrf::reg_form_MPO)
    for n in eachindex(H)
      H[n]=Hrf[n].W
    end
    N = length(Hrf)
    H[1]*=dag(Hrf.d0)
    H[N]*=dag(Hrf.dN)
    H.llim,H.rlim=Hrf.llim,Hrf.rlim
end

function ITensors.orthogonalize!(H::MPO, j::Int64; kwargs...)
  Hrf = reg_form_MPO(H)
  orthogonalize!(Hrf, j; kwargs)
  copy!(H,Hrf)
end
