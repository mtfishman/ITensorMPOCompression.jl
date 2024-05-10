#-----------------------------------------------------------------
#
#   Ac block respecting orthogonalization.
#
@doc """
    orthogonalize!(H::MPO,lr::orth_type)

Bring an MPO into left or right canonical form using block respecting QR decomposition.

# Arguments
- `H::MPO` : Matrix Product Operator to be orthogonalized
- `lr::orth_type` : Choose `left` or `right` orthgonal/canoncial form for the output.

# Keywords
- `atol::Float64 = 1e-14` : Absolute cutoff for rank revealing QR which removes zero pivot rows.  
- `rtol::Float64 = -1.0` : Relative cutoff for rank revealing QR which removes zero pivot rows.  
- `verbose::Bool = false` : Show some details of the orthogonalization process.

atol=`-1.0 and rtol=`-1.0  indicates no rank reduction.

# Examples
```julia
julia> using ITensors, ITensorMPS
julia> using ITensorMPOCompression
include("../test/hamiltonians/hamiltonians.jl");
julia> N=10; #10 sites
julia> NNN=7; #Include up to 7th nearest neighbour interactions
julia> sites = siteinds("S=1/2",N);
julia> H=transIsing_MPO(sites,NNN);
#
#  Make sure we have a regular form or orhtogonalize! won't work.
#
julia> is_regular_form(H,lower)==true
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
julia> orthogonalize!(H,left);
#
#  Wahoo .. rank reduction knocked the size of H way down, and we haven't
#  tried compressing yet!
#
julia> pprint(H[2])
I 0 0 0 
S I 0 0 
0 0 S I 
#
#  What do all the bond dimensions of H look like?  
#
julia> @show get_Dw(H);
get_Dw(H) = [3, 4, 5, 6, 7, 8, 9, 9, 9]
#
#  wrap up with two more checks on the structure of H
#
julia> is_lower_regular_form(H)==true
true
julia> isortho(H,left)==true
true


```

"""
function ITensors.orthogonalize!(H::MPO, lr::orth_type; kwargs...)
  Hrf = reg_form_MPO(H)
  orthogonalize!(Hrf, lr; kwargs)
  copy!(H,Hrf)
end

@doc """
    orthogonalize!(H::MPO,j::Int64)

Bring an MPO into mixed canonical form with the orthogonality center on site `j`.

# Arguments
- `H::MPO` : Matrix Product Operator to be orthogonalized
- `j::Int64` : Site index for the orthogonality centre.

# Keywords
- `atol::Float64 = 1e-14` : Absolute cutoff for rank revealing QR which removes zero pivot rows.  
- `rtol::Float64 = -1.0` : Relative cutoff for rank revealing QR which removes zero pivot rows.  
- `verbose::Bool = false` : Show some details of the orthogonalization process.

atol=`-1.0 and rtol=`-1.0  indicates no rank reduction.

# Examples
```julia
julia> using ITensors, ITensorMPS
julia> using ITensorMPOCompression
include("../test/hamiltonians/hamiltonians.jl");
julia> N=10; #10 sites
julia> NNN=7; #Include up to 7th nearest neighbour interactions
julia> sites = siteinds("S=1/2",N);
julia> H=transIsing_MPO(sites,NNN);
julia> orthogonalize!(H,5);
julia> pprint(H)
  n    Dw1  Dw2   d   Reg.  Orth.
                      Form  Form 
   1    1    3    2    L      B
   2    3    4    2    L      L
   3    4    5    2    L      L
   4    5    6    2    L      L
   5    6    7    2    L      M
   6    7    6    2    L      R
   7    6    5    2    L      R
   8    5    4    2    L      R
   9    4    3    2    L      R
  10    3    1    2    L      B
```

"""
function ITensors.orthogonalize!(H::MPO, j::Int64; kwargs...)
  Hrf = reg_form_MPO(H)
  orthogonalize!(Hrf, j; kwargs...)
  copy!(H,Hrf)
end


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



