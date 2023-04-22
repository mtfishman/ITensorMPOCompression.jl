
@doc """
make_3body_AutoMPO(sites;kwargs...)

Use `ITensor.autoMPO` to reproduce the 3 body Hamiltonian defined in eq. 34 of the Parker paper. 
The MPO is returned in lower regular form.

# Arguments
- `sites` : Site set defining the lattice of sites.
# Keywords
- `hx::Float64 = 0.0` : External magnetic field in `x` direction.
"""
function make_3body_AutoMPO(sites; kwargs...)
  hx::Float64 = get(kwargs, :hx, 0.0)

  N = length(sites)
  os = OpSum()
  if hx != 0
    os = make_1body(os, N, hx)
  end
  os = make_2body(os, N; kwargs...)
  os = make_3body(os, N; kwargs...)
  return MPO(os, sites; kwargs...)
end

function make_1body(os::OpSum, N::Int64, hx::Float64=0.0)::OpSum
  for n in 1:N
    add!(os, hx, "Sx", n)
  end
  return os
end

function make_2body(os::OpSum, N::Int64; kwargs...)::OpSum
  Jprime::Float64 = get(kwargs, :Jprime, 1.0)
  if Jprime != 0.0
    for n in 1:N
      for m in (n + 1):N
        Jnm = Jprime / abs(n - m)^4
        add!(os, Jnm, "Sz", n, "Sz", m)
        # if heis
        #     add!(os, Jnm*0.5,"S+", n, "S-", m)
        #     add!(os, Jnm*0.5,"S-", n, "S+", m)
        # end
      end
    end
  end
  return os
end

function make_3body(os::OpSum, N::Int64; kwargs...)::OpSum
  J::Float64 = get(kwargs, :J, 1.0)
  if J != 0.0
    for k in 1:N
      for n in (k + 1):N
        Jkn = J / abs(n - k)^2
        for m in (n + 1):N
          Jnm = J / abs(m - n)^2
          # if k==1
          #     @show k,n,m,Jkn,Jnm
          # end
          add!(os, Jnm * Jkn, "Sz", k, "Sz", n, "Sz", m)
        end
      end
    end
  end
  return os
end

@doc """
make_transIsing_AutoMPO(sites,NNN;kwargs...)

Use `ITensor.autoMPO` to build up a transverse Ising model Hamiltonian with up to `NNN` neighbour 2-body 
interactions.  The interactions are hard coded to decay like `J/(i-j)`. between sites `i` and `j`.
The MPO is returned in lower regular form.
 
# Arguments
- `sites` : Site set defining the lattice of sites.
- `NNN::Int64` : Number of neighbouring 2-body interactions to include in `H`

# Keywords
- `hx::Float64 = 0.0` : External magnetic field in `x` direction.
- `J::Float64 = 1.0` : Nearest neighbour interaction strength. Further neighbours decay like `J/(i-j)`..

"""
function make_transIsing_AutoMPO(sites, NNN::Int64; kwargs...)::MPO
  ul::reg_form = get(kwargs, :ul, lower)
  J::Float64 = get(kwargs, :J, 1.0)
  hx::Float64 = get(kwargs, :hx, 0.0)

  do_field = hx != 0.0
  N = length(sites)
  ampo = OpSum()
  if do_field
    for j in 1:N
      add!(ampo, hx, "Sx", j)
    end
  end
  for dj in 1:NNN
    f = J / dj
    for j in 1:(N - dj)
      add!(ampo, f, "Sz", j, "Sz", j + dj)
    end
  end
  H = MPO(ampo, sites; kwargs...)
  if ul == upper
    to_upper!(H)
  end
  return H
end
function make_2body_AutoMPO(sites, NNN::Int64; kwargs...)
  return make_transIsing_AutoMPO(sites, NNN; kwargs...)
end

@doc """
make_Heisenberg_AutoMPO(sites,NNN;kwargs...)

Use `ITensor.autoMPO` to build up a Heisenberg model Hamiltonian with up to `NNN` neighbour
2-body interactions.  The interactions are hard coded to decay like `J/(i-j)`. between sites `i` and `j`.
The MPO is returned in lower regular form.

# Arguments
- `sites` : Site set defining the lattice of sites.
- `NNN::Int64` : Number of neighbouring 2-body interactions to include in `H`

# Keywords
- `hz::Float64 = 0.0` : External magnetic field in `z` direction.
- `J::Float64 = 1.0` : Nearest neighbour interaction strength. Further neighbours decay like `J/(i-j)`.

"""

function make_Heisenberg_AutoMPO(sites, NNN::Int64; kwargs...)::MPO
  ul::reg_form = get(kwargs, :ul, lower)
  hz::Float64 = get(kwargs, :hz, 0.0)
  J::Float64 = get(kwargs, :J, 1.0)
  N = length(sites)
  @mpoc_assert(N >= NNN)
  ampo = OpSum()
  for j in 1:N
    add!(ampo, hz, "Sz", j)
  end
  for dj in 1:NNN
    f = J / dj
    for j in 1:(N - dj)
      add!(ampo, f, "Sz", j, "Sz", j + dj)
      add!(ampo, f * 0.5, "S+", j, "S-", j + dj)
      add!(ampo, f * 0.5, "S-", j, "S+", j + dj)
    end
  end
  H = MPO(ampo, sites; kwargs...)
  if ul == upper
    to_upper!(H)
  end
  return H
end

function make_Hubbard_AutoMPO(sites, NNN::Int64; kwargs...)::MPO
  ul::reg_form = get(kwargs, :ul, lower)
  U::Float64 = get(kwargs, :U, 1.0)
  t::Float64 = get(kwargs, :t, 1.0)
  V::Float64 = get(kwargs, :V, 0.5)
  N = length(sites)
  @mpoc_assert(N >= NNN)
  os = OpSum()
  for i in 1:N
    os += (U, "Nupdn", i)
  end
  for dn in 1:NNN
    tj, Vj = t / dn, V / dn
    for n in 1:(N - dn)
      os += -tj, "Cdagup", n, "Cup", n + dn
      os += -tj, "Cdagup", n + dn, "Cup", n
      os += -tj, "Cdagdn", n, "Cdn", n + dn
      os += -tj, "Cdagdn", n + dn, "Cdn", n
      os += Vj, "Ntot", n, "Ntot", n + dn
    end
  end
  H = MPO(os, sites; kwargs...)
  if ul == upper
    to_upper!(H)
  end
  return H
end

function reverse(i::QNIndex)
  return Index(Base.reverse(space(i)); dir=dir(i), tags=tags(i), plev=plev(i))
end
function reverse(i::Index)
  return Index(space(i); tags=tags(i), plev=plev(i))
end

function G_transpose(i::Index, iu::Index)
  D = dim(i)
  @mpoc_assert D == dim(iu)
  G = ITensor(0.0, dag(i), iu)
  for n in 1:D
    G[i => n, iu => D + 1 - n] = 1.0
  end
  return G
end

# function to_upper(W::ITensor)
#     l,r=parse_links(W)
#     lu=reverse(l)
#     ru=reverse(r)
#     Gl=G_transpose(l,lu)
#     Gr=G_transpose(r,ru)
#     if dim(l)==1
#         Wt=W*Gr
#     elseif dim(r)==1
#         Wt=Gl*W
#     else
#         Wt=Gl*W*Gr
#     end
#    return Wt
# end

function to_upper!(H::AbstractMPS)
  N = length(H)
  l, r = parse_links(H[1])
  G = G_transpose(r, reverse(r))
  H[1] = H[1] * G
  for n in 2:(N - 1)
    H[n] = dag(G) * H[n]
    l, r = parse_links(H[n])
    G = G_transpose(r, reverse(r))
    H[n] = H[n] * G
  end
  return H[N] = dag(G) * H[N]
end
