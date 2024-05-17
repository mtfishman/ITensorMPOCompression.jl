import ITensorMPOCompression: parse_links, parse_site, G_transpose, @mpoc_assert, reg_form, assign!, slice, redim

@doc """
    transIsing_MPO(sites,NNN;kwargs...)
 
Directly coded build up of a transverse Ising model Hamiltonian with up to `NNN` neighbour
2-body interactions.  The interactions are hard coded to decay like `J/(i-j)`. between sites `i` and `j`.
     
# Arguments
- `sites` : Site set defining the lattice of sites.
- `NNN::Int64=1` : Number of neighbouring 2-body interactions to include in `H`
# Keywords
- `hx::Float64=0.0` : External magnetic field in `x` direction.
- `ul::reg_form=lower` : build H with `lower` or `upper` regular form.
- `J::Float64=1.0` : Nearest neighbour interaction strength.  Further neighbours decay like `J/(i-j)`..

"""
function transIsing_MPO(::Type{ElT},sites, NNN::Int64;ul=lower,J=1.0,hx=0.0,pbc=false, kwargs...)::MPO  where {ElT<:Number}
  N = length(sites)
  Dw::Int64 = transIsing_Dw(NNN)
  use_qn::Bool = hasqns(sites[1])
  mpo = MPO(N)
  io = ul == lower ? ITensors.Out : ITensors.In

  if pbc
    prev_link = Ising_index(Dw, "Link,c=0,l=$(N)", use_qn, io)
  else
    prev_link = Ising_index(Dw, "Link,l=0", use_qn, io)
  end
  for n in 1:N
    mpo[n] = transIsing_op(sites[n], prev_link, NNN, J, hx, ul, pbc)
    prev_link = filterinds(mpo[n]; tags="Link,l=$n")[1]
  end
  if pbc
    i0 = filterinds(mpo[1]; tags="Link,c=0,l=$(N)")[1]
    i0 = replacetags(i0, "c=0", "c=1")
    in = filterinds(mpo[N]; tags="Link,c=1,l=$(N)")[1]
    mpo[N] = replaceind(mpo[N], in, i0)
  end
  if !pbc
    mpo = to_openbc(mpo) #contract with l* and *r at the edges.
  end
  return mpo
end

transIsing_MPO(sites, NNN::Int64;kwargs...)=transIsing_MPO(Float64,sites, NNN::Int64;kwargs...)

#  It turns out that the trans-Ising model
#  Dw = 2+Sum(i,i=1..NNN) =2 + NNN*(NNN+1)/2
#
function transIsing_Dw(NNN::Int64)::Int64
  return 2 + NNN * (NNN + 1) / 2
end

function Ising_index(Dw::Int64, tags::String, use_qn::Bool, dir)
  if (use_qn)
    if tags[1:4] == "Link"
      qns = fill(QN("Sz", 0) => 1, Dw)
      ind = Index(qns; dir=dir, tags=tags)
    else
      @mpoc_assert tags[1:4] == "Site"
      ind = Index(QN("Sz", 1) => Dw; dir=dir, tags=tags)
    end
  else
    ind = Index(Dw, tags)
  end
  return ind
end

# NNN = Number of Neighbours, for example
#    NNN=1 corresponds to nearest neighbour
#    NNN=2 corresponds to nearest and next nearest neighbour
function transIsing_op(
  site::Index,
  prev_link::Index,
  NNN::Int64,
  J::Float64,
  hx::Float64=0.0,
  ul::reg_form=lower,
  pbc::Bool=false,
)::ITensor
  @mpoc_assert NNN >= 1
  do_field = hx != 0.0
  Dw::Int64 = transIsing_Dw(NNN)
  d, n, space = parse_site(site)
  use_qn = hasqns(site)

  r = dag(prev_link)
  c = Ising_index(Dw, "Link,l=$n", use_qn, dir(prev_link))
  if pbc
    c = addtags(c, "c=1")
  end
  is = dag(site) #site seem to have the wrong direction!
  W = ITensor(r, c, is, dag(is'))
  Id = op(is, "Id")
  Sz = op(is, "Sz")
  if do_field
    Sx = op(is, "Sx")
  end
  assign!(W, Id, r => 1, c => 1)
  assign!(W, Id, r => Dw, c => Dw)

  iblock = 1
  if ul == lower
    if do_field
      assign!(W, hx * Sx, r => Dw, c => 1) #add field term
    end
    #very hard to explain this without a diagram.
    for iNN in 1:NNN
      assign!(W, Sz, r => iblock + 1, c => 1)
      for jNN in 1:(iNN - 1)
        assign!(W, Id, r => iblock + 1 + jNN, c => iblock + jNN)
      end
      Jn = J / (iNN) #interactions need to decay with distance in order for H to extensive 
      assign!(W, Jn * Sz, r => Dw, c => iblock + iNN)
      iblock += iNN
    end
  else
    if do_field
      assign!(W, hx * Sx, r => 1, c => Dw) #add field term
    end
    #very hard to explain this without a diagram.
    for iNN in 1:NNN
      assign!(W, Sz, r => 1, c => iblock + 1)
      for jNN in 1:(iNN - 1)
        assign!(W, Id, r => iblock + jNN, c => iblock + 1 + jNN)
      end
      Jn = J / (iNN) #interactions need to decay with distance in order for H to extensive 
      assign!(W, Jn * Sz, r => iblock + iNN, c => Dw)
      iblock += iNN
    end
  end
  return W
end

#
#  implement eq. E3 from the Parker paper.
#
#         | I c1 c2 d1+d2 |
# W1+W2 = | 0 A1 0    b1  |
#         | 0 0  A2   b2  |
#         | 0 0  0    I   |
# or
#
#         |  I    0  0  0 |
# W1+W2 = |  b1   A1 0  0 |
#         |  b2   0  A2 0 |
#         | d1+d2 c1 c2 I |
# χ
#
function add_ops(i1::ITensors.QNIndex,i2::ITensors.QNIndex)
  qns=[space(i1)[1:end-1]...,space(i2)[2:end]...]
  return Index(qns,tags=tags(l1),plev=plev(i1),dir=dir(i1))
end
function add_ops(i1::Index,i2::Index)
  return Index(space(i1)+space(i2)-2,tags=tags(i1),plev=plev(i1),dir=dir(i1))
end

function add_ops(W1::ITensor, W2::ITensor)::ITensor
  #@pprint(W1)
  #@pprint(W2)
  is1 = inds(W1; tags="Site", plev=0)[1]
  is2 = inds(W2; tags="Site", plev=0)[1]
  @mpoc_assert is1 == is2
  l1, r1 = parse_links(W1)
  l2, r2 = parse_links(W2)
  @mpoc_assert tags(l1) == tags(l2)
  @mpoc_assert tags(r1) == tags(r2)
  χl1, χr1 = dim(l1) - 2, dim(r1) - 2
  χl2, χr2 = dim(l2) - 2, dim(r2) - 2
  χl, χr = χl1 + χl2, χr1 + χr2
  l, r = add_ops(l1,l2), add_ops(r1,r2)
  W = ITensor(0.0, l, r, is1, is1')
  Id = slice(W1, l1 => 1, r1 => 1)
  assign!(W, Id, l => 1, r => 1)
  assign!(W, Id, l => χl + 2, r => χr + 2)
  # d1+d2 block
  d1 = slice(W1, l1 => χl1 + 2, r1 => 1)
  d2 = slice(W2, l2 => χl2 + 2, r2 => 1)
  assign!(W, d1 + d2, l => χl + 2, r => 1)

  W[l => 2:(χl1 + 1), r => 1:1] = W1[l1 => 2:(χl1 + 1), r1 => 1:1]  # b1 block
  W[l => (χl1 + 2):(χl1 + χl2 + 1), r => 1:1] = W2[l2 => 2:(χl2 + 1), r2 => 1:1]  # b2 block
  W[l => (χl + 2):(χl + 2), r => 2:(χr1 + 1)] = W1[
    l1 => (χl1 + 2):(χl1 + 2), r1 => 2:(χr1 + 1)
  ]  # c1 block
  W[l => (χl + 2):(χl + 2), r => (χr1 + 2):(χr1 + χr2 + 1)] = W2[
    l2 => (χl2 + 2):(χl2 + 2), r2 => 2:(χr2 + 1)
  ]  # c2 block
  W[l => 2:(χl1 + 1), r => 2:(χr1 + 1)] = W1[l1 => 2:(χl1 + 1), r1 => 2:(χr1 + 1)]  # A1 block
  W[l => (χl1 + 2):(χl1 + χl2 + 1), r => (χr1 + 2):(χr1 + χr2 + 1)] = W2[
    l2 => 2:(χl2 + 1), r2 => 2:(χr2 + 1)
  ]  # A2 block
  return W
end

function daisychain_links!(H::MPO; pbc=false)
  N = length(H)
  for n in 2:N
    if pbc
      ln = inds(H[n]; tags="c=0")[1]
      replacetags!(H[n], "c=1", "c=$n")
    else
      ln = inds(H[n]; tags="l=0")[1]
      replacetags!(H[n], "l=1", "l=$n")
    end
    l1, r1 = parse_links(H[n - 1])
    #@show l1 r1 ln
    #@show inds(H[n])
    replaceind!(H[n], ln, r1)
    #@show inds(H[n])
  end
  if pbc
    i0 = filterinds(H[1]; tags="Link,c=0,l=$(N)")[1]
    i0 = replacetags(i0, "c=0", "c=1")
    in = filterinds(H[N]; tags="Link,c=1,l=$(N)")[1]
    H[N] = replaceind(H[N], in, i0)
  end

  if !pbc
    H = to_openbc(H) #contract with l* and *r at the edges.
  end
end

# Add the W operator to each site and fix up all tags accordingly.
function addW!(sites, H, W)
  N = length(sites)
  H[1] = add_ops(H[1], W)
  for n in 2:N
    #@show inds(H[n])
    iw1, iw2 = inds(W; tags="Link")
    ih1, ih2 = inds(H[n]; tags="Link")
    replacetags!(W, tags(iw2), tags(ih2))
    replacetags!(W, tags(iw1), tags(ih1))
    is = sites[n]
    replaceinds!(W, inds(W; tags="Site"), (is, dag(is)'))
    H[n] = add_ops(H[n], W)
  end
end

function two_body_MPO(sites, NNN::Int64; pbc=false, Jprime=1.0, presummed=true, kwargs...)
  N = length(sites)
  H = MPO(N)
  if pbc
    l, r = Index(1, "Link,c=0,l=1"), Index(1, "Link,c=1,l=1")
  else
    l, r = Index(1, "Link,l=0"), Index(1, "Link,l=1")
  end
  ul = lower
  H[1] = one_body_op(sites[1], l, r, ul; kwargs...)

  for n in 2:N
    H[n] = one_body_op(sites[n], l, r, ul; kwargs...)
  end

  W = H[1]
  if Jprime != 0.0
    if presummed
      W = add_ops(W, two_body_sum(sites[1], l, r, NNN, ul; kwargs...))
    else
      for n in 1:NNN
        W = add_ops(W, two_body_op(sites[1], l, r, n, ul; kwargs...))
      end
    end
  end
  addW!(sites, H, W)
  daisychain_links!(H; kwargs...)
  return H
end

function three_body_MPO(sites, NNN::Int64; pbc=false, Jprime=1.0, presummed=true, J=1.0,  kwargs...)
  N = length(sites)
  H = MPO(N)
  if pbc
    l, r = Index(1, "Link,c=0,l=1"), Index(1, "Link,c=1,l=1")
  else
    l, r = Index(1, "Link,l=0"), Index(1, "Link,l=1")
  end
  ul = lower
  H[1] = one_body_op(sites[1], l, r, ul; kwargs...)

  for n in 2:N
    H[n] = one_body_op(sites[n], l, r, ul; kwargs...)
  end

  W = H[1]
  if Jprime != 0.0
    if presummed
      W = add_ops(W, two_body_sum(sites[1], l, r, NNN, ul; kwargs...))
    else
      for m in 1:NNN
        W = add_ops(W, two_body_op(sites[1], l, r, m, ul; kwargs...))
      end
    end
  end

  if J != 0.0
    for n in 2:NNN
      for m in (n + 1):NNN
        W = add_ops(W, three_body_op(sites[1], l, r, n, m, ul; kwargs...))
      end
    end
  end
  addW!(sites, H, W)
  daisychain_links!(H; pbc=pbc)
  H.llim,H.rlim=-1,1

  return H
end

#
#  d-block = Hx*Sx
#
function one_body_op(site::Index, r1::Index, c1::Index, ul::reg_form; hx=0.0,  kwargs...)::ITensor
  Dw::Int64 = 2
  #d,n,space=parse_site(site)
  #use_qn=hasqns(site)
  r, c = redim(r1, Dw,0), redim(c1, Dw,0)
  is = dag(site) #site seem to have the wrong direction!
  W = ITensor(0.0, r, c, is, dag(is'))
  Id = op(is, "Id")
  assign!(W, Id, r => 1, c => 1)
  assign!(W, Id, r => Dw, c => Dw)
  if hx != 0.0
    Sx = op(is, "Sx") #This will crash if Sz is a good QN
    assign!(W, hx * Sx, r => Dw, c => 1)
  end
  return W
end
#
#  J_1n * Z_1*Z_n
#
function two_body_op(
  site::Index, r1::Index, c1::Index, n::Int64, ul::reg_form; Jprime=1.0, kwargs...
)::ITensor
  @mpoc_assert n >= 1
  Dw::Int64 = 2 + n
  #d,n,space=parse_site(site)
  #use_qn=hasqns(site)
  r, c = redim(r1, Dw,0), redim(c1, Dw,0)

  is = dag(site) #site seem to have the wrong direction!
  W = ITensor(0.0, r, c, is, dag(is'))
  Id = op(is, "Id")
  Sz = op(is, "Sz")
  assign!(W, Id, r => 1, c => 1)
  assign!(W, Id, r => Dw, c => Dw)
  Jn = Jprime / n^4 #interactions need to decay with distance in order for H to extensive 
  if ul == lower
    assign!(W, Sz, r => 2, c => 1)
    for j in 1:(n - 1)
      assign!(W, Id, r => 2 + j, c => 1 + j)
    end
    assign!(W, Jn * Sz, r => Dw, c => Dw - 1)
  else
    assign!(W, Sz, r => 1, c => 2)
    for j in 1:(n - 1)
      assign!(W, Id, r => 1 + j, c => 2 + j)
    end
    assign!(W, Jn * Sz, r => Dw - 1, c => Dw)
  end
  return W
end

function two_body_sum(
  site::Index, r1::Index, c1::Index, NNN::Int64, ul::reg_form;Jprime=1.0, kwargs...
)::ITensor
  @mpoc_assert NNN >= 1
  Dw::Int64 = 2 + NNN
  #d,n,space=parse_site(site)
  #use_qn=hasqns(site)
  r, c = redim(r1, Dw,0), redim(c1, Dw,0)

  is = dag(site) #site seem to have the wrong direction!
  W = ITensor(0.0, r, c, is, dag(is'))
  Id = op(is, "Id")
  Sz = op(is, "Sz")
  assign!(W, Id, r => 1, c => 1)
  assign!(W, Id, r => Dw, c => Dw)
  if ul == lower
    assign!(W, Sz, r => Dw, c => Dw - 1)
  else
    assign!(W, Sz, r => Dw - 1, c => Dw)
  end
  for n in 1:NNN
    Jn = Jprime / n^4 #interactions need to decay with distance in order for H to extensive 
    if ul == lower
      assign!(W, Id, r => Dw - n, c => Dw - 1 - n)
      assign!(W, Jn * Sz, r => Dw - n, c => 1)
    else
      assign!(W, Id, r => Dw - 1 - n, c => Dw - n)
      assign!(W, Jn * Sz, r => 1, c => Dw - n)
    end
  end
  return W
end

#
#  J1n*Jnm * Z_1*Z_n*Z_m
#
function three_body_op(
  site::Index, r1::Index, c1::Index, n::Int64, m::Int64, ul::reg_form; J=1.0, kwargs...
)::ITensor
  @mpoc_assert n >= 1
  @mpoc_assert m > n
  W = two_body_op(site, r1, c1, m - 1, ul)
  #@pprint(W)
  r, c = inds(W; tags="Link")
  Dw = dim(r)
  Sz = op(dag(site), "Sz")
  k = 1
  Jkn = J / (n - k)^2
  Jnm = J / (m - n)^2
  #@show k,n,m,Jkn,Jnm
  if ul == lower
    assign!(W, Sz, r => 2 + n - k, c => 1 + n - k)
    assign!(W, Jkn * Jnm * Sz, r => Dw, c => Dw - 1)
  else
    assign!(W, Sz, r => 1 + n - k, c => 2 + n - k)
    assign!(W, Jkn * Jnm * Sz, r => Dw - 1, c => Dw)
  end
  return W
end

function to_openbc(mpo::MPO)::MPO
  N = length(mpo)
  l, r = get_lr(mpo)
  mpo[1] = l * mpo[1]
  mpo[N] = mpo[N] * r
  @mpoc_assert length(filterinds(inds(mpo[1]); tags="Link")) == 1
  @mpoc_assert length(filterinds(inds(mpo[N]); tags="Link")) == 1
  return mpo
end

function get_lr(mpo::MPO)::Tuple{ITensor,ITensor}
  ul::reg_form = is_regular_form(mpo, lower) ? lower : upper

  N = length(mpo)
  W1 = mpo[1]
  llink = filterinds(inds(W1); tags="l=0")[1]
  l = ITensor(0.0, dag(llink))

  WN = mpo[N]
  rlink = filterinds(inds(WN); tags="l=$N")[1]
  r = ITensor(0.0, dag(rlink))
  if ul == lower
    l[llink => dim(llink)] = 1.0
    r[rlink => 1] = 1.0
  else
    l[llink => 1] = 1.0
    r[rlink => dim(rlink)] = 1.0
  end

  return l, r
end

function fast_GS(H::MPO, sites, nsweeps::Int64=5)::Tuple{Float64,MPS}
  state = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
  psi0 = random_mps(sites, state)
  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, 2, 4, 8, 16, 32)
  setcutoff!(sweeps, 1E-10)
  #setnoise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
  E, psi = dmrg(H, psi0, sweeps; outputlevel=0)
  return E, psi
end


include("hamiltonians_AutoMPO.jl")
