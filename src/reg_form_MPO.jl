#-----------------------------------------------------------------------
#
#  Finite lattice with open BCs, of regulat form Tensos: l*W1*W2...*WN*r
#  Edge tensors have dim=1 dummy indices (order(W)==4) so generic code can work the same
#  on all tensors.
#
mutable struct reg_form_MPO <: AbstractMPS
    data::Vector{reg_form_Op}
    llim::Int # orthocenter-1
    rlim::Int # orthocenter+1
    d0::ITensor # onehot tensor used to create the l=0 dummy index.
    dN::ITensor # onehot tensor used to create the l=N dummy index.
    ul::reg_form #upper or lower
    
    function reg_form_MPO(
      Ws::Vector{reg_form_Op},
      llim::Int64,
      rlim::Int64,
      d0::ITensor,
      dN::ITensor,
      ul::reg_form,
    )
      return new(Ws, llim, rlim, d0, dN, ul)
    end
  end
  
  function reg_form_MPO(
    H::MPO, ils::Indices, irs::Indices, d0::ITensor, dN::ITensor, ul::reg_form
  )
    N = length(H)
    @assert length(ils) == N
    @assert length(irs) == N
    data = Vector{reg_form_Op}(undef, N)
    for n in eachindex(H)
      data[n] = reg_form_Op(H[n], ils[n], irs[n], ul)
    end
    return reg_form_MPO(data, H.llim, H.rlim, d0, dN, ul)
  end

  function add_edge_links!(H::MPO)
    N = length(H)
    irs = map(n -> linkind(H, n), 1:(N - 1)) #right facing index, which can be thought of as a column index
    ils = dag.(irs) #left facing index, or row index.
  
    ts = trivial_space(irs[1])
    T = eltype(H[1])
    il0 = Index(ts; tags="Link,l=0", dir=dir(dag(irs[1])))
    ilN = Index(ts; tags="Link,l=$N", dir=dir(irs[1]))
    d0 = onehot(T, il0 => 1)
    dN = onehot(T, ilN => 1)
    H[1] *= d0
    H[N] *= dN
    ils, irs = [il0, ils...], [irs..., ilN]
    for n in 1:N
      il, ir, W = ils[n], irs[n], H[n]
      #@show il ir inds(W,tags="Link")
      @assert !hasqns(il) || dir(W, il) == dir(il)
      @assert !hasqns(ir) || dir(W, ir) == dir(ir)
    end
    return ils, irs, d0, dN
  end
  
  function reg_form_MPO(H::MPO;honour_upper=false, kwargs...)
    (bl, bu) = detect_regular_form(H;kwargs...)
    if !(bl || bu)
      throw(ErrorException("MPO++(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
      # @pprint(H[1])
      @assert false
    end
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    ils, irs, d0, dN = add_edge_links!(H)
    Hrf=reg_form_MPO(H, ils, irs, d0, dN, ul)
    # flip to lower regular form by default.
    if ul==upper && !honour_upper
      Hrf=transpose(Hrf)
    end
    check(Hrf)
    return Hrf
  end

  function check(Hrf::reg_form_MPO)
    check.(Hrf)
  end
  
  
  function ITensors.MPO(Hrf::reg_form_MPO)::MPO
    N = length(Hrf)
    H = MPO(Ws(Hrf))
    H[1] *= dag(Hrf.d0)
    H[N] *= dag(Hrf.dN)
    H.llim,H.rlim=Hrf.llim,Hrf.rlim
    return H
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

  
  data(H::reg_form_MPO) = H.data
  Base.eltype(H::reg_form_MPO) = eltype(H[1])

  function Ws(H::reg_form_MPO)
    return map(n -> H[n].W, 1:length(H))
  end
  
  Base.length(H::reg_form_MPO) = length(H.data)
  function Base.reverse(H::reg_form_MPO)
    return reg_form_MPO(Base.reverse(H.data), H.llim, H.rlim, H.d0, H.dN, H.ul)
  end
  Base.iterate(H::reg_form_MPO, args...) = iterate(H.data, args...)
  Base.getindex(H::reg_form_MPO, args...) = getindex(H.data, args...)
  Base.setindex!(H::reg_form_MPO, args...) = setindex!(H.data, args...)
  
  function get_Dw(H::reg_form_MPO)
    return get_Dw(MPO(H))
  end
  
  function is_regular_form(H::reg_form_MPO;kwargs...)::Bool
    for W in H
      !is_regular_form(W;kwargs...) && return false
    end
    return true
  end
  
  @doc """
  check_ortho(H,lr)::Bool

  Test if all sites in an MPO statisfty the condition for `lr` orthogonal (canonical) form.

  # Arguments
  - `H:MPO` : MPO to be characterized.
  - `lr::orth_type` : choose `left` or `right` orthogonality condition to test for.

  # Keywrds
  - `eps::Float64 = 1e-14` : operators inside H with norm(W[i,j])<eps are assumed to be zero.

  Returns `true` if the MPO is in `lr` orthogonal (canonical) form.  This is an expensive operation which scales as N*Dw^3 which should mostly be used only for unit testing code or in debug mode.  In production code use isortho which looks at the cached ortho state.

  """
  check_ortho(H::MPO, lr::orth_type;kwargs...)=check_ortho(reg_form_MPO(copy(H)), lr;kwargs...)
  
  function check_ortho(H::reg_form_MPO, lr::orth_type;kwargs...)::Bool
    for n in sweep(H, lr) #skip the edge row/col opertors
      !check_ortho(H[n], lr;kwargs...) && return false
    end
    return true
  end

  function Base.transpose(Hrf::reg_form_MPO)::reg_form_MPO
    Ws=copy(data(Hrf))
    N=length(Hrf)
    ul1=mirror(Hrf.ul)
    for n in 1:N-1
      ir=Ws[n].iright
      G = G_transpose(ir, reverse(ir))
      Ws[n]*=G
      Ws[n+1]*=dag(G)
      Ws[n].ul=ul1
    end
    Ws[N].ul=ul1
    return reg_form_MPO(Ws,Hrf.llim, Hrf.rlim,Hrf. d0, Hrf.dN, ul1)
  end
  
  