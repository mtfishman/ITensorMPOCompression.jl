#-----------------------------------------------------------------------
#
#  Infinite lattice iMPO with repeating unit cell
#
mutable struct reg_form_iMPO <: AbstractInfiniteMPS
  data::CelledVector{reg_form_Op}
  reverse::Bool
  ul::reg_form
  function reg_form_iMPO(H::InfiniteMPO, ul::reg_form)
    N = length(H)
    data = CelledVector{reg_form_Op}(undef, N)
    for n in eachindex(H)
      il, ir = parse_links(H[n])
      data[n] = reg_form_Op(H[n], il, ir, ul)
    end
    return new(data, false, ul)
  end
  function reg_form_iMPO(
    Ws::CelledVector{reg_form_Op}, reverse::Bool, ul::reg_form
  )
    return new(Ws, reverse, ul)
  end
end

  function ITensorInfiniteMPS.translatecell(
    translator::Function, Wrf::reg_form_Op, n::Integer
  )
    new_inds = ITensorInfiniteMPS.translatecell(translator, inds(Wrf), n)
    W = ITensors.setinds(Wrf.W, new_inds)
    ileft, iright = parse_links(W)
    return reg_form_Op(W, ileft, iright, Wrf.ul)
end
  

  ITensors.data(H::reg_form_iMPO) = H.data
  
  function Ws(H::reg_form_iMPO)
    return map(n -> H[n].W, 1:length(H))
  end
  
  Base.length(H::reg_form_iMPO) = length(H.data)
  function Base.reverse(H::reg_form_iMPO)
    return reg_form_iMPO(Base.reverse(H.data), H.llim, H.rlim, H.reverse, H.ul)
  end
  Base.iterate(H::reg_form_iMPO, args...) = iterate(H.data, args...)
  Base.getindex(H::reg_form_iMPO, n::Integer) = getindex(H.data, n)
  Base.setindex!(H::reg_form_iMPO, W::reg_form_Op, n::Integer) = setindex!(H.data, W, n)
  Base.copy(H::reg_form_iMPO) = reg_form_iMPO(copy(H.data), H.reverse, H.ul)
  
  function reg_form_iMPO(H::InfiniteMPO;honour_upper=false, kwargs...)
    (bl, bu) = detect_regular_form(H;kwargs...)
    if !(bl || bu)
      throw(ErrorException("MPO++(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
      # @pprint(H[1])
      @assert false
    end
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    Hrf=reg_form_iMPO(H, ul)
    if ul==upper && !honour_upper
      Hrf=transpose(Hrf)
    end
    check(Hrf;honour_upper=honour_upper)
    return Hrf
  end
  
  #same as reg_form_MPO version
  function check(Hrf::reg_form_iMPO;honour_upper=false)
    check.(Hrf)
    if !honour_upper
      @mpoc_assert Hrf.ul==lower
    end
  end

  function ITensorInfiniteMPS.InfiniteMPO(Hrf::reg_form_iMPO)::InfiniteMPO
    return InfiniteMPO(Ws(Hrf))
  end
  
  function to_openbc(Hrf::reg_form_iMPO)::reg_form_iMPO
    N = length(Hrf)
    if N > 1
      l, r = get_lr(Hrf)
      Hrf[1].W = l * prime(Hrf[1].W, Hrf[1].ileft)
      Hrf[N].W = prime(Hrf[N].W, Hrf[N].iright) * r
      @mpoc_assert length(inds(Hrf[1].W; tags="Link")) == 1
      @mpoc_assert length(inds(Hrf[N].W; tags="Link")) == 1
    end
    return Hrf
  end
  
  function get_lr(Hrf::reg_form_iMPO)::Tuple{ITensor,ITensor}
    N = length(Hrf)
    llink, rlink = linkinds(Hrf[1])
    l = ITensor(0.0, dag(llink'))
    r = ITensor(0.0, dag(rlink'))
    if Hrf.ul == lower
      l[llink' => dim(llink)] = 1.0
      r[rlink' => 1] = 1.0
    else
      l[llink' => 1] = 1.0
      r[rlink' => dim(rlink)] = 1.0
    end
  
    return l, r
  end
  
  function get_Dw(Hrf::reg_form_iMPO)
    return map(n -> dim(Hrf[n].iright), eachindex(Hrf))
  end
  
  maxlinkdim(Hrf::reg_form_iMPO)=maximum(get_Dw(Hrf))
  maxlinkdim(Hrf::reg_form_MPO)=maximum(get_Dw(Hrf))
  
  
  function check_ortho(H::reg_form_iMPO, lr::orth_type;kwargs...)::Bool
    for n in sweep(H, lr) #skip the edge row/col opertors
      !check_ortho(H[n], lr;kwargs...) && return false
    end
    return true
  end
  
  function is_regular_form(H::reg_form_iMPO;kwargs...)::Bool
    for W in H
      !is_regular_form(W;kwargs...) && return false
    end
    return true
  end

  function Base.transpose(Hrf::reg_form_iMPO)::reg_form_iMPO
    Ws=copy(data(Hrf))
    N=length(Hrf)
    ul1=flip(Hrf.ul)
    for n in 1:N
      ir=Ws[n].iright
      G = G_transpose(ir, reverse(ir))
      Ws[n]*=G
      Ws[n+1]*=dag(G)
      Ws[n]=reg_form_Op(Ws[n].W,Ws[n].ileft,Ws[n].iright,ul1)
      # Ws[n].ul=ul1 setting member data in CelledVector is tricky because of the translator.
    end
    return reg_form_iMPO(Ws, false, ul1)
  end
  