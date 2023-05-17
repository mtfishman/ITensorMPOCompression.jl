#
#  Find the first dim==1 index and remove it, then return a Vector.
#
function vector_o2(T::ITensor)
  @assert order(T) == 2
  i1 = inds(T)[findfirst(d -> d == 1, dims(T))]
  return vector(T * dag(onehot(i1 => 1)))
end

function is_gauge_fixed(Wrf::reg_form_Op; eps=1e-14, b=true, c=true,  kwargs...)::Bool
  Wb = extract_blocks1(Wrf, left; c=c, b=b)
  nr, nc = dims(Wrf)
  if b && nr > 1
    !(norm(b0(Wb)) < eps) && return false
  end
  if c && nc > 1
    !(norm(c0(Wb)) < eps) && return false
  end
  return true
end

function is_gauge_fixed(Hrf::AbstractMPS; kwargs...)::Bool
  for W in Hrf
    !is_gauge_fixed(W; kwargs...) && return false
  end
  return true
end

function gauge_fix!(H::reg_form_MPO;kwargs...)
  if !is_gauge_fixed(H;kwargs)
    tₙ = Vector{Float64}(undef, 1)
    for W in H
      tₙ = gauge_fix!(W, tₙ, left)
      @assert is_regular_form(W)
    end
    #tₙ=Vector{Float64}(undef,1) end of sweep above already returns this.
    for W in reverse(H)
      tₙ = gauge_fix!(W, tₙ, right)
      @assert is_regular_form(W)
    end
  end
end

function gauge_fix!(W::reg_form_Op, tₙ₋₁::Vector{Float64}, lr::orth_type)
  @assert W.ul==lower
  @assert is_regular_form(W)
  
  Wb1 = extract_blocks1(W, lr; Abcd=true, fix_inds=true, swap_bc=true)
  𝕀, 𝐀̂, 𝐛̂, 𝐜̂, 𝐝̂ = Wb1.𝕀, Wb1.𝐀̂, Wb1.𝐛̂, Wb1.𝐜̂, Wb1.𝐝̂ #for readability below.
  nr, nc = dims(W)
  nb, nf = lr == left ? (nr, nc) : (nc, nr)
  #
  #  Make in ITensor with suitable indices from the 𝒕ₙ₋₁ vector.
  #
  if nb > 1
    𝒕ₙ₋₁ = ITensor(tₙ₋₁, dag(backward(𝐛̂,lr)), backward(𝐝̂,lr))
  end
  𝐜̂⎖ = nothing
  #
  #  First two if blocks are special handling for row and column vector at the edges of the MPO
  #
  if nb == 1 #col/row at start of sweep.
    𝒕ₙ = c0(Wb1)
    𝐜̂⎖ = 𝐜̂.W - 𝕀 * 𝒕ₙ
    𝐝̂⎖ = 𝐝̂.W
  elseif nf == 1 ##col/row at the end of the sweep
    𝐝̂⎖ = 𝐝̂.W + 𝒕ₙ₋₁ * 𝐛̂.W
    𝒕ₙ = ITensor(1.0, Index(1), Index(1)) #Not used, but required for the return statement.
  else
    𝒕ₙ = 𝒕ₙ₋₁ * A0(Wb1) + c0(Wb1)
    𝐜̂⎖ = 𝐜̂.W + 𝒕ₙ₋₁ * 𝐀̂.W - 𝒕ₙ * 𝕀
    𝐝̂⎖ = 𝐝̂.W + 𝒕ₙ₋₁ * 𝐛̂.W
  end
  set_𝐝̂_block!(W, 𝐝̂⎖)
  set_𝐛̂𝐜̂_block!(W, 𝐜̂⎖,mirror(lr))
  @assert is_regular_form(W)

  # 𝒕ₙ is always a 1xN tensor so we need to remove that dim==1 index in order for vector(𝒕ₙ) to work.
  return vector_o2(𝒕ₙ)
end

