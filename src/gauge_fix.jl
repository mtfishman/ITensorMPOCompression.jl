using SparseArrays
#
#  Find the first dim==1 index and remove it, then return a Vector.
#
function vector_o2(T::ITensor)
  @assert order(T) == 2
  i1 = inds(T)[findfirst(d -> d == 1, dims(T))]
  return vector(T * dag(onehot(i1 => 1)))
end

function is_gauge_fixed(Wrf::reg_form_Op; eps=1e-14, b=true, c=true,  kwargs...)::Bool
  Wb = extract_blocks(Wrf, left; c=c, b=b)
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
  Wb = extract_blocks(W, lr; all=true, fix_inds=true)
  𝕀, 𝐀̂, 𝐛̂, 𝐜̂, 𝐝̂ = Wb.𝕀, Wb.𝐀̂, Wb.𝐛̂, Wb.𝐜̂, Wb.𝐝̂ #for readability below.
  nr, nc = dims(W)
  nb, nf = lr == left ? (nr, nc) : (nc, nr)
  #
  #  Make in ITensor with suitable indices from the 𝒕ₙ₋₁ vector.
  #
  if nb > 1
    ibd, ibb =lr==left ? (Wb.ird, Wb.irb) : (Wb.icd, Wb.icb)
    𝒕ₙ₋₁ = ITensor(tₙ₋₁, dag(ibb), ibd)
  end
  𝐜̂⎖ = nothing
  #
  #  First two if blocks are special handling for row and column vector at the edges of the MPO
  #
  if nb == 1 #col/row at start of sweep.
    𝒕ₙ = c0(Wb)
    𝐜̂⎖ = 𝐜̂ - 𝕀 * 𝒕ₙ
    𝐝̂⎖ = 𝐝̂
  elseif nf == 1 ##col/row at the end of the sweep
    𝐝̂⎖ = 𝐝̂ + 𝒕ₙ₋₁ * 𝐛̂
    𝒕ₙ = ITensor(1.0, Index(1), Index(1)) #Not used, but required for the return statement.
  else
    𝒕ₙ = 𝒕ₙ₋₁ * A0(Wb) + c0(Wb)
    𝐜̂⎖ = 𝐜̂ + 𝒕ₙ₋₁ * 𝐀̂ - 𝒕ₙ * 𝕀
    𝐝̂⎖ = 𝐝̂ + 𝒕ₙ₋₁ * 𝐛̂
  end
  set_𝐝̂_block!(W, 𝐝̂⎖)
  set_𝐛̂𝐜̂_block!(W, 𝐜̂⎖,mirror(lr))
  @assert is_regular_form(W)

  # 𝒕ₙ is always a 1xN tensor so we need to remove that dim==1 index in order for vector(𝒕ₙ) to work.
  return vector_o2(𝒕ₙ)
end

#-----------------------------------------------------------------------
#
#  Infinite lattice with unit cell
#
function gauge_fix!(H::reg_form_iMPO;kwargs...)
  @mpoc_assert H.ul==lower
  if !is_gauge_fixed(H;kwargs...)
    sₙ, tₙ = Solve_b0c0(H)
    for n in eachindex(H)
      gauge_fix!(H[n], sₙ[n - 1], sₙ[n], tₙ[n - 1], tₙ[n])
    end
  end
end

function ITensorInfiniteMPS.translatecell(::Function, T::Float64, ::Integer)
  return T
end

function Solve_b0c0(Hrf::reg_form_iMPO)
  A0s = Vector{Matrix}()
  b0s = Vector{Float64}()
  c0s = Vector{Float64}()
  nr, nc = 0, 0
  irb, icb = Vector{Int64}(), Vector{Int64}()
  ir, ic = 1, 1
  for W in Hrf
    check(W)
    Wb = extract_blocks(W, left; all=true)
    A_0 = matrix(Wb.irA, A0(Wb), Wb.icA)
    push!(A0s, A_0)
    append!(b0s, vector_o2(b0(Wb)))
    append!(c0s, vector_o2(c0(Wb)))
    push!(irb, ir)
    push!(icb, ic)
    nr += size(A_0, 1)
    nc += size(A_0, 2)
    ir += size(A_0, 1)
    ic += size(A_0, 2)
  end
  @assert nr == nc
  n = nr
  N = length(A0s)
  Ms, Mt = spzeros(n, n), spzeros(n, n)
  for n in eachindex(A0s)
    nr, nc = size(A0s[n])
    ir, ic = irb[n], icb[n]
    #
    #  These system will generally not bee so big that sparse improves performance significantly.
    #
    sparseA0 = sparse(A0s[n])
    droptol!(sparseA0, 1e-15)
    Id=sparse(LinearAlgebra.I, nr, nc)
    Ms[ir:(ir + nr - 1), ic:(ic + nc - 1)] = sparseA0
    Mt[ir:(ir + nr - 1), ic:(ic + nc - 1)] = Id
    if n == 1
      Ms[ir:(ir + nr - 1), icb[N]:(icb[N] + nc - 1)] -= Id
      Mt[ir:(ir + nr - 1), icb[N]:(icb[N] + nc - 1)] -= sparseA0
    else
      Ms[ir:(ir + nr - 1), icb[n - 1]:(ic - 1)] -= Id
      Mt[ir:(ir + nr - 1), icb[n - 1]:(ic - 1)] -= sparseA0
    end
  end
  s = Ms \ b0s
  t = Array(Base.transpose(Base.transpose(Mt) \ c0s))
  @assert norm(Ms * s - b0s) < 1e-15 * n
  @assert norm(Base.transpose(t * Mt) - c0s) < 1e-15 * n
  @assert size(t,1)==1 #t ends up as a 1xN matrix becuase of all the transposing.
  ss = map(n -> s[irb[n]:(irb[n] + nr - 1)], 1:N)
  ts = map(n -> t[1,irb[n]:(irb[n] + nr - 1)], 1:N)
  return CelledVector(ss), CelledVector(ts)
end

function gauge_fix!(
  W::reg_form_Op,
  sₙ₋₁::Vector{Float64},
  sₙ::Vector{Float64},
  tₙ::Vector{Float64},
  tₙ₋₁::Vector{Float64},
)
  @assert is_regular_form(W)
  Wb = extract_blocks(W, left; all=true, fix_inds=true)
  𝕀, 𝐀̂, 𝐛̂, 𝐜̂, 𝐝̂ = Wb.𝕀, Wb.𝐀̂, Wb.𝐛̂, Wb.𝐜̂, Wb.𝐝̂ #for readability below.

  𝒕ₙ₋₁ = ITensor(tₙ₋₁, dag(Wb.irb), Wb.ird)
  𝒕ₙ = ITensor(tₙ, Wb.irc, Wb.icc)
  𝒔ₙ₋₁ = ITensor(sₙ₋₁, Wb.irb, Wb.icb)
  𝒔ₙ = ITensor(sₙ, Wb.icb, dag(Wb.icA))
  𝐛̂⎖ = 𝐛̂ + 𝒔ₙ₋₁ * 𝕀 - 𝐀̂ * 𝒔ₙ
  𝐜̂⎖ = 𝐜̂ - 𝒕ₙ * 𝕀 + 𝒕ₙ₋₁ * 𝐀̂
  𝐝̂⎖ = 𝐝̂ + 𝒕ₙ₋₁ * 𝐛̂ - 𝒔ₙ * 𝐜̂⎖

  set_𝐛̂_block!(W, 𝐛̂⎖)
  set_𝐜̂_block!(W, 𝐜̂⎖)
  set_𝐝̂_block!(W, 𝐝̂⎖)
  return check(W)
end
