
#-------------------------------------------------------------------------------
#
#  Blocking functions
#               ul=lower         ul=upper
#                1 0 0           1 b d
#                b A 0           0 A c
#                d c I           0 0 I
#
#  Structure to hold all blocks as reg_form_Ops so we don't need to store in indices separately.
#
mutable struct regform_blocks
  𝕀::Union{ITensor}
  𝐀̂::Union{reg_form_Op,Nothing}
  𝐛̂::Union{reg_form_Op,Nothing}
  𝐜̂::Union{reg_form_Op,Nothing}
  𝐝̂::Union{reg_form_Op,Nothing}
  𝐀̂𝐜̂::Union{reg_form_Op,Nothing}
  𝐕̂::Union{reg_form_Op,Nothing}
end

#
#  Transpose inds for upper, no-op for lower
#
function swap_ul(ileft::Index, iright::Index, ul::reg_form)
  return if ul == lower
    (ileft, iright, dim(ileft), dim(iright))
  else
    (iright, ileft, dim(iright), dim(ileft))
  end
end
function swap_ul(Wrf::reg_form_Op)
  return if Wrf.ul == lower
    (Wrf.ileft, Wrf.iright, dim(Wrf.ileft), dim(Wrf.iright))
  else
    (Wrf.iright, Wrf.ileft, dim(Wrf.iright), dim(Wrf.ileft))
  end
end
# lower left or upper right
llur(ul::reg_form, lr::orth_type)::Bool = lr == left && ul == lower || lr == right && ul == upper
llur(W::reg_form_Op, lr::orth_type)::Bool = llur(W.ul, lr)

#  Use recognizably distinct UTF symbols for operators, and op valued vectors and matrices: 
#  𝐀̂ 𝐛̂ 𝐜̂ 𝐝̂ 𝐕̂ 
function extract_blocks(
  Wrf::reg_form_Op,
  lr::orth_type;
  Abcd=false,
  c=false,
  b=false,
  d=false,
  A=false,
  Ac=false,
  V=false,
  fix_inds=false,
  swap_bc=false,
)::regform_blocks
  check(Wrf)
  @assert plev(Wrf.ileft) == 0
  @assert plev(Wrf.iright) == 0
  ir, ic = linkinds(Wrf)
  ul=Wrf.ul
  if ul == upper
    ir, ic = ic, ir #transpose
  end
  nr, nc = dim(ir), dim(ic)
  @assert nr > 1 || nc > 1
  if Abcd || fix_inds #does not include Ac
    A = b = c = d = true
  end
  if !llur(Wrf, lr) && swap_bc #not lower-left or upper-right
    b, c = c, b #swap flags
  end

  A = A && (nr > 1 && nc > 1)
  b = b && nr > 1
  c = c && nc > 1

  𝕀 = nr > 1 ? slice(Wrf.W, ir => 1, ic => 1) : slice(Wrf.W, ir => 1, ic => nc)

  𝐀̂ = A ? Wrf[ir=>2:(nr - 1), ic=>2:(nc - 1)] : nothing
  𝐛̂ = b ? Wrf[ir=>2:(nr - 1), ic=>1:1] : nothing
  𝐜̂ = c ? Wrf[ir=>nr:nr,ic=>2:(nc - 1)] : nothing
  𝐝̂ = d ? (nr > 1 ? Wrf[ir=>nr:nr, ic=>1:1] : Wrf[ir=>1:1,ic=> 1:1]) : nothing

  if Ac
    if llur(Wrf, lr)
      𝐀̂𝐜̂ = nr > 1 ? Wrf[ir=>2:nr, ic=>2:(nc - 1)] : Wrf[ir=>1:1, ic=>2:(nc - 1)]
    else
      𝐀̂𝐜̂ = nc > 1 ? Wrf[ir=>2:(nr - 1), ic=>1:(nc - 1)] : Wrf[ ir=>2:(nr - 1), ic=>1:1]
    end
  else
    𝐀̂𝐜̂ = nothing
  end

  if V
    i1, i2, n1, n2 = swap_ul(Wrf)
    if llur(Wrf, lr) #lower left/upper right
      min1 = min(n1, 2)
      min2 = min(n2, 2)
      𝐕̂ = Wrf[i1 => min1:n1, i2 => min2:n2] #Bottom right corner
    else #lower right/upper left
      max1 = max(n1 - 1, 1)
      max2 = max(n2 - 1, 1)
      𝐕̂ = Wrf[i1 => 1:max1, i2 => 1:max2] #top left corner
    end
  else
    𝐕̂ = nothing
  end
  
  if fix_inds
    if ul==lower
      c && ( 𝐜̂ = replaceind(𝐜̂, 𝐜̂.ileft, 𝐝̂.ileft))
      b && ( 𝐛̂ = replaceind(𝐛̂, 𝐛̂.iright, 𝐝̂.iright))
      A && ( 𝐀̂ = replaceinds(𝐀̂, [𝐀̂.ileft,𝐀̂.iright], [𝐛̂.ileft, 𝐜̂.iright]))
    else
      c && ( 𝐜̂ = replaceind(𝐜̂, 𝐜̂.iright, 𝐝̂.iright))
      b && ( 𝐛̂ = replaceind(𝐛̂, 𝐛̂.ileft, 𝐝̂.ileft))
      A && ( 𝐀̂ = replaceinds(𝐀̂, [𝐀̂.ileft,𝐀̂.iright], [𝐜̂.ileft, 𝐛̂.iright]))
    end
  
  end
  if !llur(Wrf, lr) && swap_bc #not lower-left or upper-right
    𝐛̂, 𝐜̂ = 𝐜̂, 𝐛̂
    # Wb.irb, Wb.irc = Wb.irc, Wb.irb
    # Wb.icb, Wb.icc = Wb.icc, Wb.icb
  end
  return regform_blocks(𝕀,𝐀̂,𝐛̂,𝐜̂,𝐝̂,𝐀̂𝐜̂,𝐕̂)
end

d(Wb::regform_blocks)::Float64 = scalar(Wb.𝕀 * dag(Wb.𝕀))
b0(Wb::regform_blocks)::ITensor = Wb.𝐛̂.W * dag(Wb.𝕀) / d(Wb)
c0(Wb::regform_blocks)::ITensor = Wb.𝐜̂.W * dag(Wb.𝕀) / d(Wb)
A0(Wb::regform_blocks)::ITensor = Wb.𝐀̂.W * dag(Wb.𝕀) / d(Wb)


function set_𝐛̂_block!(Wrf::reg_form_Op, 𝐛̂::ITensor)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  return Wrf.W[i1 => 2:(n1 - 1), i2 => 1:1] = 𝐛̂
end

function set_𝐜̂_block!(Wrf::reg_form_Op, 𝐜̂::ITensor)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  return Wrf.W[i1 => n1:n1, i2 => 2:(n2 - 1)] = 𝐜̂
end

function set_𝐛̂𝐜̂_block!(Wrf::reg_form_Op, 𝐛̂𝐜̂::ITensor, lr::orth_type)
  @mpoc_assert Wrf.ul==lower
  if lr==left
    set_𝐛̂_block!(Wrf, 𝐛̂𝐜̂)
  else
    set_𝐜̂_block!(Wrf, 𝐛̂𝐜̂)
  end
end

function set_𝐝̂_block!(Wrf::reg_form_Op, 𝐝̂::ITensor)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  return Wrf.W[i1 => n1:n1, i2 => 1:1] = 𝐝̂
end


function set_𝐛̂_block!(Wrf::reg_form_Op, 𝐛̂::reg_form_Op)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  return Wrf.W[i1 => 2:(n1 - 1), i2 => 1:1] = 𝐛̂.W
end

function set_𝐜̂_block!(Wrf::reg_form_Op, 𝐜̂::reg_form_Op)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  return Wrf.W[i1 => n1:n1, i2 => 2:(n2 - 1)] = 𝐜̂.W
end

function set_𝐛̂𝐜̂_block!(Wrf::reg_form_Op, Wb::regform_blocks, lr::orth_type)
  @mpoc_assert Wrf.ul==lower
  if lr==left
    set_𝐛̂_block!(Wrf, Wb.𝐛̂)
  else
    set_𝐜̂_block!(Wrf, Wb.𝐜̂)
  end
end

function set_𝐝̂_block!(Wrf::reg_form_Op, 𝐝̂::reg_form_Op)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  return Wrf.W[i1 => n1:n1, i2 => 1:1] = 𝐝̂.W
end



function set_𝕀_block!(Wrf::reg_form_Op, 𝕀::ITensor)
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  n1 > 1 && assign!(Wrf.W, 𝕀, i1 => 1, i2 => 1)
  return n2 > 1 && assign!(Wrf.W, 𝕀, i1 => n1, i2 => n2)
end

function set_𝐀̂𝐜̂_block(Wrf::reg_form_Op, 𝐀̂𝐜̂::ITensor, lr::orth_type)
  @mpoc_assert Wrf.ul==lower
  check(Wrf)
  i1, i2, n1, n2 = swap_ul(Wrf)
  if lr==left #lower left/upper right
    min1 = Base.min(n1, 2)
    Wrf.W[i1 => min1:n1, i2 => 2:(n2 - 1)] = 𝐀̂𝐜̂
  else #lower right/upper left
    max2 = Base.max(n2 - 1, 1)
    Wrf.W[i1 => 2:(n1 - 1), i2 => 1:max2] = 𝐀̂𝐜̂
  end
end
# noop versions for when b/c are empty.  Happens in edge ops of H.
function set_𝐛̂𝐜̂_block!(::reg_form_Op, ::Nothing, ::orth_type) end
function set_𝐛̂_block!(::reg_form_Op, ::Nothing) end
function set_𝐜̂_block!(::reg_form_Op, ::Nothing) end

# 
#  Given R, build R⎖ such that lr=left  R=M*R⎖, lr=right R=R⎖*M
#
function build_R⎖(R::ITensor, iqx::Index, ilf::Index)::Tuple{ITensor,Index}
  @mpoc_assert order(R) == 2
  @mpoc_assert hasinds(R, iqx, ilf)
  @mpoc_assert dim(iqx) == dim(ilf) #make sure RL is square
  @checkflux(R)
  im = Index(space(iqx); tags=tags(iqx), dir=dir(iqx), plev=1) #new common index between M and R⎖
  R⎖ = ITensor(eltype(R),0.0, im, ilf)
  #R⎖+=δ(im,ilf) #set diagonal ... blocksparse + diag is not supported yet.  So we do it manually below.
  Dw = dim(im)
  for j1 in 2:(Dw - 1)
    R⎖[im => j1, ilf => j1] = 1.0 # Fill in the interior diagonal
  end
  #
  #  Copy over the perimeter of RL.
  #
  R⎖[im => Dw:Dw, ilf => 2:Dw] = R[iqx => Dw:Dw, ilf => 2:Dw] #last row
  R⎖[im => 1:Dw, ilf => 1:1] = R[iqx => 1:Dw, ilf => 1:1] #first col
  @checkflux(R⎖)
  #
  #  Fix up index tags and primes.
  #
  im = noprime(settags(im, "Link,m"))
  RL_prime = noprime(replacetags(R⎖, tags(iqx), tags(im); plev=1))

  return RL_prime, dag(im)
end

function my_similar(T::DenseTensor, inds...) 
  return ITensor(eltype(T),inds...)
end

function my_similar(T::BlockSparseTensor, inds...)
  return ITensor(eltype(T),inds...)
end

function my_similar(T::DiagTensor, inds...) 
  return diagITensor(eltype(T),inds...)
end

function my_similar(T::DiagBlockSparseTensor, inds...) 
  return diagITensor(eltype(T),inds...)
end

function my_similar(T::ITensor, inds...)
  return my_similar(tensor(T), inds...)
end

function warn_space(A::ITensor, ig::Index)
  ia, = inds(A; tags=tags(ig), plev=plev(ig))
  @mpoc_assert dim(ia) + 2 == dim(ig)
  if hasqns(A)
    sa, sg = space(ia), space(ig)
    if dir(ia) != dir(ig)
      sa = -sa
    end
    if sa != sg[2:(nblocks(ig) - 1)]
      @warn "Mismatched spaces:"
      @show sa sg[2:(nblocks(ig) - 1)] dir(ia) dir(ig)
      #@assert false
    end
  end
end
#                      |1 0 0|
#  given A, spit out G=|0 A 0| , indices of G are provided.
#                      |0 0 1|
function grow(A::ITensor, ig1::Index, ig2::Index)
  @checkflux(A)
  @mpoc_assert order(A) == 2
  warn_space(A, ig1)
  warn_space(A, ig2)
  Dw1, Dw2 = dim(ig1), dim(ig2)
  G = my_similar(A, ig1, ig2)
  G[ig1 => 1, ig2 => 1] = 1.0
  @checkflux(G)
  G[ig1 => Dw1, ig2 => Dw2] = 1.0
  @checkflux(G)
  G[ig1 => 2:(Dw1 - 1), ig2 => 2:(Dw2 - 1)] = A
  @checkflux(G)
  return G
end
