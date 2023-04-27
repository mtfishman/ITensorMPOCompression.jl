using ITensors
using ITensors.NDTensors
using ITensorMPOCompression
using Printf
using Test
using Revise

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

#-----------------------------------------------------------------------
#
#  Test helper Use loops over ITensor indices to perform and subtensor extraction
#
function get_subtensor_I(T::ITensor, irs::IndexRange...)
  is = indices(irs)
  iothers = Tuple(noncommoninds(T, is))
  iso = is..., iothers...
  p = getperm(inds(T), ntuple(n -> iso[n], length(iso)))
  is_sub = redim(irs) #get re-dimensied Indices
  iso_subp = permute((is_sub..., iothers...), p)

  T_sub = ITensor(eltype(T), iso_subp)
  for (i1, i) in zip(eachindval(is_sub), eachindval(irs))
    assign!(T_sub, slice(T, i...), i1...)
  end
  return T_sub
end

@testset "subtensor with dense storage" begin
  # setup a random 3 index ITensor
  i = Index(5, "i")
  j = Index(6, "j")
  k = Index(2, "k")
  T = randomITensor(k, j, i)
  #
  # Choose ranges for two of the indices.
  #
  ir = IndexRange(i, 2:4)
  jr = IndexRange(j, 3:5)
  #
  #  Extract subvector using two different methods.  Use a mixture of pair in IndexRange.
  #
  Ts1 = T[i => 2:4, jr] #production version NDTensors. 
  Ts2 = get_subtensor_I(T, ir, jr) #development version does everying at the ITensors level
  # Ts1 and Ts2 have different i,j index IDs so we can't subtract them at the ITensor level.
  @test norm(tensor(Ts2) - tensor(Ts1)) == 0.0 #should be the same index permuations are correct.
  #
  #  Now swap the order of the ranges, should have no effect
  #
  Ts3 = T[j => range(jr), i => range(ir)]
  Ts4 = get_subtensor_I(T, jr, ir)
  @test norm(tensor(Ts3) - tensor(Ts4)) == 0.0
  @test norm(tensor(Ts1) - tensor(Ts3)) == 0.0
  #
  #  Test assignment back into T
  #
  Ts1 .*= 2.121 #change the data
  T[i => range(ir), jr] = Ts1 #bulk assignment using mixed pair in IndexRange arguments.
  is1 = inds(Ts1; tags="i")[1] #find the new inds of Ts1 
  js1 = inds(Ts1; tags="j")[1] #k should be the same
  for iv in eachindval(is1)
    for jv in eachindval(js1)
      for kv in eachindval(k)
        @test Ts1[iv, jv, kv] == T[
          i => (val(iv) + start(ir) - 1), j => (val(jv) + start(jr) - 1), k => val(kv)
        ]
      end
    end
  end

  #
  #  assign a value
  #
  T[i => range(ir), jr] = 3.14
  Ts1 = T[i => range(ir), jr]
  @test norm(tensor(Ts1) .- 3.14) == 0
end

models = [
  [transIsing_MPO, "S=1/2", true],
  [transIsing_AutoMPO, "S=1/2", true],
  [Heisenberg_AutoMPO, "S=1/2", true],
  [Heisenberg_AutoMPO, "S=1", true],
  [Hubbard_AutoMPO, "Electron", false],
]

@testset "subtensor with block sparse storage$(model[1]), qns=$qns, ul=$ul" for model in
                                                                                models,
  qns in [true],
  ul in [lower, upper]

  N, NNN = 5, 2
  sites = siteinds(model[2], N; conserve_qns=qns)
  H = model[1](sites, NNN)
  W = H[3]

  il = inds(W; tags="Link")
  Dw = dim(il[1])
  #
  #  extract sub tensors
  #
  i1, i2 = IndexRange(il[1], 1:(Dw - 1)), IndexRange(il[2], 1:(Dw - 1))
  V = W[il[1] => 1:(Dw - 1), i2]
  V1 = get_subtensor_I(W, i1, i2)
  @test norm(tensor(dense(V)) - tensor(dense(V1))) == 0.0

  i1, i2 = IndexRange(il[1], 2:Dw), IndexRange(il[2], 2:Dw)
  V = W[i1, i2]
  V1 = get_subtensor_I(W, i1, i2)
  @test norm(tensor(dense(V)) - tensor(dense(V1))) == 0.0
  #
  #  Assign sub tensor
  #
  V .*= 2.5345
  W[i1, i2] = V

  iv1, iv2 = inds(V; tags="Link")
  for iiv1 in eachindval(iv1)
    for iiv2 in eachindval(iv2)
      opV = slice(V, iiv1, iiv2)
      opW = slice(
        W, il[1] => (val(iiv1) + start(i1) - 1), il[2] => (val(iiv2) + start(i2) - 1)
      )
      @test opV == opW
    end
  end

  #make another version of W, but with indexes re-ordered
  iW = inds(W)
  iW1 = (iW[3], iW[1], iW[4], iW[2]) #interleave blocked and un-blocked indices.
  W1 = ITensor(QN("Sz", 0), iW1)
  V = W1[i1, i2]
  V1 = get_subtensor_I(W1, i1, i2)
  @test norm(tensor(dense(V)) - tensor(dense(V1))) == 0.0
end
nothing
