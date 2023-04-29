using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise

N = 6
model = Model"fqhe_2b_pot"()
model_params = (Vs= [1.0, 1, 0.1], Ly = 6.0, prec = 1e-8)
function initstate(n)
	return (n%3 == 0) ? 2 : 1
end
p = 1
q = 3
conserve_momentum = true


function fermion_momentum_translator(i::Index, n::Integer; N=N)
  ts = tags(i)
  translated_ts = ITensorInfiniteMPS.translatecelltags(ts, n)
  new_i = replacetags(i, ts => translated_ts)
  for j in 1:length(new_i.space)
    ch = new_i.space[j][1][1].val
    mom = new_i.space[j][1][2].val
    new_i.space[j] = Pair(QN(("Nf", ch), ("NfMom", mom + n * N * ch)), new_i.space[j][2])
  end
  return new_i
end

function ITensors.space(::SiteType"FermionK", pos::Int; p=1, q=1, conserve_momentum=true)
  if !conserve_momentum
    return [QN("Nf", -p) => 1, QN("Nf", q - p) => 1]
  else
    return [
      QN(("Nf", -p), ("NfMom", -p * pos)) => 1,
      QN(("Nf", q - p), ("NfMom", (q - p) * pos)) => 1,
    ]
  end
end

function ITensors.op!(Op::ITensor, opname::OpName, ::SiteType"FermionK", s::Index...)
  return ITensors.op!(Op, opname, SiteType("Fermion"), s...)
end


s = infsiteinds(
  "FermionK", N; translator=fermion_momentum_translator, initstate, conserve_momentum, p, q
);

ψ = InfMPS(s, initstate);

Hm = InfiniteMPOMatrix(model, s; model_params...);
for H in Hm
  lx,ly=size(H)
  for ix in 1:lx
    for iy in 1:ly
      M=H[ix,iy]
      if !isempty(M)
        ITensors.checkflux(M)
      end
    end
  end
end

# @show inds(removeqns(Hm[10][2,1]),tags="Link")
# @show inds(removeqns(Hm[9][3,2]),tags="Link")
for N in 0:15
for ix in 1:9
  ixp=ix+1
  ic=commonind(Hm[N-ix][ix+1,ix],Hm[N-ixp][ixp+1,ixp])
  @assert !isnothing(ic)
  # @show removeqns(ic)
end 
end

#for n in 1:6
#@show dims(Hm[n][11,10])
#end
# @show inds(Hm[n][5,4],tags="Link")
# @show inds(Hm[n][4,3],tags="Link")
# @show inds(Hm[n][3,2],tags="Link")
# @show inds(Hm[n][2,1],tags="Link")
#Hprod=Hm[1][6,5]*Hm[2][5,4]*Hm[3][4,3]*Hm[4][3,2]*Hm[5][2,1]*Hm[6][1,1]

#@show inds(Hprod,tags="Link") 
#@show inds(Hprod) 


# Hi=InfiniteMPO(Hm)
# ITensors.checkflux.(Hi)

# Hi1 = InfiniteMPO(model, s, fermion_momentum_translator; model_params...);
# ITensors.checkflux.(Hi)

# Hrf=reg_form_iMPO(Hi)
# @test !is_gauge_fixed(Hrf)
# @test is_regular_form(Hrf)
# @show nsites(Hrf)
# # for W in Hrf
# #   pprint(W)
# #   @show W.ileft
# # end
# #gauge_fix!(Hrf)
# @show Hrf[1].ileft Hrf[N].iright
# #Ht,BondSpectrums = truncate(Hi) #Use default cutoff,C is now diagonal




nothing
