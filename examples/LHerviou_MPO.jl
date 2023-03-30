using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise

#
# InfiniteMPO has dangling links at the end of the chain.  We contract these on the outside
#   with l,r terminating vectors, to make a finite lattice MPO.
#
function terminate(h::InfiniteMPO)::MPO
  Ncell=nsites(h)
  # left termination vector
  il1=commonind(h[1],h[2])
  il0,=noncommoninds(h[1],il1,tags="Link")
  l=ITensor(0.0,il0)
  l[il0=>dim(il0)]=1.0 #assuming lower reg form in h
  # right termination vector
  iln=commonind(h[Ncell-1],h[Ncell])
  ilnp,=noncommoninds(h[Ncell],iln,tags="Link")
  r=ITensor(0.0,ilnp)
  r[ilnp=>1]=1.0 #assuming lower reg form in h
  # build up a finite MPO
  hf=MPO(Ncell)
  hf[1]=dag(l)*h[1] #left terminate
  hf[Ncell]=h[Ncell]*dag(r) #right terminate
  for n in 2:Ncell-1
      hf[n]=h[n] #fill in the bulk.
  end
  return hf
end
#
# Terminate and then call expect
# for inf ψ and finite h, which is already supported in src/infinitecanonicalmps.jl
#
function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteMPO)
  return expect(ψ,terminate(h)) #defer to src/infinitecanonicalmps.jl
end

N = 6
model = Model"fqhe_2b_pot"()
model_params = (Vs= [1.0, 0.0, 1.0, 0.0, 0.1], Ly = 6.0, prec = 1e-12)
function initstate(n)
	mod1(n, 3) == 1 && return 2
	return 1
end
p = 1
q = 3
conserve_momentum = true

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

function fermion_momentum_translator(i::Index, n::Integer; N=N)
      ts = tags(i)
      translated_ts = ITensorInfiniteMPS.translatecelltags(ts, n)
      new_i = replacetags(i, ts => translated_ts)
      for j in 1:length(new_i.space)
        ch = new_i.space[j][1][1].val
        mom = new_i.space[j][1][2].val
        new_i.space[j] = Pair(
          QN(("Nf", ch), ("NfMom", mom + n * N * ch)), new_i.space[j][2]
        )
      end
      return new_i
    end

function ITensors.op!(Op::ITensor, opname::OpName, ::SiteType"FermionK", s::Index...)
  return ITensors.op!(Op, opname, SiteType("Fermion"), s...)
end

s = infsiteinds("FermionK", N; translator = fermion_momentum_translator, initstate, conserve_momentum, p, q);
ψ = InfMPS(s, initstate);



#H = InfiniteMPOMatrix(model, s; model_params...);
Hs=InfiniteSum{MPO}(model, s; model_params...);
Hi=InfiniteMPO(model, s, fermion_momentum_translator; model_params...);
@show dims(Hs[1][1]) dims(Hi[1])
#@show  inds(Hi[1])
#@pprint Hi[1]
Hid=InfiniteMPO(6)
for n in 1:6
  Hid[n]=dense(Hi[n])
end
#orthogonalize(Hid;rr_cutoff=1e-12,verbose=true,rr_verbose=true)
@show get_Dw(Hid)
Ss,ss=truncate!(Hid;rr_cutoff=1e-14,cutoff=1e-17,verbose=true,rr_verbose=true)
@show get_Dw(Hid) ss
#@pprint Hid[1]
Es=expect(ψ,Hs)
Ei=expect(ψ,Hi)
@show Es Ei
nothing
