using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise

N = 6
model = Model"fqhe_2b_pot"()
model_params = (Vs= [1.0, 0.0, 1.0, 0.0, 0.1], Ly = 6.0, prec = 1e-8)
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
#ψ = InfMPS(s, initstate);



#H = InfiniteMPOMatrix(model, s; model_params...);
H=InfiniteSum{MPO}(model, s; model_params...);
H1=H[1]
@show get_Dw(H1)
@show inds(H[1][3]) inds(H[2][2])
nothing
#H12=add_ops(H[1][2],H[2][1]) 
# pprint(H1[5])
# truncate!(H1)
# @show get_Dw(H1) inds(H1[5])
# pprint(H1[5])
#@show dense(H[1][4,1])
#Hi=make_AutoiMPO(H[1],2,4)
#@show get_Dw(Hi) 

# os=ITensorInfiniteMPS.unit_cell_terms(model;model_params...)
# ss=[s[1]]
# for i in 2:24
#   push!(ss,s[i])
# end
# @show typeof(os) ss
# H=MPO(os,ss)
# @show get_Dw(H)