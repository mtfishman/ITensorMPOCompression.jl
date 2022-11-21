using ITensors
using ITensorInfiniteMPS

function space_shifted(::Model"ising_extended", q̃sz; conserve_qns=true)
   if conserve_qns
     return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
   else
     return [QN() => 2]
   end
end


nsites = 2
model = Model"ising_extended"()
model_kwargs = (J=1.0, h=1.1, J₂=0.2)
# initstate(n) = "↑"
space_ = fill(space_shifted(model, 1; conserve_qns=false), nsites)
#@show space_
s = infsiteinds("S=1/2", nsites; space=space_)
H = InfiniteSum{MPO}(model, s; model_kwargs...)
@show [inds(H[3][n]) for n in 1:3]
Nothing
#ψ = InfMPS(s, initstate)