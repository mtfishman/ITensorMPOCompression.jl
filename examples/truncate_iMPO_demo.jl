using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
initstate(n) = "↑"
sites = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
H = transIsing_iMPO(sites, 7);
is_lower_regular_form(H) == true
H0 = copy(H)
Ss, spectrum = truncate!(H; rr_cutoff=1e-15, cutoff=1e-15)
Dw1 = get_Dw(H)[1]
Dw0 = get_Dw(H0)[1]
@pprint(H[1]) #Shows regular form is preserve but triangularity is not.
println("Starting Dw=$Dw0, After one truncation sweep Dw=$Dw1")
@show spectrum
#
#  Make random state
#
ψ = InfMPS(sites, initstate)
for n in 1:Ncell
  ψ[n] = randomITensor(inds(ψ[n]))
end
E0 = expect(ψ, InfiniteSum{MPO}(H0, NNN))[1]
Et = expect(ψ, InfiniteSum{MPO}(H, NNN))[1]
@printf(
  "Before truncation <ψ|H|ψ>=%1.5f, after truncation <ψ|H|ψ>=%1.5f, error=%1.2e",
  E0,
  Et,
  E0 - Et
)
nothing
