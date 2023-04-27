using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output
Ncell = 1; #One site per unit cell
NNN = 5; #Include up to 7th nearest neighbour interactions
initstate(n) = "↑"
sites = infsiteinds("S=1/2", 1; initstate, conserve_szparity=false)
H = transIsing_iMPO(sites, 5);
H0 = copy(H);
orthogonalize!(H; orth=right, rr_cutoff=1e-15);
Dw1 = get_Dw(H)[1];
Gs = orthogonalize!(H; orth=left, rr_cutoff=1e-15);
Dw2 = get_Dw(H)[1]
Dw0 = get_Dw(H0)[1]
@pprint(H[1]) #Shows regular and triangular form are preserved.
@show H.llim H.rlim ortho_lims(H) isortho(H)

println("Starting Dw=$Dw0, After one orthogonalize sweep Dw=$Dw1, after two sweeps Dw=$Dw2")
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
  "Before orthogonalize <ψ|H|ψ>=%1.5f, after orthogonalize <ψ|H|ψ>=%1.5f, error=%1.2e",
  E0,
  Et,
  E0 - Et
)
nothing
