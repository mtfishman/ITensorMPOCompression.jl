using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output
initstate(n) = "↑"
Ncell=1; #One site per unit cell
NNN=7; #Include up to 7th nearest neighbour interactions
svd_cutoff=1e-15
sites = infsiteinds("S=1/2", Ncell;initstate, conserve_szparity=false)
H=make_transIsing_iMPO(sites,NNN);
is_lower_regular_form(H)==true
H0=copy(H)
Gs,spectrum=truncate!(H;cutoff=svd_cutoff)
Dw1=get_Dw(H)[1]
Dw0=get_Dw(H0)[1]
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
E0=expect(ψ,InfiniteSum{MPO}(H0,NNN))[1]
Et=expect(ψ,InfiniteSum{MPO}(H ,NNN))[1]
@printf("Before truncation <ψ|H|ψ>=%1.5f, after truncation <ψ|H|ψ>=%1.5f, error=%1.2e",E0,Et,E0-Et)
nothing
