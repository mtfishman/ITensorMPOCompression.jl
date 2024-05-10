using ITensors, ITensorMPS
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

function Heisenberg_2D_AutoMPO(sites, Nx::Int64, Ny::Int64, hz::Float64=0.0, J::Float64=1.0)
  N = length(sites)
  @mpoc_assert N == Nx * Ny
  ampo = OpSum()
  for j in 1:N
    add!(ampo, hz, "Sz", j)
  end
  lattice = square_lattice(Nx, Ny; yperiodic=false)

  # Define the Heisenberg spin Hamiltonian on this lattice
  ampo = OpSum()
  for b in lattice
    ampo .+= 0.5 * J, "S+", b.s1, "S-", b.s2
    ampo .+= 0.5 * J, "S-", b.s1, "S+", b.s2
    ampo .+= J, "Sz", b.s1, "Sz", b.s2
  end
  return MPO(ampo, sites)
end

Nx = 10
Ny = 6
N = Nx * Ny; #10 sites
sites = siteinds("S=1/2", N);
H = Heisenberg_2D_AutoMPO(sites, Nx, Ny)
Dw_auto = get_Dw(H)
bond_spectrum = truncate!(H,left) #if you look at the SV spectrum we min(sv)=0.25 so there is nothing small to truncate.
Dw_trunc = get_Dw(H)
if Dw_auto == Dw_trunc
  println("It is very hard to beat autoMPO!!!!")
  println("  And here is why:")
  println("  For a pseudo 2D MPOs there are no small singular values to truncate")
else
  println("Wa-hoo truncate did something useful :)")
  @show Dw_auto Dw_trunc
end
@show bond_spectrum
nothing
