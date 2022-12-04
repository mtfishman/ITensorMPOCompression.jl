using ITensors
using ITensorMPOCompression

function make_2D_Heisenberg_AutoMPO(sites,Nx::Int64,Ny::Int64,hz::Float64=0.0,J::Float64=1.0)
    N=length(sites)
    @assert N==Nx*Ny
    ampo = OpSum()
    for j=1:N
        add!(ampo, hz   ,"Sz", j)
    end
    lattice = square_lattice(Nx, Ny; yperiodic = false)

    # Define the Heisenberg spin Hamiltonian on this lattice
    ampo = OpSum()
    for b in lattice
        ampo .+= 0.5*J, "S+", b.s1, "S-", b.s2
        ampo .+= 0.5*J, "S-", b.s1, "S+", b.s2
        ampo .+=     J, "Sz", b.s1, "Sz", b.s2
    end
    return MPO(ampo,sites)
end
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
Nx=10
Ny=6
N=Nx*Ny; #10 sites
sites = siteinds("S=1/2",N);
H=make_2D_Heisenberg_AutoMPO(sites,Nx,Ny)
Dw_auto=get_Dw(H)
ss=truncate!(H) #if you look at the SV spectrum we min(sv)=0.25 so there is nothing small to truncate.
Dw_trunc=get_Dw(H)
if Dw_auto==Dw_trunc
    println("You can't beat autoMPO!!!!")
else
    println("Wa-hoo truncate did something useful :)")
end
