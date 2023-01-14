

@doc """
make_3body_AutoMPO(sites;kwargs...)

Use `ITensor.autoMPO` to reproduce the 3 body Hamiltonian defined in eq. 34 of the Parker paper. 
The MPO is returned in lower regular form.

# Arguments
- `sites` : Site set defining the lattice of sites.
# Keywords
- `hx::Float64 = 0.0` : External magnetic field in `x` direction.
"""
function make_3body_AutoMPO(sites;kwargs...)
    hx::Float64=get(kwargs,:hx,0.0)

    N=length(sites)
    os = OpSum()
    if hx!=0
        os=make_1body(os,N,hx)
    end
    os=make_2body(os,N;kwargs...)
    os=make_3body(os,N;kwargs...)
    MPO(os,sites;kwargs...)
end

function make_1body(os::OpSum,N::Int64,hx::Float64=0.0)::OpSum
    for n=1:N
        add!(os, hx   ,"Sx", n)
    end
    return os
end

function make_2body(os::OpSum,N::Int64;kwargs...)::OpSum
    Jprime::Float64=get(kwargs,:Jprime,1.0)
    if Jprime!=0.0
        for n=1:N
            for m=n+1:N
                Jnm=Jprime/abs(n-m)^4
                add!(os, Jnm    ,"Sz", n, "Sz", m)
                # if heis
                #     add!(os, Jnm*0.5,"S+", n, "S-", m)
                #     add!(os, Jnm*0.5,"S-", n, "S+", m)
                # end
            end
        end
    end
    return os
end

function make_3body(os::OpSum,N::Int64;kwargs...)::OpSum
    J::Float64=get(kwargs,:J,1.0)
    if J!=0.0
        for k=1:N
            for n=k+1:N
                Jkn=J/abs(n-k)^2
                for m=n+1:N
                    Jnm=J/abs(m-n)^2
                    # if k==1
                    #     @show k,n,m,Jkn,Jnm
                    # end
                    add!(os, Jnm*Jkn    ,"Sz", k, "Sz", n,"Sz",m)
                end
            end
        end
    end
    return os
end


@doc """
make_transIsing_AutoMPO(sites,NNN;kwargs...)

Use `ITensor.autoMPO` to build up a transverse Ising model Hamiltonian with up to `NNN` neighbour 2-body 
interactions.  The interactions are hard coded to decay like `J/(i-j)`. between sites `i` and `j`.
The MPO is returned in lower regular form.
 
# Arguments
- `sites` : Site set defining the lattice of sites.
- `NNN::Int64` : Number of neighbouring 2-body interactions to include in `H`

# Keywords
- `hx::Float64 = 0.0` : External magnetic field in `x` direction.
- `J::Float64 = 1.0` : Nearest neighbour interaction strength. Further neighbours decay like `J/(i-j)`..

"""
function make_transIsing_AutoMPO(sites,NNN::Int64;kwargs...)::MPO
    J::Float64=get(kwargs,:J,1.0)
    hx::Float64=get(kwargs,:hx,0.0)

    do_field = hx!=0.0
    N=length(sites)
    @mpoc_assert(N>NNN)
    ampo = OpSum()
    if do_field
        for j=1:N
            add!(ampo, hx   ,"Sx", j)
        end
    end
    for dj=1:NNN
        f=J/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
        end
    end
    return MPO(ampo,sites)
end
make_2body_AutoMPO(sites,NNN::Int64;kwargs...)=make_transIsing_AutoMPO(sites,NNN;kwargs...)

 

@doc """
make_Heisenberg_AutoMPO(sites,NNN;kwargs...)

Use `ITensor.autoMPO` to build up a Heisenberg model Hamiltonian with up to `NNN` neighbour
2-body interactions.  The interactions are hard coded to decay like `J/(i-j)`. between sites `i` and `j`.
The MPO is returned in lower regular form.

# Arguments
- `sites` : Site set defining the lattice of sites.
- `NNN::Int64` : Number of neighbouring 2-body interactions to include in `H`

# Keywords
- `hz::Float64 = 0.0` : External magnetic field in `z` direction.
- `J::Float64 = 1.0` : Nearest neighbour interaction strength. Further neighbours decay like `J/(i-j)`.

"""


 function make_Heisenberg_AutoMPO(sites,NNN::Int64;kwargs...)::MPO
    hz::Float64=get(kwargs,:hz,0.0)
    J::Float64=get(kwargs,:J,1.0)
    N=length(sites)
    @mpoc_assert(N>=NNN)
    ampo = OpSum()
    for j=1:N
        add!(ampo, hz   ,"Sz", j)
    end
    for dj=1:NNN
        f=J/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
            add!(ampo, f*0.5,"S+", j, "S-", j+dj)
            add!(ampo, f*0.5,"S-", j, "S+", j+dj)
        end
    end
    return MPO(ampo,sites)
end
