using ITensors
function make_Heisenberg_AutoMPO(sites,NNN::Int64=1)::MPO
    N=length(sites)
    @assert(N>=NNN)
    ampo = OpSum()
    for dj=1:NNN
        f=1.0/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
            add!(ampo, f*0.5,"S+", j, "S-", j+dj)
            add!(ampo, f*0.5,"S-", j, "S+", j+dj)
        end
    end
    return MPO(ampo,sites)
end
function make_transIsing_AutoMPO(sites,NNN::Int64=1)::MPO
    N=length(sites)
    @assert(N>NNN)
    ampo = OpSum()
    for dj=1:NNN
        f=1.0/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
        end
    end
    return MPO(ampo,sites)
end


N=10
sites = siteinds("SpinHalf", N;conserve_qns=true)
H=make_transIsing_AutoMPO(sites,2)
@show inds(H[4])[2] #show any interior link
H=make_Heisenberg_AutoMPO(sites,2)
@show inds(H[4])[2] #show any interior link


