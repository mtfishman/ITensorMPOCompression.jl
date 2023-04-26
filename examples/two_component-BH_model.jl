using ITensors
function two_component_BH(N::Int64;U=1.0,U12=-0.5,t1=0.5,t2=0.25,kwargs...)
    sites = siteinds("Boson",N;conserve_qns=true,conserve_number=false)
    op = OpSum()
    for i in 1:2:N-3
        op += -U/2, "N", i
        op += U/2, "N", i,"N",i
        op += -t1, "a", i, "a†", i + 2
        op += -t1, "a†", i, "a", i + 2
    end
    op += -U/2, "N", N-1
    op += U/2, "N", N-1,"N",N-1

    for i in 2:2:N-2
        op += -U/2, "N", i
        op += U/2, "N", i,"N",i
        op += -t2, "a", i, "a†", i + 2
        op += -t2, "a†", i, "a", i + 2
    end
    op += -U/2, "N", N
    op += U/2, "N", N,"N",N

    for i in 1:2:N-1
        op +=U12, "N", i,"N",i+1
    end
    MPO(op,sites;kwargs...)
end

two_component_BH(6)

