using ITensors
using ITensorMPOCompression
using Test
using Revise,Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

models=[
    [make_transIsing_MPO,"S=1/2",true],
    [make_transIsing_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1",true],
    [make_Hubbard_AutoMPO,"Electron",false],
    ]

@testset "Gauge fix $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false,true], ul=[lower,upper]
    eps=1e-14
    
    N=10 #5 sites
    NNN=7 #Include 2nd nearest neighbour interactions
    sites = siteinds(model[2],N,conserve_qns=qns)
    Hrf=reg_form_MPO(model[1](sites,NNN;ul=ul))
    pre_fixed=model[3] #Hamiltonian starts gauge fixed
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]

    H=MPO(Hrf)
    psi=randomMPS(sites,state)
    E0=inner(psi',H,psi)
    
    @test is_regular_form(Hrf)
    @test pre_fixed==is_gauge_fixed(Hrf,eps)
    gauge_fix!(Hrf)
    @test is_regular_form(Hrf)
    @test is_gauge_fixed(Hrf,eps)
    He=MPO(Hrf)
    E1=inner(psi',He,psi)
    @test E0 ≈ E1 atol = eps
end

nothing
