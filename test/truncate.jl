using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf
using Profile

#using ITensorMPOCompression: orthogonalize!,truncate!
using NDTensors: Diag, BlockSparse, tensor
#brute force method to control the default float display format.
Base.show(io::IO, f::Float64) = @printf(io, "%1.1e", f)

#
#  We need consistent output from randomMPS in order to avoid flakey unit testset
#  So we just pick any seed so that randomMPS (hopefull) gives us the same 
#  pseudo-random output for each test run.
#
using Random
Random.Random.seed!(12345);
ITensors.ITensors.disable_debug_checks()

verbose = false #verbose at the outer test level
verbose1 = false #verbose inside orth algos

@testset verbose = verbose "Truncate/Compress" begin
  models = [
    [make_transIsing_MPO, "S=1/2", true],
    [make_transIsing_AutoMPO, "S=1/2", true],
    [make_Heisenberg_AutoMPO, "S=1/2", true],
    [make_Heisenberg_AutoMPO, "S=1", true],
    [make_Hubbard_AutoMPO, "Electron", false],
  ]

  @testset "Truncate/Compress MPO $(model[1]), qns=$qns, ul=$ul, lr=$lr" for model in
                                                                             models,
    qns in [false, true],
    ul in [lower, upper],
    lr in [left, right]

    eps = 1e-14
    pre_fixed = model[3] #Hamiltonian starts gauge fixed
    N = 10 #5 sites
    NNN = 4 #Include 6nd nearest neighbour interactions
    sites = siteinds(model[2], N; conserve_qns=qns)
    Hrf = reg_form_MPO(model[1](sites, NNN; ul=ul))
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state)
    E0 = inner(psi', MPO(Hrf), psi)

    bs = truncate!(Hrf, lr)
    @test is_regular_form(Hrf)
    @test check_ortho(Hrf, lr)
    @test is_gauge_fixed(Hrf) #Now everything should be fixed, unless NNN is big
    #
    #  Expectation value check.
    #
    E1 = inner(psi', MPO(Hrf), psi)
    @test E0 ≈ E1 atol = eps
  end

 
  models = [
    (make_transIsing_iMPO, "S=1/2"),
    (make_transIsing_AutoiMPO, "S=1/2"),
    (make_Heisenberg_AutoiMPO, "S=1/2"),
    (make_Heisenberg_AutoiMPO, "S=1"),
    (make_Hubbard_AutoiMPO, "Electron"),
  ]

  @testset "Truncate/Compress iMPO Check gauge relations, H=$(model[1]), ul=$ul, qbs=$qns, N=$N, NNN=$NNN" for model in
                                                                                                               models,
    ul in [lower,upper],
    qns in [false, true],
    N in [1, 2, 3, 4],
    NNN in [1, 4]

    initstate(n) = "↑"
    makeH = model[1]
    site_type = model[2]
    eps = qns ? 1e-14 * NNN : 3e-14 * NNN #dense and larger NNN both get more roundoff noise.
    si = infsiteinds(site_type, N; initstate, conserve_qns=qns)
    H0 = reg_form_iMPO(model[1](si, NNN; ul=ul))
    @test is_regular_form(H0)
    Dw0 = Base.max(get_Dw(H0)...)
    #
    #  Do truncate outputting left ortho Hamiltonian
    #
    HL = copy(H0)
    Ss, ss, HR = truncate!(HL, left; verbose=verbose1)
    #@show Ss ss
    @test typeof(storage(Ss[1])) == (
      if qns
        NDTensors.DiagBlockSparse{Float64,Vector{Float64},2}
      else
        Diag{Float64,Vector{Float64}}
      end
    )

    DwL = Base.max(get_Dw(HL)...)
    @test is_regular_form(HL)
    @test check_ortho(HL, left)
    @test check_ortho(HR, right)
    #
    #  Now test guage relations using the diagonal singular value matrices
    #  as the gauge transforms.
    #
    for n in 1:N
      # @show inds(Ss[n-1]) inds(HR[n].W,tags="Link") inds(Ss[n]) inds(HL[n].W,tags="Link") 
      D1 = Ss[n - 1] * HR[n].W
      @assert order(D1) == 4
      D2 = HL[n].W * Ss[n]
      @assert order(D2) == 4
      @test norm(Ss[n - 1] * HR[n].W - HL[n].W * Ss[n]) ≈ 0.0 atol = eps
    end
    #
    #  Do truncate from H0 outputting right ortho Hamiltonian
    #
    HR = copy(H0)
    Ss, ss, HL = truncate!(HR, right; verbose=verbose1)
    @test typeof(storage(Ss[1])) == (
      if qns
        NDTensors.DiagBlockSparse{Float64,Vector{Float64},2}
      else
        Diag{Float64,Vector{Float64}}
      end
    )
    DwR = Base.max(get_Dw(HR)...)
    @test is_regular_form(HR)
    @test check_ortho(HL, left)
    @test check_ortho(HR, right)
    for n in 1:N
      @test norm(Ss[n - 1] * HR[n].W - HL[n].W * Ss[n]) ≈ 0.0 atol = eps
    end
    if verbose
      @printf " %4i %4i   %4i   %4i  %4i \n" N NNN Dw0 DwL DwR
    end
  end

  # @testset "Test ground states" for qns in [false,true]
  #     eps=3e-13
  #     N=10
  #     NNN=5
  #     hx= qns ? 0.0 : 0.5 

  #     sites = siteinds("SpinHalf", N;conserve_qns=qns)
  #     H=make_transIsing_MPO(sites,NNN;hx=hx)

  #     E0,psi0=fast_GS(H,sites)
  #     truncate!(H;verbose=verbose1,orth=left)

  #     E1,psi1=fast_GS(H,sites)

  #     overlap=abs(inner(psi0,psi1))
  #     RE=abs((E0-E1)/E0)
  #     if verbose
  #         @printf "Trans. Ising E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
  #     end
  #     @test E0 ≈ E1 atol = eps
  #     @test overlap ≈ 1.0 atol = eps

  #     hx=0.0
  #     H=make_Heisenberg_AutoMPO(sites,NNN;hx=hx)
  #     E0,psi0=fast_GS(H,sites)

  #     truncate!(H;verbose=verbose1,orth=right)

  #     E1,psi1=fast_GS(H,sites)
  #     overlap=abs(inner(psi0,psi1))
  #     RE=abs((E0-E1)/E0)
  #     if verbose
  #         @printf "Heisenberg   E0/N=%1.15f E1/N=%1.15f rel. error=%.1e overlap-1.0=%.1e \n" E0/(N-1) E1/(N-1) RE overlap-1.0
  #     end
  #     @test E0 ≈ E1 atol = eps
  #     @test overlap ≈ 1.0 atol = eps
  # end 

  # @testset "Look at bond singular values for large lattices" begin
  #     NNN=5

  #     if verbose
  #         @printf "           |------ max(sv) ----|  min(sv)\n"
  #         @printf "  N    Dw  left     mid    right   mid\n"
  #     end
  #     for N in [6,8,10,12,18,20,40,80]
  #         sites = siteinds("SpinHalf", N)
  #         H=make_transIsing_MPO(sites,NNN;hx=0.5)
  #         specs=truncate!(H;verbose=verbose1,cutoff=1e-10)
  #         imid::Int64=div(N,2)
  #         max_1  =specs[1   ].eigs[1]
  #         max_mid=specs[imid].eigs[1]
  #         max_N  =specs[N-1 ].eigs[1]
  #         min_mid=specs[imid].eigs[end]
  #         Dw=maximum(get_Dw(H))
  #         if verbose
  #             @printf "%4i %4i %1.5f %1.5f %1.5f %1.5f\n" N Dw  max_1 max_mid max_N min_mid
  #         end
  #         @test (max_1-max_N)<1e-10
  #         @test max_mid<1.0
  #     end
  # end

  # @testset "Try a lattice with alternating S=1/2 and S=1 sites. MPO. Qns=$qns" for qns in [false,true]
  #     N=10
  #     NNN=4
  #     eps=1e-14
  #     sites = siteinds(n->isodd(n) ? "S=1/2" : "S=1",N; conserve_qns=qns)
  #     Ha=make_transIsing_AutoMPO(sites,NNN)
  #     H=make_transIsing_MPO(sites,NNN)
  #     #@show get_Dw(H)
  #     state=[isodd(n) ? "Up" : "Dn" for n=1:N]
  #     psi=randomMPS(sites,state)
  #     Ea=inner(psi',Ha,psi)
  #     E0=inner(psi',H,psi)
  #     @test E0 ≈ Ea atol = eps
  #     Ho=copy(H)
  #     orthogonalize!(Ho;verbose=verbose1,rr_cutoff=1e-12)
  #     #@show get_Dw(Ho)
  #     Eo=inner(psi',Ho,psi)
  #     @test E0 ≈ Eo atol = eps
  #     Ht=copy(H)
  #     ss=truncate!(Ht)
  #     #@show get_Dw(Ht)
  #     #@show ss
  #     Et=inner(psi',Ht,psi)
  #     #@show E0 Ea Eo Et E0-Eo E0-Et
  #     @test E0 ≈ Et atol = eps
  #     #
  #     #  Run some GS calculations
  #     #
  #     #E0,psi0=fast_GS(H,sites) unreliable
  #     # All of these use and emperically determined number of sweeps to get full convergence
  #     # This model can easily get stuck in a false minimum. 
  #     Ea,psia=fast_GS(Ha,sites,6) 
  #     Eo,psio=fast_GS(Ho,sites,8)
  #     Et,psit=fast_GS(Ht,sites,7)
  #     #@show Ea Eo Et Ea-Eo Ea-Et
  #     E2a=inner(Ha,psia,Ha,psia)
  #     E2o=inner(Ho,psio,Ho,psio)
  #     E2t=inner(Ht,psit,Ht,psit)
  #     #@show E2a-Ea^2 E2o-Eo^2 E2t-Et^2
  #     @test E2a-Ea^2 ≈ 0.0 atol = eps
  #     @test E2o-Eo^2 ≈ 0.0 atol = eps
  #     @test E2t-Et^2 ≈ 0.0 atol = eps
  #     @test Ea ≈ Eo atol = eps
  #     @test Ea ≈ Et atol = eps

  # end

 
  # @testset "Try a lattice with alternating S=1/2 and S=1 sites. iMPO. Qns=$qns" for qns in [false,true]
  #     initstate(n) = isodd(n) ? "Dn" : "Up"
  #     ul=lower
  #     if verbose
  #         @printf "               Dw     Dw    Dw   \n"
  #         @printf " Ncell  NNN  uncomp. left  right \n"
  #     end
  #     #
  #     #  This one is tricky to set for QNs=true up due to QN flux constraints.
  #     #  An Ncell=3 with S={1/2,1,1/2} seems to work. 
  #     #
  #     for  NNN in [2,4,6], N in [3] #3 site unit cell fails inside ITensorInfiniteMPS for qns=true.
  #         si = infsiteinds(n->isodd(n) ? "S=1" : "S=1/2",N; initstate, conserve_qns=qns)
  #         H0=make_transIsing_iMPO(si,NNN;ul=ul)
  #         @test is_regular_form(H0)
  #         Dw0=Base.max(get_Dw(H0)...)
  #         #
  #         #  Do truncate outputting left ortho Hamiltonian
  #         #
  #         HL=copy(H0)
  #         Ss,ss,HR=truncate!(HL;verbose=verbose1,orth=left)
  #         #@test typeof(storage(Ss[1])) == (qns ? BlockSparse{Float64, Vector{Float64}, 2} : Diag{Float64, Vector{Float64}})
  #         DwL=Base.max(get_Dw(HL)...)
  #         @test is_regular_form(HL)
  #         @test isortho(HL,left)
  #         @test check_ortho(HL,left)
  #         #
  #         #  Now test guage relations using the diagonal singular value matrices
  #         #  as the gauge transforms.
  #         #
  #         for n in 1:N
  #             @test norm(Ss[n-1]*HR[n]-HL[n]*Ss[n]) ≈ 0.0 atol = 1e-14
  #         end    
  #         #
  #         #  Do truncate from H0 outputting right ortho Hamiltonian
  #         #
  #         HR=copy(H0)
  #         Ss,ss,HL=truncate!(HR;verbose=verbose1,orth=right)
  #         #@test typeof(storage(Ss[1])) == (qns ? BlockSparse{Float64, Vector{Float64}, 2} : Diag{Float64, Vector{Float64}})
  #         DwR=Base.max(get_Dw(HR)...)
  #         @test is_regular_form(HR)
  #         @test isortho(HR,right)
  #         @test check_ortho(HR,right)
  #         for n in 1:N
  #             @test norm(Ss[n-1]*HR[n]-HL[n]*Ss[n]) ≈ 0.0 atol = 1e-14
  #         end   
  #         if verbose
  #             @printf " %4i %4i   %4i   %4i  %4i \n" N NNN Dw0 DwL DwR
  #         end

  #     end
  # end

  # @testset "Orthogonalize/truncate verify gauge invariace of <ψ|H|ψ>, ul=$ul, qbs=$qns" for ul in [lower,upper], qns in [false,true]
  #     initstate(n) = "↑"
  #     for N in [1], NNN in [2,4] #3 site unit cell fails for qns=true.
  #         svd_cutoff=1e-15 #Same as autoMPO uses.
  #         si = infsiteinds("S=1/2", N; initstate, conserve_szparity=qns)
  #         ψ = InfMPS(si, initstate)
  #         for n in 1:N
  #             ψ[n] = randomITensor(inds(ψ[n]))
  #         end
  #         H0=make_transIsing_iMPO(si,NNN;ul=ul)
  #         H0.llim=-1
  #         H0.rlim=1
  #         Hsum0=InfiniteSum{MPO}(H0,NNN)
  #         E0=expect(ψ,Hsum0)

  #         HL=copy(H0)
  #         orthogonalize!(HL;verbose=verbose1,orth=left)
  #         HsumL=InfiniteSum{MPO}(HL,NNN)
  #         EL=expect(ψ,HsumL)
  #         @test EL ≈ E0 atol = 1e-14

  #         HR=copy(HL)
  #         orthogonalize!(HR;verbose=verbose1,orth=right)
  #         HsumR=InfiniteSum{MPO}(HR,NNN)
  #         ER=expect(ψ,HsumR)
  #         @test ER ≈ E0 atol = 1e-14

  #         HL=copy(H0)
  #         truncate!(HL;verbose=verbose1,orth=left)
  #         HsumL=InfiniteSum{MPO}(HL,NNN)
  #         EL=expect(ψ,HsumL)
  #         @test EL ≈ E0 atol = 1e-14

  #         HR=copy(H0)
  #         truncate!(HR;verbose=verbose1,orth=right)
  #         HsumR=InfiniteSum{MPO}(HR,NNN)
  #         ER=expect(ψ,HsumR)
  #         @test ER ≈ E0 atol = 1e-14

  #         H=copy(H0)
  #         truncate!(H;verbose=verbose1)
  #         Hsum=InfiniteSum{MPO}(HR,NNN)
  #         E=expect(ψ,Hsum)
  #         @test E ≈ E0 atol = 1e-14

  #         #@show get_Dw(H0) get_Dw(HL) get_Dw(HR)
  #     end
  # end

  # Slow test, turn off if you are making big changes.
  # @testset "Head to Head autoMPO with 2-body Hamiltonian ul=$ul, QNs=$qns" for ul in [lower,upper],qns in [false,true]
  #     for N in 3:15
  #         NNN=N-1#div(N,2)
  #         sites = siteinds("SpinHalf", N;conserve_qns=qns)
  #         Hauto=make_transIsing_AutoMPO(sites,NNN;ul=ul) 
  #         Dw_auto=get_Dw(Hauto)
  #         Hr=make_transIsing_MPO(sites,NNN;ul=ul) 
  #         truncate!(Hr;verbose=verbose1) #sweep left to right
  #         delta_Dw=sum(get_Dw(Hr)-Dw_auto)
  #         @test delta_Dw<=0
  #         if verbose && delta_Dw<0 
  #             println("Compression beat AutoMPO by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
  #         end
  #         if verbose && delta_Dw>0
  #             println("AutoMPO beat Compression by deltaDw=$delta_Dw for N=$N, NNN=$NNN,lr=right,ul=$ul,QNs=$qns")
  #         end
  #     end  
  # end 

  # Slow test, turn off if you are making big changes.
  # @testset "Head to Head autoMPO with 3-body Hamiltonian" begin
  #     if verbose
  #         @printf "+-----+---------+--------------------+--------------------+\n"
  #         @printf "|     |autoMPO  |    1 truncation    | 2 truncations      |\n"
  #         @printf "|  N  | dE      |   dE     RE   Dw   |   dE     RE   Dw   |\n"
  #     end
  #     eps=1e-15
  #         for N in [6,10,16,20,24]
  #             sites = siteinds("SpinHalf", N;conserve_qns=false)
  #             Hnot=make_3body_AutoMPO(sites;cutoff=-1.0) #No truncation inside autoMPO
  #             H=make_3body_AutoMPO(sites) #Truncated by autoMPO
  #             #@show get_Dw(Hnot)
  #             Dw_auto = get_Dw(H)
  #             psi=randomMPS(sites)
  #             Enot=inner(psi',Hnot,psi)
  #             E=inner(psi',H,psi)
  #             @test E ≈ Enot atol = sqrt(eps)
  #             truncate!(Hnot;verbose=verbose1,max_sweeps=1)
  #             Dw_1=get_Dw(Hnot)
  #             delta_Dw_1=sum(Dw_auto-Dw_1)
  #             Enott1=inner(psi',Hnot,psi)
  #             @test E ≈ Enott1 atol = sqrt(eps)
  #             RE1=abs(E-Enott1)/sqrt(eps)

  #             truncate!(Hnot;verbose=verbose1)
  #             Dw_2=get_Dw(Hnot)
  #             delta_Dw_2=sum(Dw_auto-Dw_2)
  #             Enott2=inner(psi',Hnot,psi)
  #             @test E ≈ Enott2 atol = sqrt(eps)
  #             RE2=abs(E-Enott2)/sqrt(eps)
  #             if verbose
  #                 @printf "| %3i | %1.1e | %1.1e %1.3f %4i | %1.1e %1.3f %4i | \n" N abs(E-Enot) abs(E-Enott1) RE1 delta_Dw_1 abs(E-Enott2) RE2 delta_Dw_2 
  #             end
  #         end
  # end

end #@testset "Truncate/Compress" 
nothing
