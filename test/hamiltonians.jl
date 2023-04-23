
using ITensors
using ITensorInfiniteMPS
using ITensorMPOCompression
using Revise
using Test

import ITensorMPOCompression: redim

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)

function make_random_qindex(d::Int64, nq::Int64)::Index
  qns = Pair{QN,Int64}[]
  for n in 1:nq
    append!(qns, [QN() => rand(1:d)])
  end
  return Index(qns, "Link,l=1")
end

@testset verbose = true "Hamiltonians" begin
  NNEs = [
    (1, -1.5066685458330529),
    (2, -1.4524087749432490),
    (3, -1.4516941302867301),
    (4, -1.4481111362390489),
  ]
  @testset "MPOs hand coded versus autoMPO give same GS energies" for nne in NNEs
    N = 5
    model_kwargs = (hx=0.5,)
    eps = 4e-14 #this is right at the lower limit for passing the tests.
    NNN = nne[1]
    Eexpected = nne[2]

    sites = siteinds("SpinHalf", N; conserve_qns=false)
    ITensors.ITensors.disable_debug_checks() #dmrg crashes when this in.
    #
    #  Use autoMPO to make H
    #
    Hauto = make_transIsing_AutoMPO(sites, NNN; model_kwargs...)
    Eauto, psi = fast_GS(Hauto, sites)
    @test Eauto ≈ Eexpected atol = eps
    Eauto1 = inner(psi', Hauto, psi)
    @test Eauto1 ≈ Eexpected atol = eps

    #
    #  Make H directly ... should be lower triangular
    #
    Hdirect = make_transIsing_MPO(sites, NNN; model_kwargs...) #defaults to lower reg form
    #@show Hauto[1] Hdirect[1]
    @test order(Hdirect[1]) == 3
    @test inner(psi', Hdirect, psi) ≈ Eexpected atol = eps
    Edirect, psidirect = fast_GS(Hdirect, sites)
    @test Edirect ≈ Eexpected atol = eps
    overlap = abs(inner(psi, psidirect))
    @test overlap ≈ 1.0 atol = eps
  end

  @testset "Redim function with non-trivial QN spaces" begin
    for offset in 0:2
      for d in 1:5
        for nq in 1:5
          il = make_random_qindex(d, nq)
          Dw = dim(il)
          if Dw > 1 + offset
            ilr = redim(il, Dw - offset - 1, offset)
            @test dim(ilr) == Dw - offset - 1
          end
        end #for nq
      end #for d
    end #for offset
  end #@testset

  makeHs = [make_transIsing_AutoMPO, make_transIsing_MPO, make_Heisenberg_AutoMPO]
  @testset "Auto MPO Ising Ham with Sz blocking" for makeH in makeHs
    N = 5
    eps = 4e-14 #this is right at the lower limit for passing the tests.
    NNN = 2

    sites = siteinds("SpinHalf", N; conserve_qns=true)
    H = makeH(sites, NNN; ul=lower)
    il = filterinds(inds(H[2]); tags="Link")
    for i in 1:2
      for start_offset in 0:2
        for end_offset in 0:2
          Dw = dim(il[i])
          Dw_new = Dw - start_offset - start_offset
          if Dw_new > 0
            ilr = redim(il[i], Dw_new, start_offset)
            @test dim(ilr) == Dw_new
          end #if
        end #for end_offset
      end #for start_offset 
    end #for i
  end #@testset

  @testset "hand coded versus autoMPO with conserve_qns=true have the same index directions" begin
    N = 5
    hx = 0.0 #Hx!=0 breaks symmetry.
    eps = 4e-14 #this is right at the lower limit for passing the tests.
    NNN = 2

    sites = siteinds("SpinHalf", N; conserve_qns=true)
    Hauto = make_transIsing_AutoMPO(sites, NNN)
    Hhand = make_transIsing_MPO(sites, NNN)
    for (Wauto, Whand) in zip(Hauto, Hhand)
      for ia in inds(Wauto)
        ih = filterinds(Whand; tags=tags(ia), plev=plev(ia))[1]
        @test dir(ia) == dir(ih)
      end
    end
  end

  @testset "Parker eq. 34 3-body Hamiltonian" begin
    N = 15
    sites = siteinds("SpinHalf", N; conserve_qns=false)
    Hnot = make_3body_AutoMPO(sites; cutoff=-1.0) #No truncation inside autoMPO
    H = make_3body_AutoMPO(sites; cutoff=1e-15) #Truncated by autoMPO
    psi = randomMPS(sites)
    Enot = inner(psi', Hnot, psi)
    E = inner(psi', H, psi)
    #@show E-Enot get_Dw(Hnot) get_Dw(H)
    @test E ≈ Enot atol = 3e-9
  end

  function verify_links(H::MPO)
    N = length(H)
    @test order(H[1]) == 3
    @test hastags(H[1], "Link,l=1")
    @test hastags(H[1], "Site,n=1")
    for n in 2:(N - 1)
      @test order(H[n]) == 4
      @test hastags(H[n], "Link,l=$(n-1)")
      @test hastags(H[n], "Link,l=$n")
      @test hastags(H[n], "Site,n=$n")
      il, = inds(H[n - 1]; tags="Link,l=$(n-1)")
      ir, = inds(H[n]; tags="Link,l=$(n-1)")
      @test id(il) == id(ir)
    end
    @test order(H[N]) == 3
    @test hastags(H[N], "Link,l=$(N-1)")
    @test hastags(H[N], "Site,n=$N")
    il, = inds(H[N - 1]; tags="Link,l=$(N-1)")
    ir, = inds(H[N]; tags="Link,l=$(N-1)")
    @test id(il) == id(ir)
  end

  function verify_links(H::InfiniteMPO)
    Ncell = length(H)
    @test order(H[1]) == 4
    @test hastags(H[1], "Link,c=0,l=$Ncell")
    @test hastags(H[1], "Link,c=1,l=1")
    @test hastags(H[1], "Site,c=1,n=1")
    for n in 2:Ncell
      @test order(H[n]) == 4
      @test hastags(H[n], "Link,c=1,l=$(n-1)")
      @test hastags(H[n], "Link,c=1,l=$n")
      @test hastags(H[n], "Site,c=1,n=$n")
      il, = inds(H[n - 1]; tags="Link,c=1,l=$(n-1)")
      ir, = inds(H[n]; tags="Link,c=1,l=$(n-1)")
      @test id(il) == id(ir)
    end
    il, = inds(H[1]; tags="Link,c=0,l=$Ncell")
    ir, = inds(H[Ncell]; tags="Link,c=1,l=$Ncell")
    @test id(il) == id(ir)
    @test order(H[Ncell]) == 4
  end

  @testset "Production of iMPOs from AutoMPO, Ncell=$Ncell, NNN=$NNN, qns=$qns" for qns in
                                                                                    [
      false, true
    ],
    Ncell in [1, 2, 3, 4],
    NNN in [1, 2, 3, 4, 5]

    initstate(n) = "↑"
    site_type = "S=1/2"
    si = infsiteinds(site_type, Ncell; initstate, conserve_qns=qns)
    H = make_transIsing_AutoiMPO(si, NNN; ul=lower)
    @test length(H) == Ncell
    Dws = get_Dw(H)
    @test all(y -> y == Dws[1], Dws)
    verify_links(H)
  end

  makeHs = [
    (make_transIsing_MPO, "S=1/2"),
    (make_transIsing_AutoMPO, "S=1/2"),
    (make_Heisenberg_AutoMPO, "S=1/2"),
    (make_Hubbard_AutoMPO, "Electron"),
  ]

  @testset "Reg for H=$(makeH[1]), ul=$ul, qns=$qns" for makeH in makeHs,
    qns in [false, true],
    ul in [lower, upper]

    N = 5
    sites = siteinds(makeH[2], N; conserve_qns=qns)
    H = makeH[1](sites, 2; ul=ul)
    @test is_regular_form(H, ul)
    @test hasqns(H[1]) == qns
    verify_links(H)
  end

  makeHs = [
    (make_transIsing_iMPO, "S=1/2"),
    (make_transIsing_AutoiMPO, "S=1/2"),
    (make_Heisenberg_AutoiMPO, "S=1/2"),
    (make_Hubbard_AutoiMPO, "Electron"),
  ]

  @testset "Reg for H=$(makeH[1]), ul=$ul, qns=$qns" for makeH in makeHs,
    qns in [false, true],
    ul in [lower, upper],
    Ncell in 1:5

    initstate(n) = "↑"
    sites = infsiteinds(makeH[2], Ncell; initstate, conserve_qns=qns)
    H = makeH[1](sites, 2; ul=ul)
    @test is_regular_form(H, ul)
    @test hasqns(H[1]) == qns
    verify_links(H)
  end

  

  @testset "Convert upper to lower using lattice reverse idea" begin
    N,NNN=5,2
    sites = siteinds("SpinHalf", N; conserve_qns=false)
    Hu = reg_form_MPO(make_transIsing_MPO(sites, NNN;ul=upper))
    Hl = transpose(Hu)
    @test is_regular_form(Hu)
    @test is_regular_form(Hl)
    @test !check_ortho(Hu,left)
    @test !check_ortho(Hl,left)
    @test !check_ortho(Hu,right)
    @test !check_ortho(Hl,right)
    Hl1=MPO(Hl)
    @test order(Hl1[1])==3
    @test order(Hl1[N])==3

    ac_orthogonalize!(Hl,left)

    @test check_ortho(Hl,left)
    @test !check_ortho(Hu,left)
    Hu1=transpose(Hl)
    @test check_ortho(Hu1,right)
    Hl1=MPO(Hl)
    @test order(Hl1[1])==3
    @test order(Hl1[N])==3
    
  end
end #Hamiltonians testset

nothing
