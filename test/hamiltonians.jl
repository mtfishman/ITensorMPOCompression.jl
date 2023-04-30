
using ITensors
using ITensorMPOCompression
using Revise
using Test

include("hamiltonians/hamiltonians.jl")


import ITensorMPOCompression: transpose, orthogonalize!, redim

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)

function random_qindex(d::Int64, nq::Int64)::Index
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
    Hauto = transIsing_AutoMPO(sites, NNN; model_kwargs...)
    Eauto, psi = fast_GS(Hauto, sites)
    @test Eauto ≈ Eexpected atol = eps
    Eauto1 = inner(psi', Hauto, psi)
    @test Eauto1 ≈ Eexpected atol = eps

    #
    #  Make H directly ... should be lower triangular
    #
    Hdirect = transIsing_MPO(sites, NNN; model_kwargs...) #defaults to lower reg form
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
          il = random_qindex(d, nq)
          Dw = dim(il)
          if Dw > 1 + offset
            ilr = redim(il, Dw - offset - 1, offset)
            @test dim(ilr) == Dw - offset - 1
          end
        end #for nq
      end #for d
    end #for offset
  end #@testset

  makeHs = [transIsing_AutoMPO, transIsing_MPO, Heisenberg_AutoMPO]
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
    Hauto = transIsing_AutoMPO(sites, NNN)
    Hhand = transIsing_MPO(sites, NNN)
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
    Hnot = three_body_AutoMPO(sites; cutoff=-1.0) #No truncation inside autoMPO
    H = three_body_AutoMPO(sites; cutoff=1e-15) #Truncated by autoMPO
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

  

  makeHs = [
    (transIsing_MPO, "S=1/2"),
    (transIsing_AutoMPO, "S=1/2"),
    (Heisenberg_AutoMPO, "S=1/2"),
    (Hubbard_AutoMPO, "Electron"),
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

  
  
  models = [
    (transIsing_MPO, "S=1/2", true),
    (transIsing_AutoMPO, "S=1/2", true),
    (Heisenberg_AutoMPO, "S=1/2", true),
    (Heisenberg_AutoMPO, "S=1", true),
    (Hubbard_AutoMPO, "Electron", false),
  ]

  @testset "Convert upper MPO to lower, H=$(model[1]), qns=$qns" for model in models, qns in [false,true]
    N,NNN=5,2
    sites = siteinds(model[2], N; conserve_qns=qns)
    Hu = reg_form_MPO(model[1](sites, NNN;ul=upper);honour_upper=true)
    @test is_regular_form(Hu)
    @test Hu.ul==upper
    
    Hl = transpose(Hu)
    @test Hu.ul==upper
    @test is_regular_form(Hu)
    @test Hl.ul==lower
    @test is_regular_form(Hl)

    @test !check_ortho(Hu,left)
    @test !check_ortho(Hl,left)
    @test !check_ortho(Hu,right)
    @test !check_ortho(Hl,right)
    Hl1=MPO(Hl)
    @test order(Hl1[1])==3
    @test order(Hl1[N])==3

    orthogonalize!(Hl,left)

    @test Hu.ul==upper
    @test is_regular_form(Hu)
    @test Hl.ul==lower
    @test is_regular_form(Hl)

    @test check_ortho(Hl,left)
    @test !check_ortho(Hu,left) 
    @test !check_ortho(Hu,right) 
    Hu1=transpose(Hl)
    #@test check_ortho(Hu1,right)
    @test check_ortho(Hu1,left)
    Hl1=MPO(Hl)
    @test order(Hl1[1])==3
    @test order(Hl1[N])==3
    
  end

  

end #Hamiltonians testset

nothing
