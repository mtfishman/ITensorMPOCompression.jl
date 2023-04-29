using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS

using Test
using Revise, Printf

include("hamiltonians/hamiltonians.jl")
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

import ITensorMPOCompression: extract_blocks, regform_blocks, reg_form_iMPO


@testset "Extract blocks qns=$qns, ul=$ul" for qns in [false, true], ul in [lower, upper]
  eps = 1e-15
  N = 5 #5 sites
  NNN = 2 #Include 2nd nearest neighbour interactions
  sites = siteinds("Electron", N; conserve_qns=qns)
  d = dim(inds(sites[1])[1])
  H = reg_form_MPO(Hubbard_AutoMPO(sites, NNN; ul=ul);honour_upper=true)

  lr = ul == lower ? left : right

  Wrf = H[1]
  nr, nc = dims(Wrf)
  #pprint(Wrf.W)
  Wb = extract_blocks(Wrf, lr; all=true, V=true)
  @test norm(matrix(Wb.𝕀) - 1.0 * Matrix(LinearAlgebra.I, d, d)) < eps
  @test isnothing(Wb.𝐀̂)
  if ul == lower
    @test isnothing(Wb.𝐛̂)
    @test norm(array(Wb.𝐝̂) - array(Wrf[nr:nr, 1:1])) < eps
    @test norm(array(Wb.𝐜̂) - array(Wrf[nr:nr, 2:(nc - 1)])) < eps
  else
    @test isnothing(Wb.𝐜̂)
    @test norm(array(Wb.𝐝̂) - array(Wrf[1:1, nc:nc])) < eps
    @test norm(array(Wb.𝐛̂) - array(Wrf[1:1, 2:(nc - 1)])) < eps
  end
  @test norm(array(Wb.𝐕̂) - array(Wrf[1:1, 2:nc])) < eps

  Wrf = H[N]
  nr, nc = dims(Wrf)
  Wb = extract_blocks(Wrf, lr; all=true, V=true, fix_inds=true)
  @test norm(matrix(Wb.𝕀) - 1.0 * Matrix(LinearAlgebra.I, d, d)) < eps
  @test isnothing(Wb.𝐀̂)
  if ul == lower
    @test isnothing(Wb.𝐜̂)
    @test norm(array(Wb.𝐝̂) - array(Wrf[nr:nr, 1:1])) < eps
    @test norm(array(Wb.𝐛̂) - array(Wrf[2:(nr - 1), 1:1])) < eps
  else
    @test isnothing(Wb.𝐛̂)
    @test norm(array(Wb.𝐝̂) - array(Wrf[1:1, nc:nc])) < eps
    @test norm(array(Wb.𝐜̂) - array(Wrf[2:(nr - 1), nc:nc])) < eps
  end
  @test norm(array(Wb.𝐕̂) - array(Wrf[2:nr, 1:1])) < eps

  Wrf = H[2]
  nr, nc = dims(Wrf)
  Wb = extract_blocks(Wrf, lr; all=true, V=true, fix_inds=true, Ac=true)
  if ul == lower
    @test norm(matrix(Wb.𝕀) - 1.0 * Matrix(LinearAlgebra.I, d, d)) < eps
    @test norm(array(Wb.𝐝̂) - array(Wrf[nr:nr, 1:1])) < eps
    @test norm(array(Wb.𝐛̂) - array(Wrf[2:(nr - 1), 1:1])) < eps
    @test norm(array(Wb.𝐜̂) - array(Wrf[nr:nr, 2:(nc - 1)])) < eps
    @test norm(array(Wb.𝐀̂) - array(Wrf[2:(nr - 1), 2:(nc - 1)])) < eps
    @test norm(array(Wb.𝐀̂𝐜̂) - array(Wrf[2:nr, 2:(nc - 1)])) < eps
  else
    @test norm(matrix(Wb.𝕀) - 1.0 * Matrix(LinearAlgebra.I, d, d)) < eps
    @test norm(array(Wb.𝐝̂) - array(Wrf[1:1, nc:nc])) < eps
    @test norm(array(Wb.𝐛̂) - array(Wrf[1:1, 2:(nc - 1)])) < eps
    @test norm(array(Wb.𝐜̂) - array(Wrf[2:(nr - 1), nc:nc])) < eps
    @test norm(array(Wb.𝐀̂) - array(Wrf[2:(nr - 1), 2:(nc - 1)])) < eps
    @test norm(array(Wb.𝐀̂𝐜̂) - array(Wrf[2:(nr - 1), 2:nc])) < eps
  end
  @test norm(array(Wb.𝐕̂) - array(Wrf[2:nr, 2:nc])) < eps
end

@testset "Extract blocks MPO with fix_inds=true ul=$ul" for ul in [lower, upper]

  N = 5 #5 sites
  NNN = 2 #Include 2nd nearest neighbour interactions
  sites = siteinds("Electron", N; conserve_qns=false)
  d = dim(inds(sites[1])[1])
  H = reg_form_MPO(Hubbard_AutoMPO(sites, NNN; ul=ul);honour_upper=true)

  lr = ul == lower ? left : right

  Wrf = H[1]
  Wb = extract_blocks(Wrf, lr; fix_inds=true)
  if lr==left
    @test hasinds(Wb.𝐜̂,Wb.irc,Wb.icc)
    @test hasinds(Wb.𝐝̂,Wb.ird,Wb.icd)
    @test Wb.ird==Wb.irc
  else
    @test hasinds(Wb.𝐛̂,Wb.irb,Wb.icb)
    @test hasinds(Wb.𝐝̂,Wb.ird,Wb.icd)
    @test Wb.icd==Wb.icb
  end
  Wrf = H[N]
  Wb = extract_blocks(Wrf, lr; fix_inds=true)
  if lr==right
    @test hasinds(Wb.𝐜̂,Wb.irc,Wb.icc)
    @test hasinds(Wb.𝐝̂,Wb.ird,Wb.icd)
    @test Wb.ird==Wb.irc
  else
    @test hasinds(Wb.𝐛̂,Wb.irb,Wb.icb)
    @test hasinds(Wb.𝐝̂,Wb.ird,Wb.icd)
    @test Wb.icd==Wb.icb
  end

  Wrf = H[2]
  Wb = extract_blocks(Wrf, lr; fix_inds=true)
  @test hasinds(Wb.𝐀̂,Wb.irA,Wb.icA)
  @test hasinds(Wb.𝐛̂,Wb.irb,Wb.icb)
  @test hasinds(Wb.𝐜̂,Wb.irc,Wb.icc)
  @test hasinds(Wb.𝐝̂,Wb.ird,Wb.icd)
  @test Wb.ird==Wb.irc
  @test Wb.icd==Wb.icb
  @test Wb.irA==Wb.irb
end

function test_links(Wb1::regform_blocks,Wb2::regform_blocks)
  @test hasinds(Wb1.𝐀̂,Wb1.irA,Wb1.icA)
  @test hasinds(Wb1.𝐛̂,Wb1.irb,Wb1.icb)
  @test hasinds(Wb1.𝐜̂,Wb1.irc,Wb1.icc)
  @test hasinds(Wb1.𝐝̂,Wb1.ird,Wb1.icd)
  @test Wb1.ird==Wb1.irc
  @test Wb1.icd==Wb1.icb
  @test Wb1.irA==Wb1.irb
  @test Wb1.icA==Wb1.icc
  @test id(Wb1.icA)==id(Wb2.irA)
  @test dir(Wb1.icA)==dir(dag(Wb2.irA))
end

@testset "Extract blocks iMPO N=$N, qns=$qns with fix_inds=true ul=$ul" for N in [1,4], qns in [false,true], ul in [lower]
  initstate(n) = "↑"
  NNN = 2 #Include 2nd nearest neighbour interactions
  sites = infsiteinds("Electron", N;initstate, conserve_qns=qns)
  H = reg_form_iMPO(Hubbard_AutoiMPO(sites, NNN; ul=ul);honour_upper=true)

  lr = ul == lower ? left : right

  Wbs=extract_blocks(H,lr;fix_inds=true)
  for n in 1:N-1
    test_links(Wbs[n],Wbs[n+1])
  end
  test_links(Wbs[N],Wbs[1])

 
end


@testset "Detect regular form qns=$qns, ul=$ul" for qns in [false, true],
  ul in [lower, upper]

  eps = 1e-15
  N = 5 #5 sites
  NNN = 2 #Include 2nd nearest neighbour interactions
  sites = siteinds("Electron", N; conserve_qns=qns)
  H = reg_form_MPO(Hubbard_AutoMPO(sites, NNN; ul=ul);honour_upper=true)
  for W in H
    @test is_regular_form(W, ul;eps=eps)
    @test dim(W.ileft) == 1 || dim(W.iright) == 1 || !is_regular_form(W, mirror(ul);eps=eps)
  end
end

@testset "Sub tensor assign for block sparse matrices with compatable QN spaces" begin 
    qns=[QN("Sz",0)=>1,QN("Sz",0)=>3,QN("Sz",0)=>2]
    i,j=Index(qns,"i"),Index(qns,"j")
    A=randomITensor(i,j)
    nr,nc=dims(A)
    B=copy(A)
    for dr in 0:nr-1
        for dc in 0:nc-1
            #@show dr dc nr-dr nc-dc
            B[i=>1:nr-dr,j=>1:nc-dc]=A[i=>1:nr-dr,j=>1:nc-dc]
            @test norm(B-A)==0
            B[i=>1+dr:nr,j=>1:nc-dc]=A[i=>1+dr:nr,j=>1:nc-dc]
            @test norm(B-A)==0
        end
    end
end

#
#  This is a tough nut to crack.  But it is not needed for MPO compression.
#
# @testset "Sub tensor assign for block sparse matrices with in-compatable QN spaces" begin 
#     qns=[QN("Sz",0)=>1,QN("Sz",0)=>3,QN("Sz",0)=>2]
#     qnsC=[QN("Sz",0)=>2,QN("Sz",0)=>2,QN("Sz",0)=>2] #purposely miss allgined.
#     i,j=Index(qns,"i"),Index(qns,"j")
#     A=randomITensor(i,j)
#     nr,nc=dims(A)
#     ic,jc=Index(qnsC,"i"),Index(qnsC,"j")
#     C=randomITensor(ic,jc)
#     @show dense(A) dense(C)
#     C[ic=>1:nr,jc=>1:nc]=A[i=>1:nr,j=>1:nc]
#     @show matrix(A)-matrix(C)
#     @test norm(matrix(A)-matrix(C))==0
# end

nothing
