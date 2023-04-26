using ITensors
using ITensorMPOCompression
using Test
using Revise, Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

import ITensorMPOCompression: flip

@testset "Extract blocks qns=$qns, ul=$ul" for qns in [false, true], ul in [lower, upper]
  eps = 1e-15
  N = 5 #5 sites
  NNN = 2 #Include 2nd nearest neighbour interactions
  sites = siteinds("Electron", N; conserve_qns=qns)
  d = dim(inds(sites[1])[1])
  H = reg_form_MPO(make_Hubbard_AutoMPO(sites, NNN; ul=ul))

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

@testset "Detect regular form qns=$qns, ul=$ul" for qns in [false, true],
  ul in [lower, upper]

  eps = 1e-15
  N = 5 #5 sites
  NNN = 2 #Include 2nd nearest neighbour interactions
  sites = siteinds("Electron", N; conserve_qns=qns)
  H = reg_form_MPO(make_Hubbard_AutoMPO(sites, NNN; ul=ul))
  for W in H
    @test is_regular_form(W, ul, eps)
    @test dim(W.ileft) == 1 || dim(W.iright) == 1 || !is_regular_form(W, flip(ul), eps)
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
