using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Test
using Revise,Printf,SparseArrays
Base.show(io::IO, f::Float64) = @printf(io, "%1.1e", f) #dumb way to control float output

models=[
    [make_transIsing_MPO,"S=1/2",true],
    [make_transIsing_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1",true],
    [make_Hubbard_AutoMPO,"Electron",false],
    ]

# @testset "Gauge fix finite $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false,true], ul=[lower,upper]
#     eps=1e-14
    
#     N=10 #5 sites
#     NNN=7 #Include 2nd nearest neighbour interactions
#     sites = siteinds(model[2],N,conserve_qns=qns)
#     Hrf=reg_form_MPO(model[1](sites,NNN;ul=ul))
#     pre_fixed=model[3] #Hamiltonian starts gauge fixed
#     state=[isodd(n) ? "Up" : "Dn" for n=1:N]

#     H=MPO(Hrf)
#     psi=randomMPS(sites,state)
#     E0=inner(psi',H,psi)
    
#     @test is_regular_form(Hrf)
#     @test pre_fixed==is_gauge_fixed(Hrf,eps)
#     gauge_fix!(Hrf)
#     @test is_regular_form(Hrf)
#     @test is_gauge_fixed(Hrf,eps)
#     He=MPO(Hrf)
#     E1=inner(psi',He,psi)
#     @test E0 ≈ E1 atol = eps
# end


models=[
    (make_transIsing_iMPO,"S=1/2",true),
    (make_transIsing_AutoiMPO,"S=1/2",true),
    (make_Heisenberg_AutoiMPO,"S=1/2",true),
    (make_Heisenberg_AutoiMPO,"S=1",true),
    (make_Hubbard_AutoiMPO,"Electron",false)
]

import ITensorMPOCompression: check, extract_blocks, A0, b0, c0, vector_o2, reg_form_Op

# @testset "Gauge solve s/t $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false], ul=[lower]
#     initstate(n) = "↑"
#     N,NNN=4,4
#     si = infsiteinds(model[2], N; initstate, conserve_qns=qns)
#     H0=model[1](si,NNN;ul=ul)
#     Hrf=reg_form_iMPO(H0)
#     for n in eachindex(Hrf)
#         il,ir=parse_links(Hrf[n].W)
#         Hrf[n].ileft=il
#         Hrf[n].iright=ir
#     end 

#     lr=left
#     A0s=Vector{Matrix}()
#     b0s=Vector{Float64}()
#     c0s=Vector{Float64}()
#     nr,nc=0,0
#     irb,icb=Vector{Int64}(),Vector{Int64}()
#     ir,ic=1,1
#     for W in Hrf
#         check(W)
#         Wb=extract_blocks(W,lr;all=true)
#         A_0=matrix(Wb.irA,A0(Wb),Wb.icA)
#         push!(A0s,A_0)
#         append!(b0s,vector_o2(b0(Wb)))
#         append!(c0s,vector_o2(c0(Wb)))
#         push!(irb,ir)
#         push!(icb,ic)
#         nr+=size(A_0,1)
#         nc+=size(A_0,2)
#         ir+=size(A_0,1)
#         ic+=size(A_0,2)
#     end

#     @assert nr==nc
#     n=nr
#     N=length(A0s)
#     Ms,Mt=spzeros(n,n),spzeros(n,n)
#     ib,ib=1,1
#     for n in eachindex(A0s)
#         nr,nc=size(A0s[n])
#         ir,ic=irb[n],icb[n]
#         #
#         #  These system will generally not bee so big that sparse improves performance significantly.
#         #
#         sparseA0=sparse(A0s[n])
#         droptol!(sparseA0,1e-15)
#         Ms[irb[n]:irb[n]+nr-1,icb[n]:icb[n]+nc-1]=sparseA0
#         Mt[irb[n]:irb[n]+nr-1,icb[n]:icb[n]+nc-1]=sparse(LinearAlgebra.I,nr,nc)
#         if n==1
#             Ms[irb[n]:irb[n]+nr-1,icb[N]:icb[N]+nc-1]=-sparse(LinearAlgebra.I,nr,nc)
#             Mt[irb[n]:irb[n]+nr-1,icb[N]:icb[N]+nc-1]=-sparseA0
#         else
#             Ms[irb[n]:irb[n]+nr-1,icb[n-1]:icb[n]-1]=-sparse(LinearAlgebra.I,nr,nc)
#             Mt[irb[n]:irb[n]+nr-1,icb[n-1]:icb[n]-1]=-sparseA0
#         end
#     end
#     @show length(Ms.nzval)/(n*n)
#     @show length(Mt.nzval)/(n*n)
#     s=Ms\b0s
#     t=transpose(transpose(Mt)\c0s)
#     # # display(s)
#     # display(t)
#     @test norm(Ms*s-b0s)<1e-15
#     @test norm(transpose(t*Mt)-c0s)<1e-15
# end

@testset "Gauge fix infinite $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false true], ul=[lower], N in [1,2,3,4], NNN in [2,4,7]
    eps=1e-14
    initstate(n) = "↑"
    si = infsiteinds(model[2], N; initstate, conserve_qns=qns)
    H0=model[1](si,NNN;ul=ul)
    Hrf=reg_form_iMPO(H0)
    for n in eachindex(Hrf)
        il,ir=parse_links(Hrf[n].W)
        Hrf[n].ileft=il
        Hrf[n].iright=ir
    end 
    pre_fixed=model[3] #Hamiltonian starts gauge fixed

    @test pre_fixed==is_gauge_fixed(Hrf,eps)
    gauge_fix!(Hrf)
    Wb=extract_blocks(Hrf[1],left;all=true)
    @test norm(b0(Wb))<eps
    @test norm(c0(Wb))<eps
    @test is_gauge_fixed(Hrf,eps)
end
nothing
