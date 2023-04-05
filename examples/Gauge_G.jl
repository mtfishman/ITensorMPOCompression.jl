using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf

import NDTensors: matrix
import ITensorMPOCompression: need_guage_fix

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

# function insert_xblock(G::ITensor,t::Vector{Float64},il::Index,ir::Index,lr::orth_type, ul::reg_form)
#     @assert dim(ir)==length(t)+2
#     if lr==left
#         irow=ul==lower ? dim(il) : 1
#         for i in 2:dim(ir)-1
#             G[il=>irow,ir=>i]=t[i-1]
#         end
#         # Xrow=slice(G,il=>irow) #slice out the row that contains x
#         # x=Xrow[ir=>2:dim(ir)-1]
#     else
#         icol=ul==lower ? 1 : dim(il)
#         for i in 2:dim(ir)-1
#             G[il=>icol,ir=>i]=t[i-1]
#         end
#         # Xcol=slice(G,il=>icol) #slice out the row that contains x
#         # x=Xcol[ir=>2:dim(ir)-1]
#     end
# end

function extract_xblock(G::ITensor,il::Index,ir::Index,lr::orth_type, ul::reg_form)
    if lr==left 
        irow=ul==lower ? dim(il) : 1
        Xrow=slice(G,il=>irow) #slice out the row that contains x
        x=Xrow[ir=>2:dim(ir)-1]
    else
        icol=ul==lower ? 1 : dim(ir)
        Xcol=slice(G,ir=>icol) #slice out the row that contains x
        x=Xcol[il=>2:dim(il)-1]
    end
    return x
end

function insert_xblock(L::Matrix{Float64},t::Vector{Float64},lr::orth_type, ul::reg_form)
    nr,nc=size(L)
    if lr==left
        @assert nc==length(t)+2
        r = ul==lower ? nr : 1
        for i in 2:nc-1
            L[r,i]=t[i-1]
        end
    else
        @assert nr==length(t)+2
        c=ul==lower ? 1 : nc
        for i in 2:nr-1
            L[i,c]=t[i-1]
        end
    end
    return L
end

# function make_Ls(t,lw,rw,lr,ul)
#     L=dense(delta(lw',lw))
#     insert_xblock(L,t,lw',lw,lr,ul)
#     Linv=dense(delta(rw,rw'))
#     insert_xblock(Linv,-t,rw,rw',lr,ul)

#     # Id=Linv*L
#     # @show Id
#     # # for n in 1:Dw-2
#     #     L[lw'=>Dw,lw=>1+n]=t[n]
#     #     Linv[rw=>Dw,rw'=>1+n]=-t[n]
#     # end
#     return L,Linv
# end

function NDTensors.matrix(il::Index,T::ITensor,ir::Index)
        T1=ITensors.permute(T,il,ir; allow_alias=true)
        return matrix(T1)
end

#@testset verbose=true "LGL^-1 gauge transform, lr=$lr, ul=$ul" for lr in [left], ul in [lower]
@testset verbose=true "LGL^-1 gauge transform, qns=$qns, lr=$lr, ul=$ul" for qns in [false,true], lr in [left,right], ul in [lower,upper]
    
    initstate(n) = "↑"
    rr_cutoff=1e-14
    eps=1e-14
    N,NNN=1,5
    si = infsiteinds("Electron", N; initstate, conserve_qns=qns)
    H0=make_Hubbard_AutoiMPO(si,NNN;ul=ul)
    # si = infsiteinds("S=1/2", N; initstate, conserve_qns=false)
    # H0=make_Heisenberg_AutoiMPO(si,NNN;ul=ul)
    if lr==left
        HL=copy(H0)
        orthogonalize!(HL,ul;orth=mirror(lr),rr_cutoff=rr_cutoff,max_sweeps=1) 
        HR=copy(HL)
        Gs=orthogonalize!(HL,ul;orth=lr,rr_cutoff=rr_cutoff,max_sweeps=1)
    else
        HR=copy(H0)
        orthogonalize!(HR,ul;orth=mirror(lr),rr_cutoff=rr_cutoff,max_sweeps=1) 
        HL=copy(HR)
        Gs=orthogonalize!(HR,ul;orth=lr,rr_cutoff=rr_cutoff,max_sweeps=1)
    end
    @test norm(Gs[0]*HR[1]-HL[1]*Gs[1]) ≈ 0.0 atol = eps
    il,ir=linkinds(Gs,HL,1)
    Dwl,Dwr=dim(il),dim(ir)
    Gm=matrix(il,Gs[1],ir)
    x=extract_xblock(Gs[1],il,ir,lr,ul)
    @test norm(x)>eps
    M=Gm[2:Dwl-1,2:Dwr-1]
    t=(LinearAlgebra.I-M)\vector(x) #solve [I-G]*t=x for t.
    # for i in 1:length(t)
    #     if abs(t[i])<eps
    #         t[i]=0.0
    #     end
    # end
    if lr==right 
        t=-t #swaps L Linv
    end
    L=insert_xblock(1.0*Matrix(LinearAlgebra.I,Dwr,Dwl),t,lr,ul)
    Linv=insert_xblock(1.0*Matrix(LinearAlgebra.I,Dwr,Dwl),-t,lr,ul)
    Gmp=L*Gm*Linv
    Gp=CelledVector([ITensor(Gmp,il,ir)])

    iHLr=commonind(HL[1],Gs[1])
    iHLl=noncommonind(HL[1],iHLr,tags="Link")
    LT=ITensor(L,iHLl',dag(iHLl)) #fails here with qns
    LinvT=ITensor(Linv,dag(iHLr),iHLr')
    HLp=noprime(LT*HL[1]*LinvT,tags="Link")
    
    iHRl=commonind(HR[1],Gs[0])
    iHRr=noncommonind(HR[1],iHRl,tags="Link")
    LT=ITensor(L,iHRl',dag(iHRl))
    LinvT=ITensor(Linv,dag(iHRr),iHRr')
    HRp=noprime(LT*HR[1]*LinvT,tags="Link")
    @test norm(Gp[0]*HRp-HLp*Gp[1]) ≈ 0.0 atol = eps
end
nothing
