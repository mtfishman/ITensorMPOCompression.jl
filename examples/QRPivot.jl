using ITensors
using ITensorMPOCompression
using Test,Printf

import ITensors: tensor
import ITensorMPOCompression: @checkflux, mpoc_checkflux, insert_xblock

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

#
#      1 0 0
#  W = b A 0, For lr=left, slice out c, for lr=right slice out b.
#      d c I
#
function get_bc_block(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state)::ITensor
    @assert hasinds(W,ilb,ilf)
    Dwf,Dwb=dim(ilf),dim(ilb)
    @assert ms.ul==lower
    if ms.lr==left
        bc_block= W[ilb=>Dwb:Dwb,ilf=>2:Dwf-1]
    else
        bc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
    end
    return bc_block
end

#
#      1 0 0
#  W = b A 0, For lr=left, slice out [A], for lr=right slice out [b A].
#      d c I                         [c]                          
#
function get_Abc_block(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state)::Tuple{ITensor,Index,Index}
    Dwf,Dwb=dim(ilf),dim(ilb)
    @assert ms.ul==lower
    if ms.lr==left
        if Dwb>1
            Abc_block= W[ilb=>2:Dwb,ilf=>2:Dwf-1]
        else
            Abc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
        end
    else
        if Dwb>1
            Abc_block= W[ilb=>1:Dwb-1,ilf=>2:Dwf-1]
        else
            Abc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
        end
    end
    ilf1,=inds(Abc_block,tags=tags(ilf))
    ilb1,=inds(Abc_block,tags=tags(ilb)) #New indices.
    return Abc_block,ilf1,ilb1
end
#
#      1 0 0                           
#  W = b A 0, make a new ITensor Wp, 
#      d c I  
#                          [1]
#  for lr=left 1) copy the [b] column from W into Wp, 2) assign Abc into [A]. 3) set bottom corner I
#                          [d]                                           [c]
#                          
#  for lr=right 1) copy the [d c I] row from W into Wp, 2) assign Abc into [b A]. 3) set top corner I
#  
function set_Abc_block(W::ITensor,Abc::ITensor,ilf::Index,ilb::Index,iq::Index,ms::matrix_state)
    is=noncommoninds(W,ilf,ilb)
    @assert hasinds(W,ilf,ilb)
    @assert hasinds(Abc,iq,is...)
    Dwb,Dwf,Dwq=dim(ilb),dim(ilf),dim(iq)+2
    # Todo provide the correct QN("??") space here.  Can we pluck it out of ilf?
    ilqp=redim(iq,Dwq,1) #replaces ilf: 
    Wp=ITensor(0.0,ilb,ilqp,is)
    if ms.lr==left
        Wp[ilb=>1:Dwb,ilqp=>1:1]=W[ilb=>1:Dwb,ilf=>1:1]
        Wp[ilb=>Dwb:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>Dwb:Dwb,ilf=>Dwf:Dwf]
        if Dwb>1
            Wp[ilb=>2:Dwb,ilqp=>2:Dwq-1]=Abc
        else
            Wp[ilb=>1:1,ilqp=>2:Dwq-1]=Abc
        end
    else
        Wp[ilb=>1:1,ilqp=>1:1]=W[ilb=>1:1,ilf=>1:1]
        Wp[ilb=>1:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>1:Dwb,ilf=>Dwf:Dwf]
        if Dwb>1
            Wp[ilb=>1:Dwb-1,ilqp=>2:Dwq-1]=Abc
        else
            Wp[ilb=>1:1,ilqp=>2:Dwq-1]=Abc
        end
    end
    return Wp,ilqp
end


function ac_qx(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state;kwargs...)::Tuple{ITensor,ITensor,Index}
    @checkflux(W)
    @assert hasinds(W,ilf)
    @assert hasinds(W,ilb)
    eps=1e-15
    I=slice(W,ilf=>1,ilb=>1)
    is=inds(I)
    d=dim(is[1])  
    bc=get_bc_block(W,ilf,ilb,ms) #TODO capture removed space
    Dwf,Dwb=dim(ilf),dim(ilb)
    # println("W0")
    # pprint(ilb,W*dag(I)/d,ilf)
    if norm(bc*dag(I))>eps && Dwb>2
        println("------------LWL transform-----------------")
        # @show ms ilb ilf
        Wg=reduce_and_transform(W,ilb,ilf,ms,eps)
        # println("Wg0")
        # pprint(ilb,Wg*dag(I)/d,ilf)
        bcz=get_bc_block(Wg,ilf,ilb,ms)
        @assert norm(bcz*dag(I))<eps
    end
    Abc,ilf1,_=get_Abc_block(W,ilf,ilb,ms)
    @checkflux(Abc)
    if ms.lr==left
        Qinds=noncommoninds(Abc,ilf1)
        Q,R,iq=qr(Abc,Qinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf1
        R,Q,iq=rq(Abc,Rinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    end
    @checkflux(Q)
    @checkflux(R)
    Q*=sqrt(d)
    R/=sqrt(d)
    Wp,iqp=set_Abc_block(W,Q,ilf,ilb,iq,ms) #TODO inject correct removed space
    R=prime(R,ilf1)
    #  TODO fix mimatched spaces when H=non auto MPO.  Need QN()=>1,QN()=>Chi,QN()=>1 space in MPO
    #@show  inds(R) dag(iqp) ilf
    Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    return Wp,Rp,iqp
end

function add_dummy_links!(H::MPO)
    N=length(H)
    ils=map(n->linkind(H,n),1:N-1)
    ts=ITensors.trivial_space(ils[1])
    T=eltype(H[1])
    il0=Index(ts;tags="Link,l=0",dir=dir(dag(ils[1])))
    ilN=Index(ts;tags="Link,l=$N",dir=dir(ils[1]))
    d0=onehot(T, il0 => 1)
    dN=onehot(T, ilN => 1)
    H[1]*=d0
    H[N]*=dN
    return [il0,ils...,ilN],d0,dN
end

#
#  Create lists of non-zero rows and columns.  Then equalize the lists if needed by adding 
#  zero rows xor zero columns to get equal size lists.
#
function reduce_rows_columns(A::Matrix,eps::Float64)
    nr,nc=size(A)
    nonzero_cols=map(n-> maximum(abs.(A[:,n]))>eps,1:nc)
    nonzero_rows=map(n-> maximum(abs.(A[n,:]))>eps,1:nr)
    pr=findall(i->(i),nonzero_rows) 
    pc=findall(i->(i),nonzero_cols) 
    nr,nc=length(pr),length(pc)
    n=Base.max(nr,nc)
    if nc>nr #not enough rows, add some of the zero rows        
        prz=findall(i->(!i),nonzero_rows) #list of zero rows
        pr=sort(append!(pr,prz[1:n-nr]))
    elseif  nr>nc #not enough cols, add some of the zero cols
        pcz=findall(i->(!i),nonzero_cols) #list of zero cols
        pc=sort(append!(pc,pcz[1:n-nc]))
    end
    return pr,pc
end

function gather(A::ITensor,ir::Index,ic::Index,pr::Vector{Int64},pc::Vector{Int64})::Tuple{ITensor,Index,Index}
    @assert length(pr)==length(pc)
    n=length(pr)
    iothers=noncommoninds(A,ir,ic)
    ir1,ic1=redim(ir,n),redim(ic,n)
    A1=ITensor(0.0,ir1,ic1,iothers)
    for r in 1:n
        for c in 1:n
            assign!(A1,slice(A,ir=>pr[r],ic=>pc[c]),ir1=>r,ic1=>c)
        end
    end
    return A1,ir1,ic1
end

function scatter!(A::ITensor,ir::Index,ic::Index,A1::ITensor,ir1::Index,ic1::Index,pr::Vector{Int64},pc::Vector{Int64})
    @assert hasinds(A,ir,ic)
    @assert hasinds(A1,ir1,ic1)
    n=length(pr)
    for r in 1:n
        for c in 1:n
            assign!(A,slice(A1,ir1=>r,ic1=>c),ir=>pr[r],ic=>pc[c])
        end
    end
end

function scatter(A::Matrix,p::Vector{Int64},n1::Int64)
    #A1=1.0*Matrix(LinearAlgebra.I,n1,n1)
    A1=zeros(eltype(A),n1,n1)
    n=length(p)
    @assert size(A,1)==n
    @assert size(A,2)==n
    for r in 1:n
        for c in 1:n
            A1[p[r],p[c]]=A[r,c]
        end
    end
    return A1
end

function get_identity(W::ITensor,ir::Index,ic::Index)
    Id=slice(W,ir=>1,ic=>1)
    d=dims(Id)[1]
    return Id,d
end

function get_bc_block(W0::Matrix,ms::matrix_state)
    @assert ms.ul==lower
    nr,nc=size(W0)
    if ms.lr==left
        bc0=W0[nr,2:nc-1]
    else
        bc0=W0[2:nr-1,1]
    end
    return bc0
end
function gauge_transform(W::ITensor,ir::Index,ic::Index,ms::matrix_state)
    @assert dim(ir)==dim(ic)
    Id,d=get_identity(W,ir,ic)
    W0=matrix(ir,W*dag(Id)/d,ic)
    n=dim(ir)
    A0=W0[2:n-1,2:n-1]
    c0=get_bc_block(W0,ms)
    if ms.lr==left
        t=transpose(LinearAlgebra.I-A0)\c0
    else
        t=-(LinearAlgebra.I-A0)\c0
    end
    L=insert_xblock(1.0*Matrix(LinearAlgebra.I,n,n),t,ms)
    Linv=insert_xblock(1.0*Matrix(LinearAlgebra.I,n,n),-t,ms)

    LT=ITensor(L,ir',dag(ir)) 
    LinvT=ITensor(Linv,dag(ic),ic')
    return noprime(LT*W*LinvT,tags="Link"),L,Linv
end

function make_projector(p::Vector{Int64},n1::Int64)
    n=length(p)
    P=zeros(Float64,n1,n)
    for c in eachindex(p)
        P[p[c],c]=1.0
    end
    return P
end

function reduce_and_transform(W::ITensor,ir::Index,ic::Index,ms::matrix_state,eps::Float64)
    I,d=get_identity(W,ir,ic)
    W0=matrix(ir,W*dag(I),ic)
    #display(W0)
    pr,pc=reduce_rows_columns(W0,eps) #List of rows and columns to keep
    # Pr=make_projector(pr,dim(ir))
    # Pc=make_projector(pc,dim(ic))
    # display(Pr*transpose(Pr))
    # display(transpose(Pr)*Pr)

    #@show pr pc
    Wr,irr,icr=gather(W,ir,ic,pr,pc) #Reduced/square W with zeros chopped out
    # PrT=ITensor(transpose(Pr),irr,ir)
    # PcT=ITensor(Pc,ic,icr)
    # Wr1=PrT*W*PcT
    # @assert norm(Wr-Wr1)<eps

    #pprint(Wr)
    Wrg,Lr,Linvr=gauge_transform(Wr,irr,icr,ms)  #Perform gauge tranform so c0=<c,I>=0

    # LinvT=ITensor(Linvr,irr',dag(irr)) 
    # LT=ITensor(Lr,dag(icr),icr')
    # Wr1=noprime(LinvT*Wrg*LT,tags="Link")
    # @assert norm(Wr-Wr1)<eps
    # @assert norm(Linvr*Lr-LinearAlgebra.I)<eps
    # @assert norm(Lr*Linvr-LinearAlgebra.I)<eps


    scatter!(W,ir,ic,Wrg,irr,icr,pr,pc) #insert the tranformed rows and cols back into W
    #L=scatter(Lr,pr,dim(ir))
    # L=Pr*Lr*transpose(Pr)
    # display(L)
    # display(scatter(Lr,pr,dim(ir)))
    # Linv=scatter(Linvr,pc,dim(ic))
    # LT=ITensor(L,ir',dag(ir)) 
    # LinvT=ITensor(Linv,dag(ic),ic')
    # Wg1=noprime(LT*W*LinvT,tags="Link")
    #@pprint(W)
    #@pprint(Wg1)

    L=scatter(Lr,pc,dim(ic))
    Linv=scatter(Linvr,pr,dim(ir))

    return W,L,Linv
end

# function back_transform(W::ITensor,ir::Index,ic::Index,t::Vector{Float64},ms::matrix_state)
#     L=insert_xblock(1.0*Matrix(LinearAlgebra.I,n,n),t,ms)
#     Linv=insert_xblock(1.0*Matrix(LinearAlgebra.I,n,n),-t,ms)
# end
models=[
    # [make_transIsing_AutoMPO,"S=1/2"],
    # [make_Heisenberg_AutoMPO,"S=1/2"],
    [make_Hubbard_AutoMPO,"Electron"],
    ]

# @testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns" for model in models, qns in [false]
#     eps=1e-14
#     N=5 #5 sites
#     NNN=3 #Include 2nd nearest neighbour interactions
#     sites = siteinds(model[2],N,conserve_qns=qns);
#     H=model[1](sites,NNN);
#     @show get_Dw(H)
#     state=[isodd(n) ? "Up" : "Dn" for n=1:N]
#     psi=randomMPS(sites,state)
#     E0=inner(psi',H,psi)
#     ms=matrix_state(lower,left)
#     ils,d0,dN=add_dummy_links!(H)
#     ilb=ils[1]
#     for n in sweep(H,left)
#         @show n
#         ilf=linkind(H,n)
#         if dim(ilb)==1
#             W,R,iqp=block_qx(H[n],ilf,ms.ul;orth=ms.lr)
#         else
#             W,R,iqp=ac_qx(H[n],ilf,ilb,ms)
#         end
#         @test norm(H[n]-W*R)<1e-15
#         @test check_ortho(W,ms)
#         H[n]=W
#         H[n+1]=R*H[n+1]
#         ilb=dag(iqp)
#     end
#     @test check_ortho(H,ms)
#     qns && show_directions(H)
#     H[1]*=dag(d0)
#     H[N]*=dag(dN)
#     E1=inner(psi',H,psi)
#     @test E0 ≈ E1 atol = eps

#     # ils,d0,dN=add_dummy_links!(H)
#     # ms=matrix_state(lower,right)
#     # ilb=ils[N+1]
#     # for n in sweep(H,right)
#     #     ilf=dag(linkind(H,n-1))
#     #     W,R,iqp=ac_qx(H[n],ilf,ilb,ms)
#     #     @test norm(H[n]-W*R)<1e-15
#     #     @test check_ortho(W,ms)
#     #     H[n]=W
#     #     H[n-1]=R*H[n-1]
#     #     ilb=dag(iqp)
#     # end
#     # @test check_ortho(H,ms)
#     # qns && show_directions(H)
#     # H[1]*=dag(d0)
#     # H[N]*=dag(dN)
#     # E2=inner(psi',H,psi)
#     # @test E0 ≈ E2 atol = eps
# end

@testset "Gauge transform rectangular W" begin
    eps=1e-14
    ms=matrix_state(lower,left)
    N=5 #5 sites
    NNN=2 #Include 2nd nearest neighbour interactions
    sites = siteinds("Electron",N,conserve_qns=false)
    H=make_Hubbard_AutoMPO(sites,NNN)
    for n in 2:3
        W=copy(H[n])
        ir,ic =linkind(H,n-1),linkind(H,n)
        I,d=get_identity(W,ir,ic)
        Wg,L,Linv=reduce_and_transform(W,ir,ic,ms,eps)
        c=get_bc_block(Wg,ic,ir,ms)
        c0=c*dag(I)/d
        @test norm(c0)<eps
        #pprint(ir,Wg*dag(I)/d,ic)
        # Now do the back transform
        # LT=ITensor(L,dag(ic),ic') 
        # LinvT=ITensor(Linv,ir',dag(ir))
        # W1=noprime(LinvT*Wg*LT,tags="Link")
        pprint(Wg-H[n])
        # @show norm(W-W1)


        # ms=matrix_state(lower,right)
        # W=copy(H[n])
        # Wg,L,Linv=reduce_and_transform(W,ir,ic,ms,eps)
        # b=get_bc_block(Wg,ir,ic,ms)
        # b0=b*dag(I)/d
        # @test norm(b0)<eps
        #pprint(ir,Wg*dag(I)/d,ic)
    end
end


nothing