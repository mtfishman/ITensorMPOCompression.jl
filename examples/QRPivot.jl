using ITensors
using ITensorMPOCompression
using Test,Printf

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
    ilqp=redim(iq,Dwq,1) #replaces ilf
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
    @assert hasinds(W,ilf)
    @assert hasinds(W,ilb)
    I=slice(W,ilf=>1,ilb=>1)
    is=inds(I)
    d=dim(is[1])  
    bc=get_bc_block(W,ilf,ilb,ms) #TODO capture removed space
    @assert norm(bc*dag(I))<1e-15
    Abc,ilf1,_=get_Abc_block(W,ilf,ilb,ms)
    if ms.lr==left
        Qinds=noncommoninds(Abc,ilf1)
        Q,R,iq=qr(Abc,Qinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf1
        R,Q,iq=rq(Abc,Rinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    end
    Q*=sqrt(d)
    R/=sqrt(d)
    Wp,iqp=set_Abc_block(W,Q,ilf,ilb,dag(iq),ms) #TODO inject correct removed space
    R=prime(R,ilf1)
    #  TODO fix mimatched spaces when H=non auto MPO.  Need QN()=>1,QN()=>Chi,QN()=>1 space in MPO
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
    H[1]*=onehot(T, il0 => 1)
    H[N]*=onehot(T, ilN => 1)
    return [il0,ils...,ilN]
end

@testset "Ac/Ab block respecting decomposition, qns=$qns" for qns in [false,true]
    N=5 #5 sites
    NNN=3 #Include 2nd nearest neighbour interactions
    sites = siteinds("S=1/2",N,conserve_qns=qns);
    H=make_transIsing_AutoMPO(sites,NNN);
    ms=matrix_state(lower,left)
    ils=add_dummy_links!(H)
    ilb=ils[1]
    for n in sweep(H,left)
        ilf=linkind(H,n)
        W,R,iqp=ac_qx(H[n],ilf,ilb,ms)
        @test norm(H[n]-W*R)<1e-15
        @test check_ortho(W,ms)
        H[n]=W
        H[n+1]=R*H[n+1]
        ilb=dag(iqp)
    end
    @test check_ortho(H,ms)
    
    ms=matrix_state(lower,right)
    ilb=ils[N+1]
    for n in sweep(H,right)
        ilf=dag(linkind(H,n-1))
        W,R,iqp=ac_qx(H[n],ilf,ilb,ms)
        @test norm(H[n]-W*R)<1e-15
        @test check_ortho(W,ms)
        H[n]=W
        H[n-1]=R*H[n-1]
        ilb=dag(iqp)
    end
    @test check_ortho(H,ms)
end