using ITensors
using ITensorMPOCompression
using Test

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

function get_bc_block(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state)::ITensor
    Dwf,Dwb=dim(ilf),dim(ilb)
    @assert ms.ul==lower
    if ms.lr==left
        bc_block= W[ilb=>Dwb:Dwb,ilf=>2:Dwf-1]
    else
        bc_block= W[ilb=>1:1,ilf=>2:Dwf-1]
    end
    return bc_block
end

function get_Abc_block(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state)::Tuple{ITensor,Index,Index}
    Dwf,Dwb=dim(ilf),dim(ilb)
    @assert ms.ul==lower
    if ms.lr==left
        Abc_block= W[ilb=>2:Dwb,ilf=>2:Dwf-1]
    else
        Abc_block= W[ilb=>1:Dwb-1,ilf=>2:Dwf-1]
    end
    ilf1,=inds(Abc_block,tags=tags(ilf))
    ilb1,=inds(Abc_block,tags=tags(ilb)) #New indices.
    return Abc_block,ilf1,ilb1
end

function set_Abc_block(W::ITensor,Abc::ITensor,ilf::Index,ilb::Index,iq::Index,ms::matrix_state)
    is=noncommoninds(W,ilf,ilb)
    @assert hasinds(W,ilf,ilb)
    @assert hasinds(Abc,iq,is...)
    Dwb,Dwf,Dwq=dim(ilb),dim(ilf),dim(iq)+2
    ilqp=redim(iq,Dwq,1) #replaces ilf
    #@show inds(Abc) ilb ilqp
    Wp=ITensor(0.0,ilb,ilqp,is)
    if ms.lr==left
        Wp[ilb=>1:Dwb,ilqp=>1:1]=W[ilb=>1:Dwb,ilf=>1:1]
        Wp[ilb=>Dwb:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>Dwb:Dwb,ilf=>Dwf:Dwf]
        Wp[ilb=>2:Dwb,ilqp=>2:Dwq-1]=Abc
    else
        Wp[ilb=>1:1,ilqp=>1:1]=W[ilb=>1:1,ilf=>1:1]
        Wp[ilb=>1:Dwb,ilqp=>Dwq:Dwq]=W[ilb=>1:Dwb,ilf=>Dwf:Dwf]
        Wp[ilb=>1:Dwb-1,ilqp=>2:Dwq-1]=Abc
    end
    return Wp,ilqp
end

function ac_qx(W::ITensor,ilf::Index,ilb::Index,ms::matrix_state;kwargs...)::Tuple{ITensor,ITensor,Index}
    I=slice(W,ilf=>1,ilb=>1)
    is=inds(I)
    d=dim(is[1])  
    bc=get_bc_block(W,ilf,ilb,ms) #TODO capture removed space
    @assert norm(bc*dag(I))<1e-15
    Abc,ilf1,ilb1=get_Abc_block(W,ilf,ilb,ms)
    if ms.lr==left
        Qinds=noncommoninds(Abc,ilf1)
        Q,R,iq=qr(Abc,Qinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    else
        Rinds=ilf1
        R,Q,iq=rq(Abc,Rinds;positive=true,rr_cutoff=1e-14,tags=tags(ilf))
    end
    Q*=sqrt(d)
    R/=sqrt(d)
    Wp,iqp=set_Abc_block(W,Q,ilf,ilb,iq,ms) #TODO inject correct removed space
    R=prime(R,ilf1)
    #@show inds(R) iqp ilf
    #  TODO fix mimatched spaces when H=non auto MPO.  Need QN()=>1,QN()=>Chi,QN()=>1 space in MPO
    Rp=noprime(ITensorMPOCompression.grow(R,dag(iqp),ilf'))
    return Wp,Rp,iqp
end

@testset "Ac/Ab block respecting decomposition, qns=$qns" for qns in [false,true]
    N=5 #5 sites
    NNN=3 #Include 2nd nearest neighbour interactions
    sites = siteinds("S=1/2",N,conserve_qns=qns);
    H=make_transIsing_AutoMPO(sites,NNN);
    W,il,ir=H[2],dag(linkind(H,1)),linkind(H,2)
    
    Wp,Rp,iqp=ac_qx(W,ir,il,matrix_state(lower,left))
    @test norm(W-Wp*Rp)<1e-15
    @test check_ortho(Wp,matrix_state(lower,left))

    Wp,Rp,iqp=ac_qx(W,il,ir,matrix_state(lower,right))
    @test norm(W-Wp*Rp)<1e-15
    @test check_ortho(Wp,matrix_state(lower,right))
end