module ITensorMPOCompression

using ITensors
include("util.jl")
include("qx.jl")

export ql,assign!,getV,setV!,growRL,to_openbc,set_scale!,block_qx,canonical!,is_canonical
export tri_type,orth_type,matrix_state,full,upper,lower,none,left,right

@enum tri_type  full upper lower
@enum orth_type none left right

struct matrix_state
    ul::tri_type
    lr::orth_type
end

function is_canonical(W::ITensor,ms::matrix_state,eps::Float64)::Bool
    V=getV(W,1,1)
    d,n,r,c=parse_links(V)
    if ms.lr==left
        rc=c
    elseif ms.lr==right
        rc=r
    else
        assert(false)
    end
    Id=V*prime(V,rc)/d
    Id1=delta(rc,rc)
return norm(Id-Id1)<eps
end

#
#  Figure out the site number, and left and indicies of an MPS or MPO ITensor
#  assumes:
#       1) all tensors have two link indices (edge sites no handled yet)
#       2) link indices all have a "Link" tag
#       3) the "Link" tag is the first tag for each link index
#       4) the second tag has the for "l=nnnnn"  where nnnnn are the integer digits of the link number
#       5) the site number is the larges of the link numbers
#
#  Obviously a lot of things could go wrong with all these assumptions.
#
function parse_links(A::ITensor)::Tuple{Int64,Int64,Index,Index}
    d=dim(filterinds(inds(A),tags="Site")[1])
    ils=filterinds(inds(A),tags="Link")
    @assert length(ils)==2
    t1=tags(ils[1])
    t2=tags(ils[2])
    n1::Int64=tryparse(Int64,String(t1[2])[3:end]) # assume second tag is the "l=n" tag
    n2::Int64=tryparse(Int64,String(t2[2])[3:end]) # assume second tag is the "l=n" tag
    if n1>n2
        return d,n1,ils[2],ils[1]
    else 
        return d,n2,ils[1],ils[2]
    end
end

function assign!(W::ITensor,i1::IndexVal,i2::IndexVal,op::ITensor)
    is=inds(op)
    for s in eachindval(is)
        W[i1,i2,s...]=op[s...]
    end
end

function getV(W::ITensor,o1::Int64,o2::Int64)::ITensor
    @assert o1==0 || o1==1
    @assert o2==0 || o2==1
    ils=filterinds(inds(W),tags="Link")
    iss=filterinds(inds(W),tags="Site")
    w1=ils[1]
    w2=ils[2]
    v1=Index(dim(w1)-1,tags(ils[1]))
    v2=Index(dim(w2)-1,tags(ils[2]))
    V=ITensor(v1,v2,iss...)
    for ilv in eachindval(v1,v2)
        wlv=(IndexVal(w1,ilv[1].second+o1),IndexVal(w2,ilv[2].second+o2))
        for isv in eachindval(iss)
            V[ilv...,isv...]=W[wlv...,isv...]
    
        end
    end
    return V
end

function setV!(W::ITensor,V::ITensor,o1::Int64,o2::Int64)
    @assert o1==0 || o1==1
    @assert o2==0 || o2==1

    wils=filterinds(inds(W),tags="Link")
    vils=filterinds(inds(V),tags="Link")
    @assert length(wils)==2
    @assert length(vils)==2
    @assert dim(wils[1])==dim(vils[1])+1
    @assert dim(wils[2])==dim(vils[2])+1
    iss=filterinds(inds(W),tags="Site")
    @assert iss==filterinds(inds(V),tags="Site")

    for ilv in eachindval(vils)
        wlv=(IndexVal(wils[1],ilv[1].second+o1),IndexVal(wils[2],ilv[2].second+o2))
        for isv in eachindval(iss)
            W[wlv...,isv...]=V[ilv...,isv...]
        end
    end
end

function set_scale!(RL::ITensor,Q::ITensor,o1::Int64,o2::Int64)
    @assert o1==0 || o1==1
    @assert o2==0 || o2==1
    @assert order(RL)==2
    is=inds(RL)
    Dw1,Dw2=map(dim,is)
    i1= o1==0 ? 1 : Dw1
    i2= o2==0 ? 2 : Dw2
    scale=RL[is[1]=>i1,is[2]=>i2]
    @assert abs(scale)>1e-12
    RL./=scale
    Q.*=scale
end

#
#  o1   add row to RL       o2  add column to RL
#   0   at bottom, Dw1+1    0   at right, Dw2+1
#   1   at top, 1           1   at left, 1
#
function growRL(RL::ITensor,iWlink::Index,o1::Int64,o2::Int64)::ITensor
    @assert o1==0 || o1==1
    @assert o2==0 || o2==1
    @assert order(RL)==2
    is=inds(RL)
    iLlinks=filterinds(inds(RL),tags=tags(iWlink)) #find the link index of l
    iLqxs=noncommoninds(inds(RL),iLlinks) #find the qx link of l
    @assert length(iLlinks)==1
    @assert length(iLqxs)==1
    iLlink=iLlinks[1]
    iLqx=iLqxs[1]
    Dwl=dim(iLlink)
    Dwq=dim(iLqx)
    @assert dim(iWlink)==Dwl+1
    iq=Index(Dwq+1,tags(iLqx))
    RLplus=ITensor(0.0,iq,iWlink)
    @assert norm(RLplus)==0.0
    for jq in eachindval(iLqx)
        for jl in eachindval(iLlink)
            ip=(IndexVal(iq,jq.second+o1),IndexVal(iWlink,jl.second+o2))
            RLplus[ip...]=RL[jq,jl]
        end
    end
    #add diagonal 1.0
    if !(o1==1 && o2==1)
        RLplus[iq=>Dwq+1,iWlink=>Dwl+1]=1.0
    end
    if !(o1==0 && o2==0)
        RLplus[iq=>1,iWlink=>1]=1.0
    end
    return RLplus
end

function get_lr_lower(mpo::MPO)::Tuple{ITensor,ITensor}
    N=length(mpo)
    W1=mpo[1]
    ilink=filterinds(inds(W1),tags="l=0")[1]
    l=ITensor(0.0,ilink)
    l[ilink=>dim(ilink)]=1.0

    WN=mpo[N]
    ilink=filterinds(inds(WN),tags="l=$N")[1]
    r=ITensor(0.0,ilink)
    r[ilink=>1]=1.0

    return l,r
end


mutable struct MPOpbc
    mpo::MPO
    l::ITensor #contract  leftmost site with this to get row vector op
    r::ITensor #contract rightmost site with this to get col vector op
    MPOpbc(mpo::MPO)=new(copy(mpo),get_lr_lower(mpo)...) 
end

function to_openbc(mpo::MPOpbc)::MPO
    N=length(mpo.mpo)
    #@show mpo.mpo[1]
    #@show mpo.l
    
    mpo.mpo[1]=mpo.l*mpo.mpo[1]
    mpo.mpo[N]=mpo.mpo[N]*mpo.r
    @assert length(filterinds(inds(mpo.mpo[1]),tags="Link"))==1
    @assert length(filterinds(inds(mpo.mpo[N]),tags="Link"))==1
    return mpo.mpo
end

function to_openbc(mpo::MPO)::MPO
    pbc=MPOpbc(mpo)
    return to_openbc(pbc)
end

function block_qx(W::ITensor,n_site::Int64)
    V=getV(W,1,1) #exctract the V block
    #find the (2) site indices and the l=n_site-1 link index.
    il=filterinds(inds(V),tags="l=$n_site")[1]
    iothers=noncommoninds(inds(V),il)
    Q,L=ql(V,iothers;positive=true) #block respecting QL decomposition
    set_scale!(L,Q,1,1) #rescale so the L(n,n)==1.0
    @assert norm(V-Q*L)<1e-12 
    setV!(W,Q,1,1) #Q is the new V, stuff Q into W

    iWl=filterinds(inds(W),tags="l=$n_site")[1]
    return growRL(L,iWl,1,1) #Now make a full size version of L
end

function canonical!(H::MPO)
    N=length(H)
    #TODO check if obc or pbc
    for n in 1:N-1
        Lplus=block_qx(H[n],n)
        H[n+1]=Lplus*H[n+1]
        il=filterinds(inds(Lplus),tags="l=$n")[1]
        iq=filterinds(inds(Lplus),tags="ql")[1]
        replaceind!(H[n+1],iq,il)
    end
end



end
