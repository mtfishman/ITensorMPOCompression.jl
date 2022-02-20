module ITensorMPOCompression

using ITensors
include("util.jl")
include("qx.jl")

export ql,lq,assign!,getV,setV!,growRL,to_openbc,set_scale!,block_qx,canonical!,is_canonical
export tri_type,orth_type,matrix_state,full,upper,lower,none,left,right,parse_links
export detect_upper_lower,has_pbc,is_regular_form

@enum tri_type  full upper lower diagonal
@enum orth_type none left right

struct matrix_state
    ul::tri_type
    lr::orth_type
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
    if length(ils)==2
        t1=tags(ils[1])
        t2=tags(ils[2])
        n1::Int64=tryparse(Int64,String(t1[2])[3:end]) # assume second tag is the "l=n" tag
        n2::Int64=tryparse(Int64,String(t2[2])[3:end]) # assume second tag is the "l=n" tag
        if n1>n2
            return d,n1,ils[2],ils[1]
        else 
            return d,n2,ils[1],ils[2]
        end
    elseif length(ils)==1
        t=tags(ils[1])
        n::Int64=tryparse(Int64,String(t[2])[3:end]) # assume second tag is the "l=n" tag
        if n==1
            return d,n,Index(1),ils[1] #row vector
        else
            return d,n,ils[1],Index(1) #col vector
        end
    else
        @assert false
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

    wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    vils=filterinds(inds(V),tags="Link") #should be {lq,ql} and {l=n,l=n-1} depending on sweep direction
    @assert length(wils)==2
    @assert length(vils)==2
    @assert dim(wils[1])==dim(vils[1])+1
    @assert dim(wils[2])==dim(vils[2])+1
    iss=filterinds(inds(W),tags="Site")
    @assert iss==filterinds(inds(V),tags="Site")
    #
    #  these need to loop in the correct order in order to get the W and V indices to line properly.
    #  one index from each of W & V should be the same, so we just need these indices to loop together.
    #
    if tags(wils[1])!=tags(vils[1]) && tags(wils[2])!=tags(vils[2])
        vils=vils[2],vils[1] #swap got tags the same on index 1 or 2.
    end

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
    i2= o2==0 ? 1 : Dw2
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
    ul=detect_upper_lower(mpo,1e-10)
    @assert ul!=full
    N=length(mpo)
    W1=mpo[1]
    llink=filterinds(inds(W1),tags="l=0")[1]
    l=ITensor(0.0,llink)

    WN=mpo[N]
    rlink=filterinds(inds(WN),tags="l=$N")[1]
    r=ITensor(0.0,rlink)
    if ul==lower
        l[llink=>dim(llink)]=1.0
        r[rlink=>1]=1.0
    else
        l[llink=>1]=1.0
        r[rlink=>dim(rlink)]=1.0
    end

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

function block_qx(W::ITensor,lr::orth_type)
    d,n,r,c=parse_links(W)
    if lr==left
        V=getV(W,1,1) #extract the V block
        il=filterinds(inds(V),tags="l=$n")[1] #link to next site to the right
    elseif lr==right
        V=getV(W,0,0) #extract the V block
        il=filterinds(inds(V),tags="l=$(n-1)")[1] #link to next site to left
    else
        assert(false)
    end

    iothers=noncommoninds(inds(V),il)
    if lr==left
        Q,L=ql(V,iothers;positive=true) #block respecting QL decomposition
        set_scale!(L,Q,1,1) #rescale so the L(n,n)==1.0
        @assert norm(V-Q*L)<1e-12 
        setV!(W,Q,1,1) #Q is the new V, stuff Q into W
    
        iWl=filterinds(inds(W),tags="l=$n")[1]
        Lplus=growRL(L,iWl,1,1) #Now make a full size version of L
    elseif lr==right
        @assert detect_upper_lower(V,1e-14)==lower
        L,Q=lq(V,iothers;positive=true) #block respecting QL decomposition
        set_scale!(L,Q,0,0) #rescale so the L(n,n)==1.0
        @assert norm(V-L*Q)<1e-12 
        setV!(W,Q,0,0) #Q is the new V, stuff Q into W
        @assert detect_upper_lower(W,1e-14)==lower
        iWl=filterinds(inds(W),tags="l=$(n-1)")[1]
        Lplus=growRL(L,iWl,0,0) #Now make a full size version of L
    
    else
        assert(false)
    end
    return Lplus
end


function is_canonical(W::ITensor,ms::matrix_state,eps::Float64)::Bool
    if ms.lr==left
        V=getV(W,1,1)
    elseif ms.lr==right
        V=getV(W,0,0)
    else
        assert(false)
    end
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

function is_canonical(H::MPO,ms::matrix_state,eps::Float64)::Bool
    N=length(H)
    if ms.lr==left
        r=1:N-1
    elseif ms.lr==right
        r=2:N
    else
        assert(false)
    end
    ic=true
    for n in r
        ic=ic &&  is_canonical(H[n],ms,eps)
    end
    return ic
end

function has_pbc(H::MPO)::Bool
    N=length(H)
    nind1=length(inds(H[1]))
    nindN=length(inds(H[N]))
    nlink1=length(findinds(H[N],"Link"))
    nlinkN=length(findinds(H[N],"Link"))
    leftl=hastags(inds(H[1]),"l=0")
    rightl=hastags(inds(H[N]),"l=$N")
    obc::Bool = nind1==3 && nindN==3 &&  nlink1==1 && nlinkN==1 && leftl==0 && rightl==0
    pbc::Bool = nind1==4 && nindN==4 &&  nlink1==2 && nlinkN==2 && leftl==1 && rightl==1
    if !obc && !pbc  #if its not one or the othr something is really messed up!
        @assert false
    end
    return pbc
end

function detect_upper_lower(W::ITensor,eps::Float64)::tri_type
    d,n,r,c=parse_links(W)
    zero_upper=true
    zero_lower=true
    for ir in eachindval(r)
        for ic in eachindval(c)
            oprc=norm(slice(W,ir,ic))
            is_zero=norm(oprc)<eps
            if (ic.second>ir.second) #above diagonal
                zero_upper = zero_upper && is_zero
            end
            if (ic.second<ir.second) #below diagonal
                zero_lower = zero_lower && is_zero
            end
        end
    end
    if zero_upper && zero_lower
        ret=diagonal
    elseif zero_upper
        ret=lower
    elseif zero_lower
        ret=upper
    else
        ret=full
    end
    return ret
end


            


function detect_upper_lower(H::MPO,eps::Float64)::tri_type
    @assert length(H)>1
    ul=detect_upper_lower(H[2],eps) #skip left and right sites in case MPO is obc
    for n in 3:length(H)-1
        iln=detect_upper_lower(H[n],eps)
        if (iln!=ul && iln!=diagonal) 
            if ul==diagonal
                ul=iln
            else
                ul=full
            end
        end
    end
    return ul
end

#
# This test is complicated by two things
#   1) Itis not clear to me (JR) that the A block of an MPO matrix must be upper or lower triangular for
#      block respecting compression to work properly.  Parker et al. make no definitive statement about this.
#      It is my intention to test this empirically using auto MPO generated which tend to non-triangular.
#   2) As a consequence of 1, we cannot decide in advance whether to test of upper or lower regular forms.
#      We must therefore test for both and return true if either one is true. 
#
function is_regular_form(W::ITensor,eps::Float64)::Bool
    d,n,r,c=parse_links(W)
    Dw1,Dw2=dim(r),dim(c)
    #handle edge row and col vectors
    if Dw1==1 #left edge row vector
        return is_unit(slice(W,c=>Dw2),eps)
    end
    if Dw2==1 #right edge col vector
        return is_unit(slice(W,r=>1),eps)
    end
    #ul=detect_upper_lower(W) is the a requirement??
    irf=true #is regular form
    # There must be unit matrices in the top left and bottom right corners.
    irf=irf && is_unit(slice(W,r=>1  ,c=>1  ),eps)
    irf=irf && is_unit(slice(W,r=>Dw1,c=>Dw2),eps)
    # now look for zeroed partial rows and columns, avoiding the corner unit matricies.
    top_row_zero=true #lower
    bot_row_zero=true #upper tri
    left__col_zero=true #upper tri
    right_col_zero=true #lower tri
    #lower tri zero tests
    for ic in 2:Dw2
        top_row_zero=top_row_zero     && norm(slice(W,r=>1  ,c=>ic))<eps
    end
    for ir in 1:Dw1-1
        right_col_zero=right_col_zero && norm(slice(W,r=>ir,c=>Dw2))<eps
    end
    #upper tri zero tests
    for ic in 1:Dw2-1
        bot_row_zero=bot_row_zero     && norm(slice(W,r=>Dw1,c=>ic))<eps
    end
    for ir in 2:Dw1
        left__col_zero=left__col_zero && norm(slice(W,r=>ir,c=>1  ))<eps
    end
    irf=irf && ((top_row_zero&&right_col_zero) || (bot_row_zero && left__col_zero))
    # before returning we should also check for any unit matricies along the diagonal
    # this gets a tricky for non-square matrices.
    diag_unit = false
    if Dw1>=Dw2
        for ic in 2:Dw2-1
            diag_unit = diag_unit || abs(norm(slice(W,r=>ic,c=>ic  ))-sqrt(d))<eps
        end
    else
        dr=Dw2-Dw1
        for ir in 2:Dw1-1
            diag_unit = diag_unit || abs(norm(slice(W,r=>ir,c=>ir+dr))-sqrt(d))<eps
        end
    end
    if diag_unit
        println("ITensorMPOCompression.is_regular_for\n  Warning: found unit operator along the diagonal of an MPO")
    end    
   
    return irf
end

function is_regular_form(H::MPO,eps::Float64)::Bool
    N=length(H)
    irf=true
    for n in 1:N
        irf=irf && is_regular_form(H[n],eps)
        if !irf break end
    end
    return irf
end

function canonical!(H::MPO,lr::orth_type)
    @assert has_pbc(H)
    N=length(H)
    if lr==left
        for n in 1:N-1 #sweep left to right
            Lplus=block_qx(H[n],lr)
            H[n+1]=Lplus*H[n+1] 
            il=filterinds(inds(Lplus),tags="l=$n")[1]
            iq=filterinds(inds(Lplus),tags="ql")[1]
            replaceind!(H[n+1],iq,il)
        end
    else
        for n in N:-1:2 #sweep right to left
            Lplus=block_qx(H[n],lr)
            @assert detect_upper_lower(H[n],1e-14)==lower
            H[n-1]=Lplus*H[n-1]
            il=filterinds(inds(Lplus),tags="l=$(n-1)")[1]
            iq=filterinds(inds(Lplus),tags="lq")[1]
            replaceind!(H[n-1],iq,il)
            @assert detect_upper_lower(H[n-1],1e-14)==lower
        end
    end
end



end
