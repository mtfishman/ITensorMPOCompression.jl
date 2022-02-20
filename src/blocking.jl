#
# functions for getting and setting V blocks required for block respecting QX and SVD
#

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
