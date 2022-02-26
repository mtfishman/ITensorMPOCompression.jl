#
# functions for getting and setting V blocks required for block respecting QX and SVD
#

function getV(W::ITensor,off::V_offsets)::ITensor
    ils=filterinds(inds(W),tags="Link")
    iss=filterinds(inds(W),tags="Site")
    w1=ils[1]
    w2=ils[2]
    v1=Index(dim(w1)-1,tags(ils[1]))
    v2=Index(dim(w2)-1,tags(ils[2]))
    V=ITensor(v1,v2,iss...)
    for ilv in eachindval(v1,v2)
        wlv=(IndexVal(w1,ilv[1].second+off.o1),IndexVal(w2,ilv[2].second+off.o2))
        for isv in eachindval(iss)
            V[ilv...,isv...]=W[wlv...,isv...]
    
        end
    end
    return V
end

function setV(W::ITensor,V::ITensor,off::V_offsets)::ITensor

    wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    vils=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    #@show wils vils
    @assert length(wils)==2
    @assert length(vils)==2
    iss=filterinds(inds(W),tags="Site")
    @assert iss==filterinds(inds(V),tags="Site")
    #
    #  these need to loop in the correct order in order to get the W and V indices to line properly.
    #  one index from each of W & V should be the same, so we just need get these
    #  indices to loop together.
    #
    if tags(wils[1])!=tags(vils[1]) && tags(wils[2])!=tags(vils[2])
        vils=vils[2],vils[1] #swap tags the same on index 1 or 2.
    end

    if hastags(vils[1],"qx")
        @assert dim(wils[2])==dim(vils[2])+1
        if dim(wils[1])>dim(vils[1])+1
            #we need to rezise W_
            iw1=Index(dim(vils[1])+1,tags(wils[1]))
            W1=ITensor(iw1,wils[2],iss)
            others=noncommoninds(W,wils[1])
            for io in eachindval(others...)
                for w in eachindval(iw1)
                    if w.second<dim(iw1)
                        W1[w,io...]=W[wils[1]=>w.second,io...]
                    else
                        W1[w,io...]=W[wils[1]=>dim(wils[1]),io...]
                    end
                end
            end
            W=W1
            wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
        end
    elseif hastags(vils[2],"qx")
        @assert dim(wils[1])==dim(vils[1])+1
        #we need to rezise W
        if dim(wils[2])>dim(vils[2])+1
            iw2=Index(dim(vils[2])+1,tags(wils[2]))
            W1=ITensor(wils[1],iw2,iss) #order matters 
            others=noncommoninds(W,wils[2])
            for io in eachindval(others...)
                for w in eachindval(iw2)
                    if w.second<dim(iw2)
                        W1[w,io...]=W[wils[2]=>w.second,io...]
                    else
                        W1[w,io...]=W[wils[2]=>dim(wils[2]),io...] 
                    end
                end
            end
            W=W1
            wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
        end
    end

    #@show "in setV"  inds(W) wils
    for ilv in eachindval(vils)
        wlv=(IndexVal(wils[1],ilv[1].second+off.o1),IndexVal(wils[2],ilv[2].second+off.o2))
        for isv in eachindval(iss)
            W[wlv...,isv...]=V[ilv...,isv...]
        end
    end
    return W
end

#
#  o1   add row to RL       o2  add column to RL
#   0   at bottom, Dw1+1    0   at right, Dw2+1
#   1   at top, 1           1   at left, 1
#
function growRL(RL::ITensor,iWlink::Index,off::V_offsets)::Tuple{ITensor,Index}
    @assert order(RL)==2
    #is=inds(RL)
    iLlink=filterinds(inds(RL),tags=tags(iWlink))[1] #find the link index of RL
    iLqx=noncommonind(inds(RL),iLlink) #find the qx link of RL
    Dwl=dim(iLlink)
    Dwq=dim(iLqx)
    @assert dim(iWlink)==Dwl+1
    iq=Index(Dwq+1,tags(iLqx))
    RLplus=ITensor(0.0,iq,iWlink)
    @assert norm(RLplus)==0.0
    for jq in eachindval(iLqx)
        for jl in eachindval(iLlink)
            ip=(IndexVal(iq,jq.second+off.o1),IndexVal(iWlink,jl.second+off.o2))
            RLplus[ip...]=RL[jq,jl]
        end
    end
    #add diagonal 1.0
    if !(off.o1==1 && off.o2==1)
        RLplus[iq=>Dwq+1,iWlink=>Dwl+1]=1.0
    end
    if !(off.o1==0 && off.o2==0)
        RLplus[iq=>1,iWlink=>1]=1.0
    end
    return RLplus,iq
end
