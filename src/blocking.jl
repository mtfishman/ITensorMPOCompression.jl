using Printf
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

function setV(W::ITensor,V::ITensor,ms::matrix_state)::ITensor
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
    @assert tags(wils[1])==tags(vils[1]) || tags(wils[2])==tags(vils[2])
    if hastags(vils[1],"qx")
        ivqx=vils[1]
        iwqx=wils[1]
        ivl=vils[2]
        iwl=wils[2]
    else
        @assert hastags(vils[2],"qx")
        ivqx=vils[2]
        iwqx=wils[2]
        ivl=vils[1]
        iwl=wils[1]
    end

    off=V_offsets(ms)
    resize=false
    if ms.lr==left
        if dim(iwqx)>dim(ivqx)+1
            #@printf "left resize %4i to %4i \n" dim(iwqx) dim(ivqx)
            #pprint(W,eps)
            resize=true
                #we need to rezise W_
            iw1=Index(dim(ivqx)+1,tags(iwqx))
            W1=ITensor(0.0,iw1,iwl,iss)
            others=noncommoninds(W,iwqx)
            if ms.ul==lower
                #need to preserve the first column
                @assert off.o1==1 && off.o1==1
                for io in eachindval(others...)
                    W1[iw1=>1,io...]=W[iwqx=>1,io...]
                end
                #@show "after column W1="
                #pprint(W1,eps)

            else #upper
                #need to preserve the last column
                @assert off.o1==0 && off.o1==0
                for io in eachindval(others...)
                    W1[iw1=>dim(iw1),io...]=W[iwqx=>dim(iwqx),io...]
                end
                #@show "after column W1="
                #pprint(W1,eps)
            end #if lower
            W=W1
            wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
            iwqx=iw1
        end # if dim
    else #right
        if dim(iwqx)>dim(ivqx)+1
            #@printf "right resize %4i to %4i \n" dim(iwqx) dim(ivqx)
            #@show "right" iwqx ivqx "W="
            #pprint(W,eps)
            #we need to rezise W_
            resize=true
            iw1=Index(dim(ivqx)+1,tags(iwqx))
            W1=ITensor(iwl,iw1,iss)
            others=noncommoninds(W,iwqx)
            if ms.ul==lower
                #need to preserve the bottom row
                @assert off.o1==0 && off.o1==0
                for io in eachindval(others...)
                    W1[iw1=>dim(iw1),io...]=W[iwqx=>dim(iwqx),io...]
                end
                others=noncommoninds(W,iwqx,iwl)
                for io in eachindval(others...)
                    for j=1:dim(iw1)-1
                        W1[iwl=>dim(iwl),iw1=>j,io...]=W[iwl=>dim(iwl),iwqx=>j,io...]
                    end
                    #W1[iwl=>dim(iwl),iw1=>dim(iw1),io...]=W[iwl=>dim(iwl),iwqx=>dim(iwqx),io...]
                end
            else #upper
                #need to preserve the first column
                @assert off.o1==1 && off.o1==1
                for io in eachindval(others...)
                    W1[iw1=>1,io...]=W[iwqx=>1,io...]
                end
            end #if loswer
            W=W1
            wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
            iwqx=iw1
        end #if dim
    end #if left
    
    #@show "in setV"  inds(W) wils inds(V) vils
    for ilv in eachindval(ivqx,ivl)
        wlv=(IndexVal(iwqx,ilv[1].second+off.o1),IndexVal(iwl,ilv[2].second+off.o2))
        for isv in eachindval(iss)
            W[wlv...,isv...]=V[ilv...,isv...]
        end
    end
    if resize
        #@show "after set V W="
        #pprint(W,eps)
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
