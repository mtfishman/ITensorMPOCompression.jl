using Printf
#
# functions for getting and setting V blocks required for block respecting QX and SVD
#
# Handles W with only one link index
function getV1(W::ITensor,off::V_offsets)::ITensor
    ils=filterinds(inds(W),tags="Link")
    iss=filterinds(inds(W),tags="Site")
    @assert length(ils)==1
    w1=ils[1]
    v1=Index(dim(w1)-1,tags(ils[1]))
    V=ITensor(v1,iss...)
    for ilv in eachindval(v1)
        wlv=IndexVal(w1,ilv.second+off.o1)
        for isv in eachindval(iss)
            V[ilv,isv...]=W[wlv,isv...]
    
        end
    end
    return V
end


function getV(W::ITensor,off::V_offsets)::ITensor
    ils=filterinds(inds(W),tags="Link")
    if length(ils)==1
        return getV1(W,off)
    end
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

# Handles W with only one link index
function setV1(W::ITensor,V::ITensor,ms::matrix_state)::ITensor
    wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    vils=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    @assert length(wils)==1
    @assert length(vils)==1
    wil=wils[1]
    vil=vils[1]
    iss=filterinds(inds(W),tags="Site")
    @assert iss==filterinds(inds(V),tags="Site")
    off=V_offsets(ms)
    @assert(off.o1==off.o2)
    if dim(wil)>dim(vil)+1
        #we need to shrink W
        wil1=Index(dim(vil)+1,tags(wil))
        W1=ITensor(0.0,wil1,iss)
        if off.o1==1
            #save first element
            for isv in eachindval(iss)
                W1[wil1=>1,isv...]=W[wil=>1,isv...]
            end
        else
            #save last element
            for isv in eachindval(iss)
                W1[wil1=>dim(wil1),isv...]=W[wil=>dim(wil),isv...]
            end
        end
    else
        W1=W
        wil1=wil
    end


    for ilv in eachindval(vil)
        wlv=IndexVal(wil1,ilv.second+off.o1)
        for isv in eachindval(iss)
            W1[wlv,isv...]=V[ilv,isv...]
        end
    end
    return W1
end

#
#  This function is non trivial for 3 reasons:
#   1) V could have truncted number of rows or columns as a result of rank revealing QX
#      W then has to re-sized accordingly
#   2) If W gets resized we need to preserve the last row or column from the old W.
#   3) V and W should one common link index so we need to find those and pair them together.
#
function setV(W::ITensor,V::ITensor,ms::matrix_state)::ITensor
    wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    if length(wils)==1
        return setV1(W,V,ms) #Handle row/col vectors
    end
    vils=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    #@show wils vils
    @assert length(wils)==2
    @assert length(vils)==2
    iss=filterinds(inds(W),tags="Site")
    @assert iss==filterinds(inds(V),tags="Site")
    #
    #  these need to loop in the correct order in order to get the W and V indices to line properly.
    #  one index from each of W & V should have the same tags, so we just need get these
    #  indices paired up so they can loop together.
    #
    # this would be perfect but it only looks at the first tag "Link", 
    # we are interested in the second tag l=$n :(
    #icom=ITensors.commontags(tags(wils...),tags(vils...))

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
    resize=dim(iwqx)>dim(ivqx)+1
    if resize
        iw1=Index(dim(ivqx)+1,tags(iwqx))
        W1=ITensor(0.0,iw1,iwl,iss)
        others=noncommoninds(W,iwqx)
        @assert off.o1==off.o2
        if  off.o1==1 #save row or col 1
            for io in eachindval(others...)
                W1[iw1=>1,io...]=W[iwqx=>1,io...]
            end
        else #off.o1==0 save row or col Dw
            @assert off.o1==0
            for io in eachindval(others...)
                W1[iw1=>dim(iw1),io...]=W[iwqx=>dim(iwqx),io...]
            end
        end

        W=W1
        wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
        iwqx=iw1
    end # if resize

    for ilv in eachindval(ivqx,ivl)
        wlv=(IndexVal(iwqx,ilv[1].second+off.o1),IndexVal(iwl,ilv[2].second+off.o2))
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

#
#  factor LR such that for
#       lr=left  LR=M*RM_prime
#       lr=right LR=RL_primt*M
#  However becuase of how the ITensor index works we don't need to distinguish between left and 
#  right matrix multiplication in the code.  BUT we do need to worry about upper and lower RL
#  matrices when they are rectangular.  For an upper triangular matrix we wnat to grab the
#  matrix from the right side of R, since that is where the most meat (numerical weight) is
#  Conversly for the lower tri L we want grab M from left side.  In short we want as few
#  zeros as possible in M in order for the SVD decomp and compression to have maximum effect.
#
function getM(RL::ITensor,ms::matrix_state,eps::Float64)::Tuple{ITensor,ITensor,Index,Bool}
    ils=filterinds(inds(RL),tags="Link") 
    iqx=findinds(ils,"qx")[1] #think of this as the row index
    iln=noncommonind(ils,iqx) #think of this as the column index
    Dwq,Dwn=dim(iqx),dim(iln)
    Dwm=Base.min(Dwq,Dwn)
    irm=Index(Dwm,"Link,m") #new common index between Mplus and RL_prime
    imq=Index(Dwq-2,tags(iqx)) #mini version of iqx
    imm=Index(Dwm-2,tags(irm)) #mini version of irm
    M=ITensor(imq,imm)
    # @show inds(M)
    shift=0
    if ms.ul==lower
        shift=Base.max(0,Dwn-Dwq) #for upper rectangular R we want M over at the right
    else #upper
        shift=Base.max(0,Dwq-Dwn)
    end
    #@show shift,ms.ul,Dwn,Dwq
    for j1 in 2:Dwq-1
        for j2 in 2:Dwm-1
            M[imq=>j1-1,imm=>j2-1]=RL[iqx=>j1,iln=>j2+shift]
        end
    end
    #
    # Now we need RL_prime such that RL=M*RL_prime.
    # RL_prime is just the perimeter of RL with 1's on the diagonal
    # Well sort of, if RL is rectangular then htings get a little more involved.
    #
    #@show Dwm dim(iln) Dwn
    non_zero=false
    RL_prime=ITensor(0.0,irm,iln)
    for j1 in 1:dim(irm) #or 1:Dwm
        RL_prime[irm=>j1,iln=>1       ]=RL[iqx=>j1,iln=>1  ] #first col
        RL_prime[irm=>j1,iln=>dim(iln)]=RL[iqx=>j1,iln=>dim(iln)] #last cols
        #check for non-zero elements to right of where I is.
        #@show Dwm iln
        for j2 in Dwm:dim(iln)-1
            #@show j2
            non_zero = non_zero || abs(RL[iqx=>j1,iln=>j2]) >eps
        end
        RL_prime[irm=>j1,iln=>j1+shift]=1.0
    end
    for j2 in 1:dim(iln)
        RL_prime[irm=>1       ,iln=>j2]=RL[iqx=>1  ,iln=>j2]
        RL_prime[irm=>dim(irm),iln=>j2]=RL[iqx=>Dwq,iln=>j2]
    end
    RL_prime[irm=>dim(irm),iln=>dim(iln)]=1.0

    return M,RL_prime,irm,non_zero
end

#                      |1 0 0|
#  given A, spit out G=|0 A 0|
#                      |0 0 1|
#
function grow(A::ITensor,ig1::Index,ig2::Index)
    ils=inds(A)
    @assert length(ils)==order(A)
    #
    # we need to connect the indices of A with ig1,ig2 indices based on matching tags.
    #
    if hastags(ils[1],tags(ig1))
        ia1=ils[1]
        @assert hastags(ils[2],tags(ig2))
        ia2=ils[2]
    elseif hastags(ils[1],tags(ig2))
        ia2=ils[1]
        @assert hastags(ils[2],tags(ig1))
        ia1=ils[2]
    else
        @assert false
    end
    chi1,chi2=dim(ia1),dim(ia2)
    @assert dim(ig1)==chi1+2
    @assert dim(ig2)==chi2+2

    G=ITensor(0.0,ig1,ig2) #would be nice to use delta() but we can't set elements on it.
    G[ig1=>1     ,ig2=>1     ]=1.0;
    G[ig1=>chi1+2,ig2=>chi2+2]=1.0;
    for j1 in 1:chi1
        for j2 in 1:chi2
            G[ig1=>j1+1,ig2=>j2+1]=A[ia1=>j1,ia2=>j2]
        end
    end
    return G
end

