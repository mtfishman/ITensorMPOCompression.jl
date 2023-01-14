using Printf

#
#  create a Vblock IndexRange using the supplied offset.
#
range(i::Index,offset::Int64) = i=>1+offset:dim(i)-1+offset
#
# functions for getting and setting V blocks required for block respecting QX and SVD
#
function getV(W::ITensor,off::V_offsets)::ITensor
    if order(W)==3
        w1=filterinds(inds(W),tags="Link")[1]
        return W[range(w1,off.o1)]
    elseif order(W)==4
        w1,w2=filterinds(inds(W),tags="Link")
        return W[range(w1,off.o1),range(w2,off.o2)]
    else 
        @show inds(W)
        @error("getV(W::ITensor,off::V_offsets) Case with order(W)=$(order(W)) not supported.")
    end
end
# Handles W with only one link index
function setV1(W::ITensor,V::ITensor,ms::matrix_state)::ITensor
    iwl,=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    ivl,=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    @assert inds(W,tags="Site")==inds(V,tags="Site")
    off=V_offsets(ms)
    @assert(off.o1==off.o2)
    #
    #  If rank reduction occured in the QR process we may need to resize W.
    #
    if dim(iwl)>dim(ivl)+1         #do we need to shrink W?
        iwl1=redim(iwl,dim(ivl)+1) #re-dimension the wil index.
        T=eltype(V)
        W1=ITensor(T(0.0),iwl1,inds(W,tags="Site"))
        rc= off.o1==1 ? 1 : dim(iwl) #preserve row/col 1 or Dw
        rc1= off.o1==1 ? 1 : dim(iwl1)
        W1[iwl1=>rc1:rc1]=W[iwl=>rc:rc]
        W=W1 #overwrite W with the resized W.
        iwl=iwl1 #overwrite the old Link index with the resized version.
    end
    V=replacetags(V,tags(ivl),tags(iwl)) #V needs to have the same Link tags as W for the assignment below to work.
    W[range(iwl,off.o1)]=V #Finally we can do the assingment of the V-block.
    return W
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
    @assert off.o1==off.o2
    #
    #  If rank reduction occured in the QR process we may need to resize W.
    #
    if dim(iwqx)>dim(ivqx)+1
        iw1qx=redim(iwqx,dim(ivqx)+1) #re-dimension the wqx index.
        T=eltype(V)
        W1=ITensor(T(0.0),iw1qx,iwl,iss)
        rc= off.o1==1 ? 1 : dim(iwqx) #preserve row/col 1 or Dw
        rc1= off.o1==1 ? 1 : dim(iw1qx)
        W1[iw1qx=>rc1:rc1,iwl=>1:dim(iwl)]=W[iwqx=>rc:rc,iwl=>1:dim(iwl)] #copy row/col rc/rc1
        W=W1 #overwrite W with the resized W.
        iwqx=iw1qx #overwrite the old qx index with the resized version.
    end # if resize
    V=replacetags(V,tags(ivqx),tags(iwqx)) #V needs to have the same Link tags as W for the assignment below to work.
    W[range(iwqx,off.o1),range(iwl,off.o2)]=V #Finally we can do the assingment of the V-block.
    return W
end

#
#  o1   add row to RL       o2  add column to RL
#   0   at bottom, Dw1+1    0   at right, Dw2+1
#   1   at top, 1           1   at left, 1
#
function growRL(RL::ITensor,iwl::Index,off::V_offsets)::Tuple{ITensor,Index}
    @assert order(RL)==2
    irl,=filterinds(inds(RL),tags=tags(iwl)) #find the link index of RL
    irqx=noncommonind(inds(RL),irl) #find the qx link of RL
    @assert dim(iwl)==dim(irl)+1
    ipqx=redim(irqx,dim(irqx)+1) 
    T=eltype(RL)
    RLplus=ITensor(T(0.0),ipqx,iwl)
    RLplus[ipqx=>1,iwl=>1]=1.0 #add 1.0's in the corners
    RLplus[ipqx=>dim(ipqx),iwl=>dim(iwl)]=1.0
    RLplus[range(ipqx,off.o1),range(iwl,off.o2)]=RL #plop in RL in approtriate sub block.
    return RLplus,dag(ipqx)
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
    il=noncommonind(ils,iqx) #Get the remaining link index
    Dwq,Dwl=dim(iqx),dim(il)
    Dwm=Base.min(Dwq,Dwl)
    im=Index(Dwm,"Link,m") #new common index between Mplus and RL_prime
    
    shift=0
    if ms.ul==lower
        shift=Base.max(0,Dwl-Dwq) #for upper rectangular R we want M over at the right
    else #upper
        shift=Base.max(0,Dwq-Dwl)
    end
   
    M=RL[iqx=>2:Dwq-1,il=>2:Dwm-1] #pull out the M sub block
    M=replacetags(M,tags(il),tags(im)) #change Link,l=n to Link,m
    #
    # Now we need RL_prime such that RL=M*RL_prime.
    # RL_prime is just the perimeter of RL with 1's on the diagonal
    # Well sort of, if RL is rectangular then things get a little more involved.
    #
    RL=replacetags(RL,tags(iqx),tags(im))  #change Link,qx to Link,m
    iqx=replacetags(iqx,tags(iqx),tags(im))
    RL_prime=ITensor(0.0,im,il)
    #
    #  Copy over the perimeter of RL.
    #  TODO: Tighten this up based on ms.ul, avoid copying zeros.
    #
    irm=im=>1:Dwm
    irl=il=>1:Dwl
    RL_prime[irm,il=>1:1]=RL[iqx=>1:Dwm,il=>1:1] #first col
    RL_prime[irm,il=>Dwl:Dwl]=RL[iqx=>1:Dwm,il=>Dwl:Dwl] #last col
    RL_prime[im=>1:1,irl]=RL[iqx=>1:1 ,irl] #first row
    RL_prime[im=>Dwm:Dwm,irl]=RL[iqx=>Dwq:Dwq,irl] #last row
    
    # Fill in diaginal
    for j1 in 1:Dwm #or 1:Dwm
        RL_prime[im=>j1,il=>j1+shift]=1.0
    end
    RL_prime[im=>Dwm,il=>dim(il)]=1.0

    #
    #  Test for non-zero block in RLprime.
    #
    non_zero=false
    if Dwm<=dim(il)-1
        #we should only get here if truncate is not bailing our on rectangular RL.
        ar=abs.(RL[iqx=>1:Dwm,il=>Dwm:dim(il)-1])
        @show RL M ar dim(ar)
        non_zero = dim(ar)>0 && maximum(ar)>0.0
    end

    return M,RL_prime,im,non_zero
end

#          |1 0 0|
#  given G=|0 M 0| spit out M and its left index iml
#          |0 0 1|
#
function getM(G::ITensor,igl::Index,igr::Index)::Tuple{ITensor,Index}
    @assert order(G)==2
    #@assert tags(igl)!=tags(igr) can't use subtensor until this works.
    # M1=G[igl=>2:dim(igl)-1,igr=>2:dim(igr)-1]
    # iml1,=inds(M1,tags=tags(igl))
    Dwl,Dwr=dim(igl),dim(igr)
    iml,imr=Index(Dwl-2,tags(igl)),Index(Dwr-2,tags(igr))
    M=ITensor(iml,imr)
    for jl in 2:Dwl-1
        for jr in 2:Dwr-1
            M[iml=>jl-1,imr=>jr-1]=G[igl=>jl,igr=>jr]
        end
    end
    return M,iml
end


#                      |1 0 0|
#  given A, spit out G=|0 A 0| , indices of G are provided.
#                      |0 0 1|
#
function grow(A::ITensor,ig1::Index,ig2::Index)
    Dw1,Dw2=dim(ig1),dim(ig2)
    G=ITensor(0.0,ig1,ig2) #would be nice to use delta() but we can't set elements on it.
    G[ig1=>1  ,ig2=>1  ]=1.0;
    G[ig1=>Dw1,ig2=>Dw2]=1.0;
    G[ig1=>2:Dw1-1,ig2=>2:Dw2-1]=A
    return G
end

function grow(A::ITensor,ig1::QNIndex,ig2::Index)
    @assert !hasqns(A)
    @assert !hasqns(ig2)
    G=grow(A,removeqns(ig1),ig2) #grow A into G as dense tensors
    ig2q=addqns(ig2,[QN()=>dim(ig2)];dir=dir(dag(ig1))) #make a QN version of index ig2
    @assert id(ig2)==id(ig2q) #If the ID changes then subsequent contractions will fail.
    return convert_blocksparse(G,ig1,ig2q) #fabricate a 1-block blocksparse version.
end
function grow(A::ITensor,ig1::Index,ig2::QNIndex)
    @assert !hasqns(A)
    @assert !hasqns(ig1)
    G=grow(A,ig1,removeqns(ig2)) #grow A into G as dense tensors
    ig1q=addqns(ig1,[QN()=>dim(ig1)];dir=dir(dag(ig2))) #make a QN version of index ig1
    @assert id(ig1)==id(ig1q) #If the ID changes then subsequent contractions will fail.
    return convert_blocksparse(G,ig1q,ig2) #fabricate a 1-block blocksparse version.
end
#
#  Convert a order 2 dense tensor into a single block, block-sparse tensor
#  using provided QNIndexes.  
#  TODO: There is probably a better way to do this without any risk of doing A
#  deep copy.
#
function convert_blocksparse(A::ITensor,inds::QNIndex...)
    @assert order(A)==2 #required for Block(1,1) to be correct.
    bst=BlockSparseTensor(eltype(A),[Block(1,1)],inds )
    b=nzblocks(bst)[1]
    blockview(bst, b) .= A #is this deep copy???
    return itensor(bst)
end

#
#  One of the sample_inds needs to match one of inds(A).  This provides enough
#  info to establish QN() space and direction for the inds of A.
#
function make_qninds(A::ITensor,sample_inds::Index...)
    @assert order(A)==2
    @assert !hasqns(A)
    @assert hasqns(sample_inds)
    ic=commonind(inds(A),sample_inds)
    ins =noncommonind(ic,sample_inds)
    inA =noncommonind(ic,inds(A))
    ics =noncommonind(ins,sample_inds)
    @assert hasqns(ics)
    #can't use space(ins) to get QNs because dim could be different.
    in=addqns(inA,[QN()=>dim(inA)];dir=dir(ins)) #make a QN version of index inA
    iset=IndexSet(in, ics)
    if inds(A) != iset
        iset = ITensors.permute(iset, inds(A))
    end
    return iset
end
