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
    @mpoc_assert inds(W,tags="Site")==inds(V,tags="Site")
    off=V_offsets(ms)
    @mpoc_assert(off.o1==off.o2)
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
#   1) V could have a truncted number of rows or columns as a result of rank revealing QX
#      W then has to be re-sized accordingly
#   2) If W gets resized we need to preserve particular rows/columns outside the Vblock from the old W.
#   3) V and W should have one common link index so we need to find those and pair them together. But we
#      need to match on tags&plev because dims are different so ID matching won't work. 
#
function setV(W::ITensor,V::ITensor,ms::matrix_state)::ITensor
    if order(W)==3
        return setV1(W,V,ms) #Handle row/col vectors
    end
    @mpoc_assert order(W)==4
    wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    vils=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    @mpoc_assert length(wils)==2
    @mpoc_assert length(vils)==2
    ivqx,=inds(V,tags="Link,qx") #get the qx index.  This is the one that can change size
    ivl=noncommonind(vils,ivqx)  #get the other link l=n index on V
    iwl=wils[match_tagplev(ivl,wils...)] #now find the W link index with the same tags/plev as ivl
    iwqx=noncommonind(wils,iwl) #get the other link l=n index on W
    off=V_offsets(ms)
    @mpoc_assert off.o1==off.o2
    #
    #  If rank reduction occured in the QR process we may need to resize W.
    #
    if dim(iwqx)>dim(ivqx)+1
        iw1qx=redim(iwqx,dim(ivqx)+1) #re-dimension the wqx index.
        iss=filterinds(inds(W),tags="Site")
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
    @mpoc_assert order(RL)==2
    irl,=filterinds(inds(RL),tags=tags(iwl)) #find the link index of RL
    irqx=noncommonind(inds(RL),irl) #find the qx link of RL
    @mpoc_assert dim(iwl)==dim(irl)+1
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


#                      |1 0 0|
#  given A, spit out G=|0 A 0| , indices of G are provided.
#                      |0 0 1|
#
function grow(A::ITensor,ig1::Index,ig2::Index)
    Dw1,Dw2=dim(ig1),dim(ig2)
    G=similar(A,ig1,ig2)
    G.=0.0
    #G=ITensor(Gt,ig1,ig2)
#    G=ITensor(0.0,ig1,ig2) #would be nice to use delta() but we can't set elements on it.
    G[ig1=>1  ,ig2=>1  ]=1.0;
    G[ig1=>Dw1,ig2=>Dw2]=1.0;
    G[ig1=>2:Dw1-1,ig2=>2:Dw2-1]=A
    return G
end

function grow(A::ITensor,ig1::QNIndex,ig2::Index{Int64})
    @mpoc_assert !hasqns(A)
    @mpoc_assert !hasqns(ig2)
    G=grow(A,removeqns(ig1),ig2) #grow A into G as dense tensors
    ig2q=addqns(ig2,[QN()=>dim(ig2)];dir=dir(dag(ig1))) #make a QN version of index ig2
    @mpoc_assert id(ig2)==id(ig2q) #If the ID changes then subsequent contractions will fail.
    return convert_blocksparse(G,ig1,ig2q) #fabricate a 1-block blocksparse version.
end
function grow(A::ITensor,ig1::Index{Int64},ig2::QNIndex)
    @mpoc_assert !hasqns(A)
    @mpoc_assert !hasqns(ig1)
    G=grow(A,ig1,removeqns(ig2)) #grow A into G as dense tensors
    ig1q=addqns(ig1,[QN()=>dim(ig1)];dir=dir(dag(ig2))) #make a QN version of index ig1
    @mpoc_assert id(ig1)==id(ig1q) #If the ID changes then subsequent contractions will fail.
    return convert_blocksparse(G,ig1q,ig2) #fabricate a 1-block blocksparse version.
end
# function grow(A::ITensor,ig1::QNIndex,ig2::QNIndex)
#     #@mpoc_assert !hasqns(A)
#     G=grow(A,ig1,ig2) #grow A into G as dense tensors
#     return convert_blocksparse(G,ig1,ig2) #fabricate a 1-block blocksparse version.
# end
#
#  Convert a order 2 dense tensor into a single block, block-sparse tensor
#  using provided QNIndexes.  
#  TODO: There is probably a better way to do this without any risk of doing A
#  deep copy.
#
function convert_blocksparse(A::ITensor,inds::QNIndex...)
    @mpoc_assert order(A)==2 #required for Block(1,1) to be correct.
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
    @mpoc_assert order(A)==2
#    @mpoc_assert !hasqns(A)
    @mpoc_assert hasqns(sample_inds)
    ic=commonind(inds(A),sample_inds)
    ins =noncommonind(ic,sample_inds)
    inA =noncommonind(ic,inds(A))
    ics =noncommonind(ins,sample_inds)
    @mpoc_assert hasqns(ics)
    #can't use space(ins) to get QNs because dim could be different.
    in=addqns(inA,[QN()=>dim(inA)];dir=dir(ins)) #make a QN version of index inA
    iset=IndexSet(in, ics)
    if inds(A) != iset
        iset = ITensors.permute(iset, inds(A))
    end
    return iset
end
