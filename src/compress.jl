
#
#  factor LR such that for
#       lr=left  LR=M*RM_prime
#       lr=right LR=RL_primt*M
#
function getM(RL::ITensor,lr::orth_type)::Tuple{ITensor,ITensor,Index}
    ils=filterinds(inds(RL),tags="Link")
    iqx=findinds(ils,"qx")[1]
    iln=noncommonind(ils,iqx)
    Dwq,Dwn=dim(iqx),dim(iln)
    Dwm=min(Dwq,Dwn)
    irm=Index(Dwm,"Link,m") #common tag between M and RL_prime
    imq=Index(Dwq-2,tags(iqx)) #mini version of iqx
    imm=Index(Dwm-2,tags(irm)) #mini version of irm
    M=ITensor(imq,imm)
    for j1 in 2:Dwq-1
        for j2 in 2:Dwm-1
            M[imq=>j1-1,imm=>j2-1]=RL[iqx=>j1,iln=>j2]
        end
    end
    #
    # Now we need RL_prime such that RL=M*RL_prime.
    # RL_prime is just the perimeter of RL with 1's on the diagonal
    #
    RL_prime=ITensor(0.0,irm,iln)
    for j1 in 1:dim(irm)
        RL_prime[irm=>j1,iln=>1       ]=RL[iqx=>j1,iln=>1  ]
        RL_prime[irm=>j1,iln=>dim(iln)]=RL[iqx=>j1,iln=>Dwn]
        RL_prime[irm=>j1,iln=>j1]=1.0
    end
    for j2 in 1:dim(iln)
        RL_prime[irm=>1       ,iln=>j2]=RL[iqx=>1  ,iln=>j2]
        RL_prime[irm=>dim(irm),iln=>j2]=RL[iqx=>Dwq,iln=>j2]
    end
    RL_prime[irm=>dim(irm),iln=>dim(iln)]=1.0

    return M,RL_prime,irm
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

#
#  Compress one site
#
function compress(W::ITensor,ms::matrix_state,epsSVD::Float64)::Tuple{ITensor,ITensor}
    d,n,r,c=parse_links(W) # W[l=$(n-1)l=$n]=W[r,c]
    eps=1e-14 #relax used for testing upper/lower/regular-form etc.
    # establish some tag strings then depend on lr.
    (tsvd,tuv,tln) = ms.lr==left ? ("qx","u","l=$n") : ("m","v","l=$(n-1)")
#
# Block repecting QR/QL/LQ/RQ factorization.  RL=L or R for upper and lower.
#
    Q,RL,lq=block_qx(W,ms) #left Q[r,qx], RL[qx,c] - right RL[r,qx] Q[qx,c]
    @assert is_canonical(Q,ms,eps)
    @assert is_regular_form(Q,ms.ul,eps)
#
#  Factor RL=M*L' (left/lower) = L'*M (right/lower) = M*R' (left/upper) = R'*M (right/upper)
#
    M,RL_prime,im=getM(RL,ms.lr) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
#
#  At last we can svd and compress M using epsSVD as the cutoff.
#    
    isvd=findinds(M,tsvd)[1] #decide the left index
    U,s,V=svd(M,isvd,cutoff=epsSVD) # ns sing. values survive compression
    ns=dim(inds(s)[1])
    
    luv=Index(ns+2,"Link,$tuv") #link for expanded U,US,V,sV matricies.
    if ms.lr==left
        @assert is_upper_lower(im,RL_prime,c,ms.ul,eps)
        @assert is_upper_lower(lq,RL      ,c,ms.ul,eps)
        RL=grow(s*V,luv,im)*RL_prime #RL[l=n,u] dim ns+2 x Dw2
        W=Q*grow(U,lq,luv) #W[l=n-1,u]
    else # right
        @assert is_upper_lower(r,RL_prime,im,ms.ul,eps)
        @assert is_upper_lower(r,RL,lq,ms.ul,eps)
        RL=RL_prime*grow(U*s,im,luv) #RL[l=n-1,v] dim Dw1 x ns+2
        W=grow(V,lq,luv)*Q #W[l=n-1,v]
    end
    replacetags!(RL,tuv,tln) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
    replacetags!(W ,tuv,tln) #W[l=n-1,l=n]
    @assert is_regular_form(W,ms.ul,eps)
    @assert is_canonical(W,ms,eps)
    return W,RL
end

#
#  Compress MPO
#
function compress!(H::MPO,ms::matrix_state,epsSVD::Float64)
    eps=1e-14 #relax used for testing upper/lower/regular-form etc.
    N=length(H)
    if ms.lr==left
        @assert is_canonical(H,mirror(ms),eps) #TODO we need not(ms.lr)
        for n in 1:N-1 #sweep right
            W,RL=compress(H[n],ms,epsSVD)
            if (epsSVD==0)
                @assert norm(H[n]-W*RL)<eps
            end
            H[n]=W
            H[n+1]=RL*H[n+1]
            is_regular_form(H[n+1],ms.ul,eps)
        end
    else #lr must be right
        @assert is_canonical(H,mirror(ms),eps)#TODO we need not(ms.lr)
        for n in N:2 #sweep left
            W,RL=compress(H[n],ms,epsSVD)
            if (epsSVD==0)
                @assert norm(H[n]-W*RL)<eps
            end
            H[n]=W
            H[n-1]=H[n]*RL
            is_regular_form(H[n-1],ms.ul,eps)
        end
    end
end