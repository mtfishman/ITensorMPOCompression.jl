
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
    im=Index(Dwm,"Link,m") #common link for M_plus
    if (lr==left)
        #@show "left" iqx iln
        imq=Index(Dwq-2,tags(iqx))
        imm=Index(Dwm-2,tags(im)) #mini version of im
        irm=im
        irn=iln
    else # lr must be right
        #@show "right"  iqx iln
        imm=Index(Dwm-2,tags(im))
        imq=Index(Dwq-2,tags(iqx))
        irn=iln
        irm=im
    end
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
    RL_prime=ITensor(0.0,irm,irn)
    for j1 in 1:dim(irm)
        RL_prime[irm=>j1,irn=>1       ]=RL[iqx=>j1,iln=>1  ]
        RL_prime[irm=>j1,irn=>dim(irn)]=RL[iqx=>j1,iln=>Dwn]
        RL_prime[irm=>j1,irn=>j1]=1.0
    end
    for j2 in 1:dim(irn)
        RL_prime[irm=>1       ,irn=>j2]=RL[iqx=>1  ,iln=>j2]
        RL_prime[irm=>dim(irm),irn=>j2]=RL[iqx=>Dwq,iln=>j2]
    end
    RL_prime[irm=>dim(irm),irn=>dim(irn)]=1.0

    return M,RL_prime,im
end

function grow(ig1::Index,A::ITensor,ig2::Index)
    ils=inds(A)
    @assert length(ils)==order(A)
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
    G[ig1=>1,ig2=>1]=1.0;
    G[ig1=>chi1+2,ig2=>chi2+2]=1.0;
    for j1 in 1:chi1
        for j2 in 1:chi2
            G[ig1=>j1+1,ig2=>j2+1]=A[ia1=>j1,ia2=>j2]
        end
    end
    return G
end

function compress(W::ITensor,lr::orth_type,epsSVD::Float64)::Tuple{ITensor,ITensor}
    @assert lr==left  || lr==right
    d,n,r,c=parse_links(W) # W[l=$(n-1)l=$n]=W[r,c]
    eps=1e-14
    Q,RL,lq=block_qx(W,lr) #left Q[r,lq], RL[lq,c] - right RL[r,lq] Q[lq,c]
    @assert is_canonical(Q,matrix_state(lower,lr),eps)
    @assert is_lower_regular_form(Q,eps)
    @show inds(RL)

    M,L_prime,im=getM(RL,lr) #left M[lq,im] L_prime[im,c] - right L_prime[r,im] M[im,lq]
    @show inds(M) inds(L_prime)
    U,s,V=svd(M,inds(M)[1],cutoff=epsSVD) # ns sing. values survive compression
    ns=dim(inds(s)[1])
    @assert ns==dim(inds(s)[2]) #s should be square
    
    if lr==left
        @assert is_lower(im,L_prime,c,eps)
        @assert is_lower(lq,RL,c,eps)
        ln=Index(ns+2,"Link,l=$n")
        RL=grow(ln,s*V,im)*L_prime #RL[l=n,l=n] dim ns+2 x Dw2
        W=Q*grow(lq,U,ln) #W[l=n-1,l=n]
    elseif lr==right
        @assert is_lower(lq,L_prime,im,eps)
        @assert is_lower(r,RL,lq,eps)
        ln=Index(ns+2,"Link,l=$(n-1)")
        @show inds(L_prime)
        RL=L_prime*grow(im,U*s,ln) #RL[l=n-1,l=n-1] dim Dw1 x ns+2
        #@show ln inds(V) lq
        W=grow(lq,V,ln)*Q #W[l=n-1,l=n]
    else
        @assert(false) 
    end
    @assert is_lower_regular_form(W,eps)
    @assert is_canonical(W,matrix_state(lower,lr),eps)
return W,RL
end

function compress!(H::MPO,lr::orth_type,epsSVD::Float64)
    @assert lr==left  || lr==right
    eps=1e-14
    N=length(H)
    if lr==left
        @assert is_canonical(H,matrix_state(lower,right),eps)
        for n in 1:N-1
            W,L=compress(H[n],lr,epsSVD)
 #           pprint(W,eps)
            @assert is_lower_regular_form(W,eps)
            @assert norm(H[n]-W*L)<eps
            H[n]=W
            H[n+1]=L*H[n+1]
            is_lower_regular_form(H[n+1],eps)
        end
    else #lr must be right
        @assert is_canonical(H,matrix_state(lower,left),eps)
        for n in N:2
            W,R=compress(H[n],lr,epsSVD)
            @assert detect_upper_lower(H,eps)==lower
            @assert detect_upper_lower(W,eps)==lower
            @assert norm(H[n]-R*W)<eps
            H[n]=W
            H[n-1]=H[n]*R
        end
    end
end