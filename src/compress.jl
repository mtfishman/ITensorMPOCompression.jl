
#
#  factor LR such that for
#       lr=left  LR=M*RM_prime
#       lr=right LR=RL_primt*M
#
function getM(RL::ITensor,lr::orth_type)::Tuple{ITensor,ITensor,Index}
    ils=filterinds(inds(RL),tags="Link")
    Dw1,Dw2=map(dim,ils)
    Dwm=min(Dw1,Dw2)
    im=Index(Dwm,"Link,m") #common link for M_plus
    if (lr==left)
        im1=Index(Dw1-2,tags(ils[1]))
        im2=Index(Dwm-2,tags(im))
        ir1=im
        ir2=ils[2]
    else # lr must be right
        im1=Index(Dwm-2,tags(im))
        im2=Index(Dw2-2,tags(ils[2]))
        ir1=ils[1]
        ir2=im
    end
    M=ITensor(im1,im2)
    for j1 in 2:Dw1-1
        for j2 in 2:Dw2-1
            M[im1=>j1-1,im2=>j2-1]=RL[ils[1]=>j1,ils[2]=>j2]
        end
    end
    #
    # Now we need RL_prime such that RL=M*RL_prime.
    # RL_prime is just the perimeter of RL with 1's on the diagonal
    #
    RL_prime=ITensor(0.0,ir1,ir2)
    for j1 in 1:dim(ir1)
        RL_prime[ir1=>j1,ir2=>1       ]=RL[ils[1]=>j1,ils[2]=>1  ]
        RL_prime[ir1=>j1,ir2=>dim(ir2)]=RL[ils[1]=>j1,ils[2]=>Dw2]
        RL_prime[ir1=>j1,ir2=>j1]=1.0
    end
    for j2 in 1:dim(ir2)
        RL_prime[ir1=>1       ,ir2=>j2]=RL[ils[1]=>1  ,ils[2]=>j2]
        RL_prime[ir1=>dim(ir1),ir2=>j2]=RL[ils[1]=>Dw1,ils[2]=>j2]
    end
    RL_prime[ir1=>dim(ir1),ir2=>dim(ir2)]=1.0

    return M,RL_prime,im
end

function grow(ig1::Index,A::ITensor,ig2::Index)
    ils=inds(A)
    @assert length(ils)==order(A)
    chi1,chi2=map(dim,ils)
#    @show tags(ig2) tags(ils[2])
    @assert dim(ig1)==chi1+2
    @assert tags(ig1)==tags(ils[1])
    @assert dim(ig2)==chi2+2
    @assert tags(ig2)==tags(ils[2])
    G=ITensor(0.0,ig1,ig2) #would be nice to use delta() but we can't set elements on it.
    G[ig1=>1,ig2=>1]=1.0;
    G[ig1=>chi1+2,ig2=>chi2+2]=1.0;
    for j1 in 1:chi1
        for j2 in 1:chi2
            G[ig1=>j1+1,ig2=>j2+1]=A[ils[1]=>j1,ils[2]=>j2]
        end
    end
    return G
end

function compress(W::ITensor,lr::orth_type,epsSVD::Float64)::Tuple{ITensor,ITensor}
    @assert lr==left  || lr==right
    d,n,r,c=parse_links(W) # links: r(l=$(n-1)), c(l=$n)
    eps=1e-14
    W,Lplus,lq=block_qx(W,lr) #W becomes Q with l=$(n-1), lq links
    @assert is_lower(lq,Lplus,c,eps)
    M,L_prime,im=getM(Lplus,lr) #M has ql, l=$n links with dim Dw1-2 x Dw2-2
  

    U,s,V=svd(M,inds(M)[1],cutoff=epsSVD) # ns sing. values survive compression
    ns=dim(inds(s)[1])
    @assert ns==dim(inds(s)[2]) #s should be square
    if lr==left
        lu=filterinds(inds(U),tags="u")[1]
        lv=filterinds(inds(V),tags="v")[1]
        #lq=filterinds(inds(U),tags="ql")[1]
        sV=s*V # Links u, l=$n  with dim ns x Dw2-2
        lu=Index(ns+2,"Link,u")
        sV=grow(lu,sV,im) #Links u,l=$n with dim ns->ns+2, chi2=Dw2-2->Dw2
        @assert is_lower(im,L_prime,c,eps)
        Lplus=sV*L_prime #links u,ql with dim ns+2 x Dw2
        # W has links l=$(n-1), l=$n with dim Dw1,Dw2 + site indices
        # U has links ql,u with dim Dw1-2,ns
        Uplus=grow(lq,U,lu)
        replaceind!(W,c,lq)
        replacetags!(Lplus,"u","l=$n")
        replacetags!(Uplus,"u","l=$n")
        @assert is_lower_regular_form(W,eps)
        W=W*Uplus
        @assert is_lower_regular_form(W,eps)
    elseif lr==right
        Us=U*s
        Us=grow(Us)
        Lplus=Lplus*Us
        Q=grow(V)*W
    else
        @assert(false) 
    end
    return W,Lplus
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