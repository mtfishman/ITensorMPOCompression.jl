
using LinearAlgebra
using Printf
#
#  factor LR such that for
#       lr=left  LR=M*RM_prime
#       lr=right LR=RL_primt*M
#  However becuase of the ITensor index work we don;t need to distinguish between left and 
#  matrix multiplication in the code.  BUT we do need to worry about upper and lower RL
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
    Dwm=min(Dwq,Dwn)
    irm=Index(Dwm,"Link,m") #new common index between Mplus and RL_prime
    imq=Index(Dwq-2,tags(iqx)) #mini version of iqx
    imm=Index(Dwm-2,tags(irm)) #mini version of irm
    M=ITensor(imq,imm)
    shift=0
    if ms.ul==upper
        shift=max(0,Dwn-Dwq) #for upper rectangular R we want M over at the right
    end
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

function getInverse(U::ITensor,s::ITensor,V::ITensor)::ITensor
    as=LinearAlgebra.diag(array(s))
    asinv=Vector(as)
    for ia in 1:length(asinv) asinv[ia]=1.0/asinv[ia] end
    sinv=diagITensor(eltype(asinv),asinv,inds(s))
    Minv=dag(V)*dag(sinv)*dag(U)
    return Minv
end

#
# Solve RL=M*RL_prime we only need to solve the last few columns to the right of
# the unit matrix inside RL_prime
#
function SolveRLprime(RL::ITensor,RL_prime::ITensor,U::ITensor,s::ITensor,V::ITensor,iq::Index,im::Index,ms::matrix_state)::ITensor
    Minv=getInverse(U,s,V)
    ic1=noncommonind(RL_prime,im) #RL_prime col index
    ic2=noncommonind(RL      ,iq) #RL       col index
    @assert ic1==ic2
    @assert dim(iq)==dim(im)
    #eps=1e-14
    #@show "RL="
    #pprint(iq,RL,ic1,eps)
    #@show "RL_prime="
    #pprint(im,RL_prime,ic1,eps)
    Dm,Dc=dim(im),dim(ic1)
    @assert Dm>2
    @assert Dc>Dm
    imm=filterinds(Minv,tags=tags(im))[1]
    iqm=filterinds(Minv,tags=tags(iq))[1]
    @assert dim(imm)==dim(im)-2
    @assert dim(iqm)==dim(iq)-2
    icm=Index(Dc-Dm,tags(ic1))
    R2=ITensor(0.0,iqm,icm)
    j1_offset=1
    if ms.ul==lower
        j2_offset=Dm-1
    else #upper
        j2_offset=1
    end
    #@show ms Dm j2_offset

    for j2 in eachindval(icm)
        for j1 in eachindval(iqm)
            R2[j1,j2]=RL[iq=>j1.second+j1_offset,ic1=>j2.second+j2_offset]
        end
    end
    #pprint(iqm,R2,icm,eps)
    #@assert(false)
    R2_prime=Minv*R2
    
    #@show inds(R2_prime)
    #pprint(imm,R2_prime,icm,eps)
    for j2 in eachindval(icm)
        for j1 in eachindval(imm)
            RL_prime[im=>j1.second+j1_offset,ic1=>j2.second+j2_offset]=R2_prime[j1,j2]
        end
    end
    #pprint(im,RL_prime,ic1,eps)

    return RL_prime
end
#
#  Compress one site
#
function compress(W::ITensor,ul::tri_type;kwargs...)::Tuple{ITensor,ITensor}
    d,n,r,c=parse_links(W) # W[l=$(n-1)l=$n]=W[r,c]
    lr::orth_type=get(kwargs, :dir, right)
    ms=matrix_state(ul,lr)
    eps=1e-14 #relax used for testing upper/lower/regular-form etc.
    # establish some tag strings then depend on lr.
    (tsvd,tuv,tln) = lr==left ? ("qx","u","l=$n") : ("m","v","l=$(n-1)")

#
# Block repecting QR/QL/LQ/RQ factorization.  RL=L or R for upper and lower.
#
    Q,RL,lq=block_qx(W,ul;kwargs...) #left Q[r,qx], RL[qx,c] - right RL[r,qx] Q[qx,c]
    @assert is_canonical(Q,ms,eps)
    @assert is_regular_form(Q,ul,eps)
#
#  Factor RL=M*L' (left/lower) = L'*M (right/lower) = M*R' (left/upper) = R'*M (right/upper)
#
    c=noncommonind(RL,lq) #if size changed the old c is not lnger valid
    #@show inds(RL),lq,c,"RL="
    #pprint(lq,RL,c,eps)
    M,RL_prime,im,RLnz=getM(RL,ms,eps) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
    
    # imm=filterinds(M,tags="m")[1]
    # imq=filterinds(M,tags="qx")[1]
    # pprint(imq,M,imm,eps)
    
    
    #
#  At last we can svd and compress M using epsSVD as the cutoff.
#    
    isvd=findinds(M,tsvd)[1] #decide the left index
    U,s,V=svd(M,isvd;kwargs...) # ns sing. values survive compression
    ns=dim(inds(s)[1])
    #@show diag(array(s))
    
    # imm=filterinds(M,tags="m")[1]
    # imq=filterinds(M,tags="qx")[1]
    # Minv=getInverse(U,s,V)
    # Minvm=prime(Minv,imm)
    
    #@show inds(Minvm) inds(Minv) inds(M)
    # @printf "Minv*M-I norm = %.1e \n" norm(prime(Minv,imm)*M-delta(imq,imq'))
    
    #@assert false
    #
    #  If RL is rectangular we need to solve RL=M*RL_prime for RL_prime
    #  since we know UsV anyway we can calculate M^-1=dag(V)*1.0/s*dag(U)
    #
    if ns>0 && RLnz
        #@show "fixing RL_prime" inds(RL_prime) inds(RL)
        RL_prime=SolveRLprime(RL,RL_prime,U,s,V,lq,im,ms)
    end
    Mplus=grow(M,lq,im)
    D=RL-Mplus*RL_prime
    if norm(D)>get(kwargs, :cutoff, 1e-14)
        #@show  RLnz "RL_prime="
        #pprint(im,RL_prime,c,eps)
        @printf "High normD(D)=%.1e min(s)=%.1e \n" norm(D) min(diag(array(s))...)
        #pprint(lq,D,c,eps)
        #@show D
        #@assert(false)
    else
#        @printf "normD(D)=%.1e min(s)=%.1e \n" norm(D) min(diag(array(s))...)
    end

    luv=Index(ns+2,"Link,$tuv") #link for expanded U,US,V,sV matricies.
    if lr==left
#        @assert is_upper_lower(im,RL_prime,c,ms.ul,eps)
        @assert is_upper_lower(lq,RL      ,c,ul,eps)
        RL=grow(s*V,luv,im)*RL_prime #RL[l=n,u] dim ns+2 x Dw2
        W=Q*grow(U,lq,luv) #W[l=n-1,u]
    else # right
#        @assert is_upper_lower(r,RL_prime,im,ms.ul,eps)
        @assert is_upper_lower(r,RL      ,lq,ms.ul,eps)
        RL=RL_prime*grow(U*s,im,luv) #RL[l=n-1,v] dim Dw1 x ns+2
        W=grow(V,lq,luv)*Q #W[l=n-1,v]
    end
    replacetags!(RL,tuv,tln) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
    replacetags!(W ,tuv,tln) #W[l=n-1,l=n]
    @assert is_regular_form(W,ul,eps)
    @assert is_canonical(W,ms,eps)
    return W,RL
end

"""
    truncate!(H::MPO)

Compress an MPO using block respecting SVD techniques as described in 
> *Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147*

# Arguments
- `H` MPO for decomposition. If H is not already in the correct canonical form for compression, it will automatically be put into the correct form prior to compression.

# Keywords
- `dir::orth_type = right` : choose `left` or `right` canonical form for the final output. 
- `cutoff::Foat64` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
- `maxdim::Int64` : If the number of singular values exceeds `maxdim`, only the largest `maxdim` will be retained.
- `mindim::Int64` : At least `mindim` singular values will be retained, even if some fall below the cutoff

"""
function truncate!(H::MPO;kwargs...)
    #
    # decide left/right and upper/lower
    #
    eps=1e-14 #relax used for testing upper/lower/regular-form etc.
    lr::orth_type=get(kwargs, :dir, right)
    (bl,bu)=detect_regular_form(H,eps)
    if !(bl || bu)
        throw(ErrorException("truncate!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::tri_type = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    #
    # Now check if H required orthogonalization
    #
    ms=matrix_state(ul,lr)
    if !is_canonical(H,mirror(ms),eps) 
        orthogonalize!(H,ul;dir=mirror(lr),kwargs...) 
    end
    N=length(H)
    if ms.lr==left
        @assert is_canonical(H,mirror(ms),eps) 
        for n in 1:N-1 #sweep right
            W,RL=compress(H[n],ul;kwargs...)
            #@show norm(H[n]-W*RL)
            H[n]=W
            H[n+1]=RL*H[n+1]
            is_regular_form(H[n+1],ms.ul,eps)
        end
    else #lr must be right
        @assert is_canonical(H,mirror(ms),eps)#TODO we need not(ms.lr)
        for n in N:-1:2 #sweep left
            W,RL=compress(H[n],ul;kwargs...)
            #@show norm(H[n]-W*RL)
            H[n]=W
            H[n-1]=H[n-1]*RL
            is_regular_form(H[n-1],ms.ul,eps)
        end
    end
end