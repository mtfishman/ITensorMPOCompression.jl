
using LinearAlgebra
using Printf
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
function truncate(W::ITensor,ul::tri_type;kwargs...)::Tuple{ITensor,ITensor}
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
            W,RL=truncate(H[n],ul;kwargs...)
            #@show norm(H[n]-W*RL)
            H[n]=W
            H[n+1]=RL*H[n+1]
            is_regular_form(H[n+1],ms.ul,eps)
        end
    else #lr must be right
        @assert is_canonical(H,mirror(ms),eps)#TODO we need not(ms.lr)
        for n in N:-1:2 #sweep left
            W,RL=truncate(H[n],ul;kwargs...)
            #@show norm(H[n]-W*RL)
            H[n]=W
            H[n-1]=H[n-1]*RL
            is_regular_form(H[n-1],ms.ul,eps)
        end
    end
end