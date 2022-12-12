
using Printf

#
#  Compress one site
#
function truncate(W::ITensor,ul::reg_form;kwargs...)::Tuple{ITensor,ITensor,bond_spectrum}
    lr::orth_type=get(kwargs, :orth, right)
    ms=matrix_state(ul,lr)
    eps=1e-14 #relax used for testing upper/lower/regular-form etc.
    forward,reverse=parse_links(W,lr) # W[l=$(n-1)l=$n]=W[r,c]
    d,n,space=parse_site(W)
    # establish some tag strings then depend on lr.
    (tsvd,tuv) = lr==left ? ("qx","Link,u") : ("m","Link,v")
#
# Block repecting QR/QL/LQ/RQ factorization.  RL=L or R for upper and lower.
# here we purposely turn off rank reavealing feature (epsrr=0.0) to (mostly) avoid
# horizontal rectangular RL matricies which are hard to handle accurately.
#
   
    Q,RL,lq=block_qx(W,ul;epsrr=-1.0,kwargs...) #left Q[r,qx], RL[qx,c] - right RL[r,qx] Q[qx,c]
    ITensors.@debug_check begin
        if order(Q)==4
            @assert is_canonical(Q,ms,eps)
            @assert is_regular_form(Q,ul,eps)
        end
    end
    c=noncommonind(RL,lq) #if size changed the old c is not lnger valid
    #
    #  If the RL is rectangular in wrong way, then factoring out M is very difficult.
    #  For now we just bail out.
    #
    if dim(c)>dim(lq)
        replacetags!(RL,"Link,qx",tags(forward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
        replacetags!(Q ,"Link,qx",tags(forward)) #W[l=n-1,l=n]
        return Q,RL,bond_spectrum(n)
    end
    RLinds=inds(RL) # we will need the QN space info later to reconstruct a block sparse RL.
    
#
#  Factor RL=M*L' (left/lower) = L'*M (right/lower) = M*R' (left/upper) = R'*M (right/upper)
#  For blocksparse W, at this point we switch to dense for all RL manipulations and RL should
#  only have one block anyway.
#  TODO: use multiple dispatch on getM to get all QN specific code out of this funtion.
#
    if (hasqns(RL))
        @assert nnzblocks(RL)==1
    end
   
    M,RL_prime,im,RLnz=getM(dense(RL),ms,eps) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
    @assert RLnz==0 #make RL_prime does not require any fix ups.
#  
#  At last we can svd and compress M using epsSVD as the cutoff.  M should be dense.
#    
    isvd=findinds(M,tsvd)[1] #decide the left index
    U,s,V=svd(M,isvd;kwargs...) # ns sing. values survive compression
    ns=dim(inds(s)[1])

    spectrum=bond_spectrum(s,n)
    #@show diag(array(s))
    Mplus=grow(M,removeqns(lq),im)
    D=dense(RL)-Mplus*RL_prime
    #@show norm(D)
    # Check accuracy of RL_prime.
    if norm(D)>get(kwargs, :cutoff, 1e-14)
        @printf "High normD(D)=%.1e min(s)=%.1e \n" norm(D) Base.min(diag(array(s))...)
        replacetags!(RL,"Link,qx",tags(forward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
        replacetags!(Q ,"Link,qx",tags(forward)) #W[l=n-1,l=n]
        return Q,RL,spectrum
    end
   
    luv=Index(ns+2,"Link,$tuv") #link for expanded U,Us,V,sV matricies.
    if lr==left
        RL=grow(s*V,luv,im)*RL_prime #RL[l=n,u] dim ns+2 x Dw2
        Uplus=grow(U,dag(lq),luv)
        if hasqns(lq)
            @assert hasqns(Uplus)
        end
        W=Q*Uplus #W[l=n-1,u]
    else # right
        RL=RL_prime*grow(U*s,im,luv) #RL[l=n-1,v] dim Dw1 x ns+2
        
        Vplus=grow(V,dag(lq),luv) #lq has the dir of Q so want the opposite on Vplus
        if hasqns(lq)
            @assert hasqns(Vplus)
        end
        W=Vplus*Q #W[l=n-1,v]
    end
    replacetags!(RL,tuv,tags(forward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
    replacetags!(W ,tuv,tags(forward)) #W[l=n-1,l=n]
    # At this point RL is dense, we need to make block-sparse version with one block.
    if hasqns(RLinds)
        iRL=make_qninds(RL,RLinds...)
        RL=convert_blocksparse(RL,iRL...)
    end
    ITensors.@debug_check begin
        @assert is_regular_form(W,ul,eps)
        @assert is_canonical(W,ms,eps)
    end
    return W,RL,spectrum
end

@doc """
    truncate!(H::MPO)

Compress an MPO using block respecting SVD techniques as described in 
> *Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147*

# Arguments
- `H` MPO for decomposition. If H is not already in the correct canonical form for compression, it will automatically be put into the correct form prior to compression.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form for the final output. 
- `cutoff::Float64 = 1e-14` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
- `maxdim::Int64` : If the number of singular values exceeds `maxdim`, only the largest `maxdim` will be retained.
- `mindim::Int64` : At least `mindim` singular values will be retained, even if some fall below the cutoff

# Example
```julia
julia> using ITensors
julia> using ITensorMPOCompression
julia> N=10; #10 sites
julia> NNN=7; #Include up to 7th nearest neighbour interactions
julia> sites = siteinds("S=1/2",N);
#
# This makes H directly, bypassing autoMPO.  (AutoMPO is too smart for this
# demo, it makes maximally reduced MPOs right out of the box!)
#
julia> H=make_transIsing_MPO(sites,NNN);
#
#  Make sure we have a regular form or truncate! won't work.
#
julia> is_lower_regular_form(H)==true
true

#
#  Now we can truncate with defaults of left orthogonal cutoff=1e-14.
#  truncate! returns the spectrum of singular values at each bond.  The largest
#  singular values are remaining well under control.  i.e. no sign of divergences.
#
julia> truncate!(H)
9-element Vector{bond_spectrum}:
 bond_spectrum([0.307], 1)
 bond_spectrum([0.354, 0.035], 2)
 bond_spectrum([0.375, 0.045, 0.021], 3)
 bond_spectrum([0.385, 0.044, 0.026, 0.018], 4)
 bond_spectrum([0.388, 0.043, 0.031, 0.019, 0.001], 5)
 bond_spectrum([0.385, 0.044, 0.026, 0.018], 6)
 bond_spectrum([0.375, 0.045, 0.021], 7)
 bond_spectrum([0.354, 0.035], 8)
 bond_spectrum([0.307], 9)

julia> pprint(H[2])
I 0 0 0 
S S S 0 
0 S S I 

#
#  We can see that bond dimensions have been drastically reduced.
#
julia> get_Dw(H)
9-element Vector{Int64}: 3 4 5 6 7 6 5 4 3

julia> is_lower_regular_form(H)==true
true

julia> is_orthogonal(H,left)==true
true

```
"""
function ITensors.truncate!(H::MPO;kwargs...)::bond_spectrums
    #@printf "---- start compress ----\n"
    #
    # decide left/right and upper/lower
    #
    lr::orth_type=get(kwargs, :orth, left)
    kwargs=add_or_replace(kwargs,:orth,lr) #if lr is not yet in kwargs, we need to stuff in there
    (bl,bu)=detect_regular_form(H)
    if !(bl || bu)
        throw(ErrorException("truncate!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    #
    # Now check if H required orthogonalization
    #
    ms=matrix_state(ul,lr)
    if !is_canonical(H,mirror(ms))
        epsrr=get(kwargs,:epsrr,1e-12)
        epsrr= epsrr==0 ? 1e-12 : epsrr #0.0 not allwed here.
        orthogonalize!(H,epsrr=epsrr,orth=mirror(lr)) #TODO why fail if spec ul here??
    end
    N=length(H)
    ss=bond_spectrums(undef,N-1)
     #TODO use sweep here
    if lr==left
        rng=1:1:N-1 #sweep left to right
        link_offest=0
    else #right
        rng=N:-1:2 #sweep right to left
        link_offest=-1
    end
    for n in rng 
        nn=n+rng.step #index to neighbour
        W,RL,s=truncate(H[n],ul;kwargs...)
        H[n]=W
        H[nn]=RL*H[nn]
        is_regular_form(H[nn],ms.ul)
        ss[n+link_offest]=s
    end
    return ss
end

function ITensors.truncate!(H::InfiniteMPO;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums}
    #@printf "---- start compress ----\n"
    #
    # decide left/right and upper/lower
    #
    lr::orth_type=get(kwargs, :orth, left) #this specifies the final output orth direction.
    kwargs=add_or_replace(kwargs,:orth,lr) #if lr is not yet in kwargs, we need to stuff in there
    (bl,bu)=detect_regular_form(H)
    if !(bl || bu)
        throw(ErrorException("truncate!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    #
    # Now check if H required orthogonalization
    #
    ms=matrix_state(ul,lr)
    if !is_canonical(H,ms)
        #@show inds(H[1],tags="Link")
        orthogonalize!(H;orth=mirror(lr)) 
        @assert is_orthogonal(H,mirror(lr))
        #@show inds(H[1],tags="Link")
        Gs=orthogonalize!(H;orth=lr) #TODO why fail if spec ul here??
        @assert is_orthogonal(H,lr)
        #@show inds(H[1],tags="Link")
        #@show lr Gs
    else
        # user supplied canonical H but not the Gs so we cannot proceed unless we do one more
        # wasteful sweeps
        @assert false #for now.
    end
    N=length(H)
    ss=bond_spectrums(undef,N)
    Ss=CelledVector{ITensor}(undef,N)
    #TODO use sweep here
    if lr==left
        rng=1:1:N #sweep left to right
        link_offest=0
    else #right
        rng=1,1,N #sweep right to left
        link_offest=0
    end
    for n in 1:N 
        if lr==left
            il,ir=parse_links(H[n]) #right link of H is the left link of G
            #@show il ir inds(Gs[n])
            U,s,V,Sp=truncate(Gs[n],ir;kwargs...)
            #@show diag(array(s))
            #@show inds(H[n],tags="Link") inds(H[n+1],tags="Link") inds(U)
            H[n]=H[n]*U
            @assert order(H[n])==4
            H[n+1]=dag(U)*H[n+1]
            @assert order(H[n+1])==4
        else
            il,ir=parse_links(H[n]) #left link of H[n] is the right link of G[n-1]
            igl=noncommonind(Gs[n-1],il)
            U,s,V,Sp=truncate(Gs[n-1],igl;kwargs...)
            #@show diag(array(s))
            #@show inds(H[n],tags="Link") inds(V)
            H[n]=H[n]*dag(V)
            @assert order(H[n])==4
            #@show inds(H[n-1],tags="Link") inds(V)
            H[n-1]=V*H[n-1]
            @assert order(H[n-1])==4
        end
        Ss[n]=Sp
        ss[n+link_offest]=bond_spectrum(s,n)
    end
    return Ss,ss
end

function truncate(G::ITensor,igl::Index;kwargs...)
    igr=noncommonind(G,igl)
    M,iml=getM(G,igl,igr)
    U,s,V=svd(M,iml;kwargs...)
    iu=commonind(U,s)
    iv=commonind(V,s)
   
    iup=redim(iu,dim(iu)+2) #Index(dim(iu)+2,tags(iu))
    ivp=redim(iv,dim(iv)+2) #Index(dim(iv)+2,tags(iv))
    #lr::orth_type=get(kwargs, :orth, left) 
    Up=grow(U,igl,iup)
    Sp=grow(s,iup,ivp)
    Vp=grow(V,ivp,igr)
    replacetags!(Up,tags(iu),tags(igl))
    replacetags!(Sp,tags(iu),tags(igl))
    replacetags!(Sp,tags(iv),tags(igr))
    replacetags!(Vp,tags(iv),tags(igr))
    @assert norm(G-Up*Sp*Vp)<1e-12    
    return Up,s,Vp,Sp
end


