
using Printf

#
#  Compress one site
#
function truncate(W::ITensor,ul::reg_form;kwargs...)::Tuple{ITensor,ITensor,Spectrum}
    lr::orth_type=get(kwargs, :orth, right)
    ms=matrix_state(ul,lr)
    eps=1e-14 #relax used for testing upper/lower/regular-form etc.
    forward,reverse=parse_links(W,lr) # W[l=$(n-1)l=$n]=W[r,c]
    d,n,space=parse_site(W)
    # establish some tag strings then depend on lr.
    (tsvd,tuv) = lr==left ? ("qx","Link,u") : ("m","Link,v")
#
# Block repecting QR/QL/LQ/RQ factorization.  RL=L or R for upper and lower.
# here we purposely turn off rank reavealing feature (rr_cutoff=0.0) to (mostly) avoid
# horizontal rectangular RL matricies which are hard to handle accurately.
#
   
    Q,RL,lq=block_qx(W,ul;rr_cutoff=-1.0,kwargs...) #left Q[r,qx], RL[qx,c] - right RL[r,qx] Q[qx,c]
    if order(Q)==4
        @mpoc_assert is_canonical(Q,ms,eps)
        @mpoc_assert is_regular_form(Q,ul,eps)
    end
    c=noncommonind(RL,lq) #if size changed the old c is not lnger valid
    #
    #  If the RL is rectangular in wrong way, then factoring out M is very difficult.
    #  For now we just bail out.
    #
    if dim(c)>dim(lq) || dim(c)<3
        replacetags!(RL,"Link,qx",tags(forward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
        replacetags!(Q ,"Link,qx",tags(forward)) #W[l=n-1,l=n]
        return Q,RL,Spectrum([],0)
    end
    RLinds=inds(RL) # we will need the QN space info later to reconstruct a block sparse RL.
    
#
#  Factor RL=M*L' (left/lower) = L'*M (right/lower) = M*R' (left/upper) = R'*M (right/upper)
#  For blocksparse W, at this point we switch to dense for all RL manipulations and RL should
#  only have one block anyway.
#  TODO: use multiple dispatch on getM to get all QN specific code out of this funtion.
#
    if (hasqns(RL))
        @mpoc_assert nnzblocks(RL)==1
    end
   
    M,RL_prime,im,RLnz=getM(dense(RL),ms,eps) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
    @mpoc_assert RLnz==0 #make RL_prime does not require any fix ups.
#  
#  At last we can svd and compress M using epsSVD as the cutoff.  M should be dense.
#    
    isvd=findinds(M,tsvd)[1] #decide the left index
    U,s,V,spectrum=svd(M,isvd;kwargs...) # ns sing. values survive compression
    ns=dim(inds(s)[1])

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
            @mpoc_assert hasqns(Uplus)
        end
        W=Q*Uplus #W[l=n-1,u]
    else # right
        RL=RL_prime*grow(U*s,im,luv) #RL[l=n-1,v] dim Dw1 x ns+2
        
        Vplus=grow(V,dag(lq),luv) #lq has the dir of Q so want the opposite on Vplus
        if hasqns(lq)
            @mpoc_assert hasqns(Vplus)
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
    @mpoc_assert is_regular_form(W,ul,eps)
    @mpoc_assert is_canonical(W,ms,eps)
    return W,RL,spectrum
end

@doc """
    truncate!(H::MPO)

Compress an MPO using block respecting SVD techniques as described in 
> *Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147*

# Arguments
- `H` MPO for decomposition. If `H` is not already in the correct canonical form for compression, it will automatically be put into the correct form prior to compression.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form for the final output. 
- `cutoff::Float64 = 0.0` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
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
julia> @show truncate!(H);
site  Ns   max(s)     min(s)    Entropy  Tr. Error
   1    1  0.30739   3.07e-01   0.22292  0.00e+00
   2    2  0.35392   3.49e-02   0.26838  0.00e+00
   3    3  0.37473   2.06e-02   0.29133  0.00e+00
   4    4  0.38473   1.77e-02   0.30255  0.00e+00
   5    5  0.38773   7.25e-04   0.30588  0.00e+00
   6    4  0.38473   1.77e-02   0.30255  0.00e+00
   7    3  0.37473   2.06e-02   0.29133  0.00e+00
   8    2  0.35392   3.49e-02   0.26838  0.00e+00
   9    1  0.30739   3.07e-01   0.22292  0.00e+00

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
    (bl,bu)=detect_regular_form(H)
    if !(bl || bu)
        throw(ErrorException("truncate!(H::MPO), H must be in either lower or upper regular form"))
    end
    @mpoc_assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    
    verbose::Bool=get(kwargs, :verbose, false)
    if verbose
        previous_Dw=Base.max(get_Dw(H)...)
    end#
    # Now check if H required orthogonalization
    #
    ms=matrix_state(ul,lr)
    if !is_canonical(H,mirror(ms))
        if verbose
            println("truncate detected non-orthogonal MPO, will now orthogonalize")
        end
        orthogonalize!(H;kwargs...,orth=mirror(lr)) #TODO why fail if spec ul here??
        if verbose
            previous_Dw=Base.max(get_Dw(H)...)
        end#
    end
    N=length(H)
    ss=bond_spectrums(undef,N-1)
    link_offest = lr==left ? 0 : -1
    rng=sweep(H,lr)
    for n in rng 
        nn=n+rng.step #index to neighbour
        W,RL,s=truncate(H[n],ul;kwargs...,orth=lr)
        H[n]=W
        H[nn]=RL*H[nn]
        is_regular_form(H[nn],ms.ul)
        ss[n+link_offest]=s
    end
    if verbose
        Dw=Base.max(get_Dw(H)...)
        println("After $lr truncation sweep Dw was reduced from $previous_Dw to $Dw")
    end
    return ss
end

@doc """
    truncate!(H::InfiniteMPO;kwargs...)

Truncate a `CelledVector` representation of an infinite MPO as described in section VII and Alogrithm 5 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
It is not nessecary (or recommended) to call the `orthogonalize!` function prior to calling `truncate!`. The `truncate!` function will do this automatically.  This is because the truncation process requires the gauge transform tensors resulting from left orthogonalizing an already right orthogonalized iMPO (or converse).  So it is better to do this internally in order to be sure the correct gauge transforms are used.

# Arguments
- H::InfiniteMPO which is a `CelledVector` of MPO matrices. `CelledVector` and `InfiniteMPO` are defined in the `ITensorInfiniteMPS` module.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form for the output
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=1.0 indicate no rank reduction.
- `cutoff::Float64 = 0.0` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
- `maxdim::Int64` : If the number of singular values exceeds `maxdim`, only the largest `maxdim` will be retained.
- `mindim::Int64` : At least `mindim` singular values will be retained, even if some fall below the cutoff
   
# Returns
- Vector{ITensor} with the diagonal gauge transforms between the input and output iMPOs
- a `bond_spectrums` object which is a `Vector{Spectrum}`

# Example
```
julia> using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
julia> initstate(n) = "↑";
julia> sites = infsiteinds("S=1/2", 1;initstate, conserve_szparity=false)
1-element CelledVector{Index{Int64}, typeof(translatecelltags)}:
 (dim=2|id=224|"S=1/2,Site,c=1,n=1")
julia> H=make_transIsing_iMPO(sites,7);
julia> get_Dw(H)[1]
30
julia> Ss,spectrum=truncate!(H;rr_cutoff=1e-15,cutoff=1e-15);
julia> get_Dw(H)[1]
9
julia> pprint(H[1])
I 0 0 0 0 0 0 0 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
0 S S S S S S S I 
julia> @show spectrum
spectrum = 
site  Ns   max(s)     min(s)    Entropy  Tr. Error
   1    7  0.39565   1.26e-02   0.32644  1.23e-16

```
"""
function ITensors.truncate!(H::InfiniteMPO;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any}
    #@printf "---- start compress ----\n"
    #
    # decide left/right and upper/lower
    #
    h_mirror::Bool=get(kwargs, :h_mirror, false) #Calculate and return mirror of H
    lr::orth_type=get(kwargs, :orth, left) #this specifies the final output orth direction.
    (bl,bu)=detect_regular_form(H)
    if !(bl || bu)
        throw(ErrorException("truncate!(H::MPO), H must be in either lower or upper regular form"))
    end
    @mpoc_assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    #
    # Now check if H requires orthogonalization
    #
    ms=matrix_state(ul,lr)
    can1,can2=is_canonical(H,ms),is_canonical(H,mirror(ms))
    if !(can1 || can2)
        rr_cutoff=get(kwargs, :cutoff, 1e-15)
        orthogonalize!(H;orth=mirror(lr),rr_cutoff=rr_cutoff,max_sweeps=1) 
        Hm=h_mirror ? copy(H) : nothing
        @mpoc_assert is_orthogonal(H,mirror(lr))
        Gs=orthogonalize!(H;orth=lr,rr_cutoff=rr_cutoff,max_sweeps=1) #TODO why fail if spec ul here??
        @mpoc_assert is_orthogonal(H,lr)
    else
        # user supplied canonical H but not the Gs so we cannot proceed unless we do one more
        # wasteful sweep
        @mpoc_assert false #for now.
    end
    
    return truncate!(H,Hm,Gs,lr;kwargs...)
end

ITensors.truncate!(H::InfiniteMPO,Gs::CelledVector{ITensor},lr::orth_type;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any} = ITensors.truncate!(H,nothing,Gs,lr;kwargs...)

function ITensors.truncate!(H::InfiniteMPO,Hm::Union{InfiniteMPO,Nothing},Gs::CelledVector{ITensor},lr::orth_type;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any}
    N=length(H)
    ss=bond_spectrums(undef,N)
    Ss=CelledVector{ITensor}(undef,N)
    for n in 1:N 
        #prime the right index of G so that indices can be distinguished.
        #Ideally orthogonalize!() would spit out Gs that are already like this.
        _,igr=inds(Gs[n])
        Gs[n]=replaceind(Gs[n],igr,prime(igr))
        if lr==left
            #println("-----------------Left----------------------")
            il,igl=parse_links(H[n]) #right link of H is the left link of G
            U,Sp,V,spectrum=truncate(Gs[n],dag(igl);kwargs...)
            transform(H,U,n)
            transform(Hm,dag(V),n)
        else
            #println("-----------------Right----------------------")
            il,ir=parse_links(H[n+1]) #left link of H[n+1] is the right link of G[n]
            igl=noncommonind(Gs[n],il)
            U,Sp,V,spectrum=truncate(Gs[n],igl;kwargs...) 
            transform(H,dag(V),n)
            transform(Hm,U,n)
        end
       
        if hasqns(Gs[n])
            iSs=make_qninds(Sp,inds(dag(U))...)
            Sp=convert_blocksparse(Sp,iSs...)
        end
       
        Ss[n]=Sp
        ss[n]=spectrum
    end
    return Ss,ss,Hm

end

function transform(H::InfiniteMPO,uv::ITensor,n::Int64)
    H[n]=H[n]*uv
    H[n+1]=dag(uv)*H[n+1]
    @mpoc_assert order(H[n])==4
    @mpoc_assert order(H[n+1])==4
end
function transform(H::Nothing,uv::ITensor,n::Int64) end


function truncate(G::ITensor,igl::Index;kwargs...)
    @mpoc_assert order(G)==2
    igr=noncommonind(G,igl)
    @mpoc_assert tags(igl)!=tags(igr) || plev(igl)!=plev(igr) #Make sure subtensr can distinguish igl and igr
    M=G[igl=>2:dim(igl)-1,igr=>2:dim(igr)-1]
    iml,=inds(M,plev=plev(igl)) #tags are the same, so plev is the only way to distinguish
    U,s,V,spectrum,iu,iv=svd(M,iml;kwargs...)
    #
    # Build up U+, S+ and V+
    #
    iup=redim(iu,dim(iu)+2) #Use redim to preserve QNs
    ivp=redim(iv,dim(iv)+2) 
    Up=grow(noprime(U),noprime(igl),dag(iup))
    Sp=grow(s,iup,ivp)
    Vp=grow(noprime(V),dag(ivp),noprime(igr))
    #
    #  But external link tags in so contractions with W[n] tensors will work.
    #
    replacetags!(Up,tags(iu),tags(igl))
    replacetags!(Sp,tags(iu),tags(igl))
    replacetags!(Sp,tags(iv),tags(igr))
    replacetags!(Vp,tags(iv),tags(igr))
    #@mpoc_assert norm(dense(G)-dense(Up)*Sp*dense(Vp))<1e-12    expensive!!!
    return Up,Sp,Vp,spectrum
end


