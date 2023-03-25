
#using Printf

#
#  Compress one site
#
function truncate(W::ITensor,ul::reg_form;kwargs...)::Tuple{ITensor,ITensor,Spectrum,Bool}
    lr::orth_type=get(kwargs, :orth, right)
    ms=matrix_state(ul,lr)
    iforward,_=parse_links(W,lr) # W[l=$(n-1)l=$n]=W[r,c]
    # establish some tag strings then depend on lr.
    (tsvd,tuv) = lr==left ? ("qx","Link,u") : ("m","Link,v")
    #
    # Block repecting QR/QL/LQ/RQ factorization.  RL=L or R for upper and lower.
    # here we purposely turn off rank reavealing feature (rr_cutoff=-1.0) to (mostly) avoid
    # horizontal rectangular RL matricies which are hard to handle accurately.  All rank reduction
    # should have been done in the ortho process anyway.
    #
   
    Q,RL,iqx=block_qx(W,iforward,ul;kwargs...,rr_cutoff=-1.0) #left Q[r,qx], RL[qx,c] - right RL[r,qx] Q[qx,c]
    # expensive
    # if order(Q)==4
    #     @mpoc_assert check_ortho(Q,ms,eps)
    #     @mpoc_assert is_regular_form(Q,ul,eps)
    # end
    c=noncommonind(RL,iqx) #if size changed the old c is not lnger valid
    #
    #  If the RL is rectangular in wrong way, then factoring out M is very difficult.
    #  For now we just bail out.
    #
    if dim(c)>dim(iqx) || dim(c)<3
        replacetags!(RL,"Link,qx",tags(iforward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
        replacetags!(Q ,"Link,qx",tags(iforward)) #W[l=n-1,l=n]
        return Q,RL,Spectrum([],0),true
    end
    
    #
    #  Factor RL=M*L' (left/lower) = L'*M (right/lower) = M*R' (left/upper) = R'*M (right/upper)
    #  M will be returned as a Dw-2 X Dw-2 interior matrix.  M_sans in the Parker paper.
    #
    M,RL_prime,im=getM(RL,ms.ul) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
    #  
    #  At last we can svd and compress M using epsSVD as the cutoff.  M should be dense.
    #    
    isvd=findinds(M,tsvd)[1] #decide the left index
    U,s,V,spectrum,iu,iv=svd(M,isvd;kwargs...) # ns sing. values survive compression
    ns=dim(inds(s)[1])

    #@show diag(array(s))
   
    #
    #  No recontrsuction RL, and W in the truncated space.
    #
    if lr==left
        iup=redim(iu,ns+2,1)
        RL=grow(s*V,iup,im)*RL_prime #RL[l=n,u] dim ns+2 x Dw2
        Uplus=grow(U,dag(iqx),dag(iup))
        W=Q*Uplus #W[l=n-1,u]
    else # right
        ivp=redim(iv,ns+2,1)
        RL=RL_prime*grow(U*s,im,ivp) #RL[l=n-1,v] dim Dw1 x ns+2
        Vplus=grow(V,dag(iqx),dag(ivp)) #lq has the dir of Q so want the opposite on Vplus
        W=Vplus*Q #W[l=n-1,v]
    end

    replacetags!(RL,tuv,tags(iforward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
    replacetags!(W ,tuv,tags(iforward)) #W[l=n-1,l=n]
    # expensive.
    # @mpoc_assert is_regular_form(W,ul,eps)
    # @mpoc_assert check_ortho(W,ms,eps)
    #@show luq lvq inds(Q) inds(Wq) inds(RLq)
    return W,RL,spectrum,false
end

function one_trunc_sweep!(H::MPO,ul::reg_form;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    verbose::Bool=get(kwargs, :verbose, false)
    ss=bond_spectrums(undef,length(H)-1)
    link_offest = lr==left ? 0 : -1
    rng=sweep(H,lr)
    encountered_bailout=false
    if verbose
        previous_Dw=Base.max(get_Dw(H)...)
    end
    #@show "----------truncate-----------"
    for n in rng 
        nn=n+rng.step #index to neighbour
        W,RL,s,bail=truncate(H[n],ul;kwargs...,orth=lr)
        encountered_bailout=encountered_bailout||bail
        H[n]=W
        H[nn]=RL*H[nn]
        
        @mpoc_assert is_regular_form(H[n],ul)
        @mpoc_assert is_regular_form(H[nn],ul)
        ss[n+link_offest]=s
       
    end
    H.rlim = rng.stop+rng.step+1
    H.llim = rng.stop+rng.step-1
    
    if verbose
        Dw=Base.max(get_Dw(H)...)
        println("After $lr truncation sweep Dw was reduced from $previous_Dw to $Dw")
    end
    return ss,encountered_bailout
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

julia> isortho(H,left)==true
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
    if !isortho(H,lr)
        if verbose
            println("truncate detected non-orthogonal MPO, will now orthogonalize")
        end
        orthogonalize!(H,ul;kwargs...,orth=mirror(lr)) #TODO why fail if spec ul here??
        
        if verbose
            previous_Dw=Base.max(get_Dw(H)...)
        end#
    end

    svd_cutoff=get(kwargs, :cutoff, 1e-15)
    ss,encountered_bailout=one_trunc_sweep!(H,ul;kwargs...,cutoff=svd_cutoff)
    max_sweeps::Int64=get(kwargs,:max_sweeps,5)
   
    nsweeps=1
    while encountered_bailout && nsweeps<=max_sweeps
        if verbose
            println("Encountered bailout, doing extra truncation sweeps")
        end
        lr=mirror(lr)
        ss,encountered_bailout=one_trunc_sweep!(H,ul;kwargs...,orth=lr,cutoff=svd_cutoff)
        nsweeps+=1
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
    can1,can2=isortho(H,lr),isortho(H,mirror(lr))
    if !(can1||can2)
        rr_cutoff=get(kwargs, :cutoff, 1e-15)
        orthogonalize!(H,ul;orth=mirror(lr),rr_cutoff=rr_cutoff,max_sweeps=1) 
        Hm=h_mirror ? copy(H) : nothing
        Gs=orthogonalize!(H,ul;orth=lr,rr_cutoff=rr_cutoff,max_sweeps=1) #TODO why fail if spec ul here??
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
        iln=linkind(H,n) #Link between Hn amd Hn+1
        if need_guage_fix(Gs[n])
            #@show inds(Gs[n]) inds(H[n],tags="Link")
            @show Gs[n]
            @pprint(H[n])
            Gs[n],H[n]=re_gauge(Gs[n],H[n])
            #@show inds(Gs[n]) inds(H[n],tags="Link")
            @show Gs[n]
            @pprint(H[n])
        end
        if lr==left
            # println("-----------------Left----------------------")
            igl=iln #right link of Hn is the left link of Gn
            U,Sp,V,spectrum=truncate(Gs[n],dag(igl);kwargs...)
            #@show U*Sp*V Gs[n]
            #pprint(H[n])
            transform(H,U,n)
            transform(Hm,dag(V),n)
        else
            # println("-----------------Right----------------------")
            igl=noncommonind(Gs[n],iln) #left link of Hn+1 is the right link Gn
            U,Sp,V,spectrum=truncate(Gs[n],igl;kwargs...) 
            transform(H,dag(V),n)
            transform(Hm,U,n)
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
function transform(::Nothing,::ITensor,::Int64) end

function need_guage_fix(G_plus::ITensor)
    lp,rp=inds(G_plus)
    Dw=dim(lp)
    @mpoc_assert Dw==dim(rp)
    G_bottom=slice(G_plus,lp=>Dw)
    x=G_bottom[rp=>2:Dw-1]
    return maximum(abs.(x))>1e-15
end

function re_gauge(G_plus::ITensor,W::ITensor)
    lp,rp=inds(G_plus)
    lw,rw=parse_links(W)
    Dw=dim(lp)
    @mpoc_assert Dw==dim(rp)
    G_bottom=slice(G_plus,lp=>Dw)
    x=G_bottom[rp=>2:Dw-1]
    G=G_plus[lp=>2:Dw-1,rp=>2:Dw-1]
    Gm=matrix(G)
    xv=vector(x)
    ImG=LinearAlgebra.I-Gm
    t=ImG\xv #solve [I-G]*t=x for t.
    G_prime=grow(G,lp,rp)
    L=dense(delta(lw',lw))
    Linv=dense(delta(rw,rw'))
    for n in 1:Dw-2
        L[lw'=>Dw,lw=>1+n]=t[n]
        Linv[rw=>Dw,rw'=>1+n]=-t[n]
    end
    #@mpoc_assert norm(L*G_plus*Linv-prime(G_prime))<1e-14
    W_prime=prime(L*W*Linv,-1,tags="Link")
    return G_prime,W_prime

end

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
    iup=redim(iu,dim(iu)+2,1) #Use redim to preserve QNs
    ivp=redim(iv,dim(iv)+2,1) 
    #@show iu iup iv ivp igl s dense(s) U
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
    #@mpoc_assert norm(dense(noprime(G))-dense(Up)*Sp*dense(Vp))<1e-12    #expensive!!!
    return Up,Sp,Vp,spectrum
end


