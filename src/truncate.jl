
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




function truncate(Wrf::reg_form_Op,lr::orth_type;kwargs...)::Tuple{reg_form_Op,ITensor,Spectrum,Bool}
    ms=matrix_state(Wrf.ul,lr)
    iforward,_=parse_links(Wrf.W,lr) # W[l=$(n-1)l=$n]=W[r,c]
    # establish some tag strings then depend on lr.
    (tsvd,tuv) = lr==left ? ("qx","Link,u") : ("m","Link,v")
    #
    # Block repecting QR/QL/LQ/RQ factorization.  RL=L or R for upper and lower.
    # here we purposely turn off rank reavealing feature (rr_cutoff=-1.0) to (mostly) avoid
    # horizontal rectangular RL matricies which are hard to handle accurately.  All rank reduction
    # should have been done in the ortho process anyway.
    #
    # @show inds(Wrf.W,tags="Link") Wrf.ileft Wrf.iright
    check(Wrf)
    Q,RL,iqx=ac_qx(Wrf,lr;kwargs...) #left Q[r,qx], RL[qx,c] - right RL[r,qx] Q[qx,c]
    RL=prime(RL,iqx)
    RL=replacetags(RL,tags(iqx),"Link,qx";plev=1)
    RL=noprime(RL)
    Q.W=prime(Q.W,iqx)
    Q.W=replacetags(Q.W,tags(iqx),"Link,qx";plev=1)
    Q.W=noprime(Q.W,tags="Link")
    
    iqx=replacetags(iqx,tags(iqx),"Link,qx")
    #@show "truncate" iqx
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
    M,RL_prime,im=getM(RL,iqx,Wrf.ul) #left M[lq,im] RL_prime[im,c] - right RL_prime[r,im] M[im,lq]
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
        Wrf.W=Q.W*Uplus #W[l=n-1,u]
        Wrf.iright=settags(dag(iup),tags(iforward))
    else # right
        ivp=redim(iv,ns+2,1)
        RL=RL_prime*grow(U*s,im,ivp) #RL[l=n-1,v] dim Dw1 x ns+2
        Vplus=grow(V,dag(iqx),dag(ivp)) #lq has the dir of Q so want the opposite on Vplus
        Wrf.W=Vplus*Q.W #W[l=n-1,v]
        Wrf.ileft=settags(dag(ivp),tags(iforward))
    end

    replacetags!(RL,tuv,tags(iforward)) #RL[l=n,l=n] sames tags, different id's and possibly diff dimensions.
    replacetags!(Wrf.W ,tuv,tags(iforward)) #W[l=n-1,l=n]
    check(Wrf)
    # expensive.
    # @mpoc_assert is_regular_form(W,ul,eps)
    # @mpoc_assert check_ortho(W,ms,eps)
    #@show luq lvq inds(Q) inds(Wq) inds(RLq)
    return Wrf,RL,spectrum,false
end

function ITensors.truncate!(H::reg_form_MPO,lr::orth_type;eps=1e-14,kwargs...)::bond_spectrums
    if !isortho(H)
        ac_orthogonalize!(H,lr;eps=eps,kwargs...)
        ac_orthogonalize!(H,mirror(lr);eps=eps,kwargs...)
    end
    if !is_gauge_fixed(H,eps)
        gauge_fix!(H)
    end
    ss=bond_spectrums(undef,length(H)-1)
    link_offest = lr==left ? 0 : -1
    rng=sweep(H,lr)
    
    if lr==left
        for n in rng
            nn=n+rng.step
            #@show inds(H[n].W,tags="Link") H[n].ileft H[n].iright
            check(H[n])
            W,R,s,bail=truncate(H[n],lr;kwargs...)
            H[n]=W
            H[nn].ileft=noncommonind(R,H[nn].W)
            H[nn].W=R*H[nn].W
            check(H[n])
            check(H[nn])
            ss[n+link_offest]=s
        end
    else
        for n in rng
            nn=n+rng.step
            check(H[n])
            W,R,s,bail=truncate(H[n],lr;kwargs...)
            H[n]=W
            H[nn].iright=noncommonind(R,H[nn].W)
            H[nn].W=R*H[nn].W
            check(H[n])
            check(H[nn])
            ss[n+link_offest]=s
        end
    end
    H.rlim = rng.stop+rng.step+1
    H.llim = rng.stop+rng.step-1
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
    lr::orth_type=get(kwargs, :orth, left) #this specifies the final output orth direction.
    verbose::Bool=get(kwargs, :verbose, false)
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
        orthogonalize!(H,ul;orth=mirror(lr),rr_cutoff=rr_cutoff,max_sweeps=1,verbose=verbose) 
        Hm=copy(H)
        Gs=orthogonalize!(H,ul;orth=lr,rr_cutoff=rr_cutoff,max_sweeps=1,verbose=verbose) #TODO why fail if spec ul here??
    else
        # user supplied canonical H but not the Gs so we cannot proceed unless we do one more
        # wasteful sweep
        @mpoc_assert false #for now.
    end
    return truncate!(H,Hm,Gs,lr,ul;kwargs...)
end

ITensors.truncate!(H::InfiniteMPO,Gs::CelledVector{ITensor},lr::orth_type,ul::reg_form;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any} = ITensors.truncate!(H,nothing,Gs,lr,ul;kwargs...)

function ITensors.truncate!(H::InfiniteMPO,Hm::InfiniteMPO,Gs::CelledVector{ITensor},lr::orth_type,ul::reg_form;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any}
    verbose::Bool=get(kwargs, :verbose, false)
    N=length(H)
    ms=matrix_state(ul,lr)
    ss=bond_spectrums(undef,N)
    Ss=CelledVector{ITensor}(undef,N)
    for n in 1:N 
        if lr==left
            if need_guage_fix(Gs,H,n,ms)
                gauge_tranform!(Gs,H,Hm,ms)
                verbose && println("Gauge fixing left")
            end
        else
            if need_guage_fix(Gs,Hm,n,ms)
                gauge_tranform!(Gs,Hm,H,ms)
                verbose && println("Gauge fixing right")
            end
        end
        #prime the right index of G so that indices can be distinguished.
        #Ideally orthogonalize!() would spit out Gs that are already like this.
        igl=commonind(Gs[n],H[n])
        igr=noncommonind(Gs[n],igl)
        Gs[n]=replaceind(Gs[n],igr,prime(igr))
        iln=linkind(H,n) #Link between Hn amd Hn+1
        
        if lr==left
            @assert igl==iln
            # println("-----------------Left----------------------")
            igl=iln #right link of Hn is the left link of Gn
            U,Sp,V,spectrum=truncate(Gs[n],dag(igl);kwargs...)
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

#
#  The x block moves around depending on lr && ul.   For left/lower it is here:
#
#      1 0 0
#  G = 0 M 0
#      0 x 1
#
#  See technical notes for the other cases.
#
function extract_xblock(G::ITensor,il::Index,ir::Index,ms::matrix_state)
    if ms.lr==left 
        irow=ms.ul==lower ? dim(il) : 1
        Xrow=slice(G,il=>irow) #slice out the row that contains x
        x=Xrow[ir=>2:dim(ir)-1]
    else
        icol=ms.ul==lower ? 1 : dim(ir)
        Xcol=slice(G,ir=>icol) #slice out the row that contains x
        x=Xcol[il=>2:dim(il)-1]
    end
    return x
end

function insert_xblock(L::Matrix{Float64},t::Vector{Float64},ms::matrix_state)
    nr,nc=size(L)
    if ms.lr==left
        @assert nc==length(t)+2
        r = ms.ul==lower ? nr : 1
        for i in 2:nc-1
            L[r,i]=t[i-1]
        end
    else
        @assert nr==length(t)+2
        c=ms.ul==lower ? 1 : nc
        for i in 2:nr-1
            L[i,c]=t[i-1]
        end
    end
    return L
end


function ITensors.linkinds(Gs::CelledVector{ITensor},Hs::InfiniteMPO,n::Int64)
    igl=commonind(Gs[n],Hs[n])
    return igl,noncommonind(Gs[n],igl)
end

#
#  Is there some other zeros in the x block?
#
function need_guage_fix(Gs::CelledVector{ITensor},Hs::InfiniteMPO,n::Int64,ms::matrix_state)
    igl,igr=linkinds(Gs,Hs,n)
    x=extract_xblock(Gs[n],igl,igr,ms)
    return maximum(abs.(x))>1e-14
end

#
#  Make sure indices are ordered and then convert to a matrix
#
function NDTensors.matrix(il::Index,T::ITensor,ir::Index)
        T1=ITensors.permute(T,il,ir; allow_alias=true)
        return matrix(T1)
end



#--------------------------------------------------------------------------------------------------
#
# The hardest part of this gauge transform is keeping all the indices straight.  Step 1 in addressing this
# is giving them names that make sense.  Here are the names on the gauge relation diagram (m=n-1):
#
#   iGml     iHRl      iHRr          iHLl      iGnl     iGnr
#            iGmr                              iHLr
#  -----G[m]-----HR[n]-----   ==    -----HL[n]-----G[n]-----  
#
#  We can read off the following identities:  iGml=iHLl, iHRr=iGnr,  iHRl=dag(iGmr), iHLr=dag(iGnl)
#  We need a find_all_links(G,HL,HR,n) function that returns all of these indices, even the redundant ones.
#  How to return indices?  A giant tuple is possible, but hard for the user to get everying ordered correctly
#  A predefined struct may be better, and relieves the user creating suitable names.
#
struct GaugeIndices
    Gml::Index
    Gmr::Index
    Gnl::Index
    Gnr::Index
    HLl::Index
    HLr::Index
    HRl::Index
    HRr::Index
end

function find_all_links(G::CelledVector,HL::InfiniteMPO,HR::InfiniteMPO,n::Int64)::GaugeIndices
    m=n-1
    iGmr=commonind(G[m],HR[n])
    iHLr=commonind(HL[n],G[n])
    iHRl=dag(iGmr)
    iGnl=dag(iHLr)
    iGml=noncommonind(G[m],iGmr)
    iGnr=noncommonind(G[n],iGnl)
    iHLl=iGml
    iHRr=iGnr
    @assert noncommonind(HL[n],iHLr;tags="Link")==iHLl
    @assert noncommonind(HR[n],iHRl;tags="Link")==iHRr
    return GaugeIndices(iGml,iGmr,iGnl,iGnr,iHLl,iHLr,iHRl,iHRr)
end

#
#  Gauge transform G to zero out the x block.  Return the gauge transforms L,L^-1
#  so they can be used to transform the MPO tensors.  These are returned as Matrix
#  objects because they need different indices for transforming the left and right MPOs.
#
function gauge_tranform_G(G::ITensor,il::Index,ir::Index,ms::matrix_state)
    x=extract_xblock(G,il,ir,ms) #return and ITensor
    Dwl,Dwr=dim(il),dim(ir)
    #
    #  Drop to Matrix level.
    #
    Gm=matrix(il,G,ir)
    @assert norm(x)>1e-14
    M=Gm[2:Dwl-1,2:Dwr-1]
    t=(LinearAlgebra.I-M)\vector(x) #solve [I-M]*t=x for t.
    if ms.lr==right 
        t=-t #swaps L Linv
    end
    L=insert_xblock(1.0*Matrix(LinearAlgebra.I,Dwr,Dwl),t,ms)
    Linv=insert_xblock(1.0*Matrix(LinearAlgebra.I,Dwr,Dwl),-t,ms)
    Gmp=L*Gm*Linv
    return ITensor(Gmp,il,ir),L,Linv    
end
#
#  DO the full L*G*L^-1 gauge transform on the gauge tensors and left/right MPO tensors. 
#
function gauge_tranform!(Gs::CelledVector,HL::InfiniteMPO,HR::InfiniteMPO,ms::matrix_state)
    Gps,HLps,HRps=ITensor[],ITensor[],ITensor[]
    for n in 1:length(Gs)
        ils=find_all_links(Gs,HL,HR,n)
        Gp,L,Linv=gauge_tranform_G(Gs[n],ils.Gnl,ils.Gnr,ms)

        LT=ITensor(L,ils.HLl',dag(ils.HLl)) 
        LinvT=ITensor(Linv,dag(ils.HLr),ils.HLr')
        HLp=noprime(LT*HL[n]*LinvT,tags="Link")

        LT=ITensor(L,ils.HRl',dag(ils.HRl))
        LinvT=ITensor(Linv,dag(ils.HRr),ils.HRr')
        HRp=noprime(LT*HR[n]*LinvT,tags="Link")
        push!(Gps,Gp)
        push!(HLps,HLp)
        push!(HRps,HRp)
    end
    for n in 1:length(Gs)
        Gs[n]=Gps[n]
        HL[n]=HLps[n]
        HR[n]=HRps[n]
    end
#    return CelledVector(Gps),InfiniteMPO(HLps,HL.llim,HL.rlim),InfiniteMPO(HRps,HR.llim,HR.rlim)
end




