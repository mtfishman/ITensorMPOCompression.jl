
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
        iup=redim1(iu,1,1,space(iqx))
        RL=grow(s*V,iup,im)*RL_prime #RL[l=n,u] dim ns+2 x Dw2
        Uplus=grow(U,dag(iqx),dag(iup))
        Wrf.W=Q.W*Uplus #W[l=n-1,u]
        Wrf.iright=settags(dag(iup),tags(iforward))
    else # right
        ivp=redim1(iv,1,1,space(iqx))
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
function ITensors.truncate!(H::reg_form_iMPO,lr::orth_type;rr_cutoff=1e-14,kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any}
    #@printf "---- start compress ----\n"
    #
    # Now check if H requires orthogonalization
    #
    if isortho(H,lr)
        @warn "truncate!(iMPO), iMPO is already orthogonalized, but the truncate algorithm needs the gauge transform tensors." *
        "running orthongonalie!() again to get the gauge tranforms."        
    end
    ac_orthogonalize!(H,mirror(lr),cutoff=rr_cutoff;kwargs...) 
    Hm=copy(H)
    Gs=ac_orthogonalize!(H,lr;cutoff=rr_cutoff,kwargs...) 
    return truncate!(H,Hm,Gs,lr;kwargs...)
end

function ITensors.truncate!(H::reg_form_iMPO,Hm::reg_form_iMPO,Gs::CelledVector{ITensor},lr::orth_type;kwargs...)::Tuple{CelledVector{ITensor},bond_spectrums,Any}
    if !is_gauge_fixed(H,1e-14)
        gauge_fix!(H)
    end
    
    N=length(H)
    ss=bond_spectrums(undef,N)
    Ss=CelledVector{ITensor}(undef,N)
    for n in 1:N 
        #prime the right index of G so that indices can be distinguished.
        #Ideally orthogonalize!() would spit out Gs that are already like this.
        igl=commonind(Gs[n],H[n].W)
        igr=noncommonind(Gs[n],igl)
        Gs[n]=replaceind(Gs[n],igr,prime(igr))
        #iln=linkind(H,n) #Link between Hn amd Hn+1
        iln=H[n].iright
        #           
        #  -----G[n-1]-----HR[n]-----   ==    -----HL[n]-----G[n]-----  
        #
        if lr==left
            @assert igl==iln
            # println("-----------------Left----------------------")
            igl=iln #right link of Hn is the left link of Gn
            U,Sp,V,spectrum=truncate(Gs[n],dag(igl);kwargs...)
            check(H[n])
          
            transform(H,U,n)
            transform(Hm,dag(V),n)
            check(H[n])
            check(Hm[n])
        else
            # println("-----------------Right----------------------")
            igl=noncommonind(Gs[n],iln) #left link of Hn+1 is the right link Gn
            U,Sp,V,spectrum=truncate(Gs[n],igl;kwargs...) 
            check(H[n])
            transform(H,dag(V),n)
            transform(Hm,U,n)
            check(H[n])
            check(Hm[n])
        end
       
        Ss[n]=Sp
        ss[n]=spectrum
    end
    return Ss,ss,Hm

end

function transform(H::reg_form_iMPO,uv::ITensor,n::Int64)
    @assert length(commoninds(H[n].W,uv))==1
    @assert length(commoninds(H[n+1].W,uv))==1
    ihu=commonind(H[n].W,uv)
    ihnu=noncommonind(uv,ihu)
    ihv=commonind(H[n+1].W,dag(uv))
    ihnv=noncommonind(uv,ihv)


    H[n]=reg_form_Op(H[n].W*uv,H[n].ileft,ihnu,H[n].ul)
    H[n+1]=reg_form_Op(dag(uv)*H[n+1].W,ihnv,H[n+1].iright,H[n+1].ul)
    @mpoc_assert order(H[n].W)==4
    @mpoc_assert order(H[n+1].W)==4
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
    iup=redim1(iu,1,1,space(igl)) #Use redim to preserve QNs
    ivp=redim1(iv,1,1,space(igr))
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
#  Make sure indices are ordered and then convert to a matrix
#
function NDTensors.matrix(il::Index,T::ITensor,ir::Index)
        T1=ITensors.permute(T,il,ir; allow_alias=true)
        return matrix(T1)
end





