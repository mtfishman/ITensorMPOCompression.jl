#
#  Functions for bringing an MPO into left or right canonical form
#
function ITensors.orthogonalize!(W1::ITensor,W2::ITensor,ul::reg_form;kwargs...)
    iln=commonind(W1,W2)
    W1,Lplus=block_qx(W1,iln,ul;kwargs...) 
    W2=Lplus*W2
    @mpoc_assert order(W2)<=4 #make sure there was something to contract. 
 
    iq=filterinds(inds(Lplus),tags="qx")[1]
    il=noncommonind(Lplus,iq)
    #pprint(iq,Lplus,il,1e-14)
    il=redim(il,dim(iq)) #Index(dim(iq),tags(il))
    replaceind!(W1,iq,il)
    replaceind!(W2,iq,il)
    @mpoc_assert is_regular_form(W1,ul,1e-14)
    @mpoc_assert is_regular_form(W2,ul,1e-14)
    return W1,W2 #We should not need to return these if W1 and W2 were truely passed by reference.
end

function ITensors.orthogonalize!(H::MPO,ul::reg_form;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    verbose::Bool=get(kwargs, :verbose, false)
    if verbose
        previous_Dw=sum(get_Dw(H))
    end
    rng=sweep(H,lr)
    for n in rng 
        nn=n+rng.step #index to next neighbour in the sweep direction.
        H[n],H[nn]=orthogonalize!(H[n],H[nn],ul;kwargs...)
    end
    H.rlim = rng.stop+rng.step+1
    H.llim = rng.stop+rng.step-1
    if verbose
        Dw=sum(get_Dw(H))
        println("    MPO After $lr orth sweep sum(Dw) reduced from $previous_Dw to $Dw")
    end

end


@doc """
    orthogonalize!(H::MPO)

Bring an MPO into left or right canonical form using block respecting QR decomposition
 as described in:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=`1.0 indicates no rank reduction.

# Examples
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
#  Make sure we have a regular form or orhtogonalize! won't work.
#
julia> is_lower_regular_form(H)==true
true
#
#  Let's see what the second site for the MPO looks like.
#  I = unit operator, and S = any other operator
#
julia> pprint(H[2])
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
.
.
.
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 I 0 0 
0 S 0 S 0 0 S 0 0 0 S 0 0 0 0 S 0 0 0 0 0 S 0 0 0 0 0 0 S I 
#
#  Now we can orthogonalize or bring it into canonical form.
#  Defaults are left orthogonal with rank reduction.
#
julia> orthogonalize!(H;rr_cutoff=1e-14)
#
#  Wahoo .. rank reduction knocked the size of H way down, and we haven't
#  tried compressing yet!
#
julia> pprint(H[2])
I 0 0 0 
S I 0 0 
0 0 S I 
#
#  What do all the bond dimensions of H look like?  We will need compression 
#  (truncation) in order to further bang down the size of H
#
julia> get_Dw(H)
9-element Vector{Int64}: 3 4 5 6 7 6 5 4 3
#
#  wrap up with two more checks on the structure of H
#
julia> is_lower_regular_form(H)==true
true
julia> isortho(H,left)==true
true


```

"""
function ITensors.orthogonalize!(H::MPO;kwargs...)
    #
    #  Establish uperr or lower regular form
    #
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    @mpoc_assert !(bl && bu) #should not be diagonal.
    ul::reg_form = bl ? lower : upper 
    #
    #  Establish what options the user specified
    #
    verbose::Bool=get(kwargs, :verbose, false)
    request_lr::orth_type=get(kwargs,:orth,left)
    rr_cutoff::Float64=get(kwargs,:rr_cutoff,1e-15)
    max_sweeps::Int64=get(kwargs,:max_sweeps,5)
    spec_lr::Bool=haskey(kwargs,:orth) #Did the user explictely request an orth. direction?
    spec_ms::Bool=haskey(kwargs,:max_sweeps) #Did the user explictely request max_sweeps?
    rr_enabled::Bool=rr_cutoff>=0.0 #
    sumDw::Int64=sum(get_Dw(H))
    if verbose 
        println("---------Ortho. Sweep rr_cutoff=$rr_cutoff----------------")
    end

    if rr_cutoff==0.0
        @warn "orthogonalize!(::MPO) esprr=0.0 is not very effective for rank reduction. Set rr_cutoff<0.0 to disable rank reduction, os set rr_cutoff>=1e-15 for effective rank reduction."
    end
    #@show rr_enabled rr_cutoff
    if max_sweeps>1 && rr_enabled
        lr=ul==lower ? left : right #optimal start direction depends on ul for some reason.
        nsweeps=0
        old_sumDw=sumDw+1
        oldH=H
        while nsweeps<max_sweeps && sumDw<old_sumDw
            oldH=spec_lr ? copy(H) : nothing
            old_sumDw=sumDw
            orthogonalize!(H,ul;kwargs...,orth=lr)
            nsweeps+=1
            sumDw=sum(get_Dw(H))
            lr=mirror(lr) #need to undo this after we break out.
            @mpoc_assert sumDw<=old_sumDw #Make sure Dw did not grow!!
            #@show nsweeps max_sweeps Dw oldDw
            
        end
        if nsweeps>=1
            lr=mirror(lr) #undo last mirror op
        end
        #@show lr spec_lr request_lr 
        #pprint(H)
        #pprint(oldH)

        # Now make sure we have lr that user specified
        if request_lr!=lr
            if oldH≠nothing && nsweeps>1
                @mpoc_assert isortho(oldH,request_lr)
                # THis fails to copy data and lims: H=copy(oldH)
                H.=oldH
                H.llim=oldH.llim
                H.rlim=oldH.rlim
            else
                # THis means we did 1 sweep with no change in Dw, and now need to
                # to do another sweep to into the requested ortho. state.
                orthogonalize!(H,ul;kwargs...,orth=request_lr)
            end
        end
        
    else 
        old_sumDw=sumDw
        orthogonalize!(H,ul;kwargs...,orth=request_lr)
        sumDw=sum(get_Dw(H))
    end
    @mpoc_assert isortho(H,request_lr)

end


#--------------------------------------------------------------------------------------------
#
#  Functions for bringing an iMPO into left or right canonical form
#
function qx_step!(W::ITensor,n::Int64,ul::reg_form,eps::Float64;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    forward,_=parse_links(W,lr)
    Q,RL,iq=block_qx(W,forward,ul;rr_cutoff=1e-12,kwargs...) # r-Q-qx qx-RL-c
    #
    #  How far are we from RL==Id ?
    #
    if dim(forward)==dim(iq)
        eta=norm(dense(RL)-δ(Float64, inds(RL))) #block sparse - diag no supported yet
    else
        eta=99.0 #Rank reduction occured to keep going.
    end
    #
    #  Fix up "Link,qx" indices.
    #
    ilnp=prime(settags(iq,tags(forward))) #"qx" -> "l=$n" prime
    replaceind!(RL,iq,ilnp)
    replaceind!(Q ,iq,ilnp)
   
    return Q,RL,eta
end


#
#  Loop throught the sites in correct direction
#
function qx_iterate!(H::InfiniteMPO,ul::reg_form;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    verbose::Bool=get(kwargs, :verbose, false)
    N=length(H)
    #
    #  Init gauge transform with unit matrices.
    #
    Gs=CelledVector{ITensor}(undef,N)
    for n in 1:N
        ln=lr==left ? linkind(H,n) : dag(linkind(H,n)) #get the forward link index
        Gs[n]=δ(Float64,dag(ln),ln') 
    end
    RLs=CelledVector{ITensor}(undef,N)
    
    eps=1e-13
    niter=0
    max_iter=40
    previous_Dw=max_Dw(H)
    # if verbose
    #     previous_Dw=Base.max(get_Dw(H)...)
    #     @printf "niter eta\n" 
    # end
    loop=true
    rng=sweep(H,lr)
    while loop
        eta=0.0
        for n in rng
            H[n],RLs[n],etan=qx_step!(H[n],n,ul,eps;kwargs...)
            if lr==left
                Gs[n]=noprime(RLs[n]*Gs[n])  #  Update the accumulated gauge transform
            else
                Gs[n-1]=noprime(RLs[n]*Gs[n-1])  #  Update the accumulated gauge transform
            end
            @mpoc_assert order(Gs[n])==2 #This will fail if the indices somehow got messed up.
            eta=Base.max(eta,etan)
        end
        #
        #  H now contains all the Qs.  We need to transfer the RL's
        #
        for n in rng
            H[n]=RLs[n-rng.step]*H[n] #W(n)=RL(n-1)*Q(n)
            H[n]=noprime(H[n],tags="Link")
        end
        niter+=1
        loop=eta>1e-13 && niter<max_iter
        # if eta<1.0 && verbose
        #     @printf "%4i %1.1e\n" niter eta
        # end
    end
    H.rlim = rng.stop+1
    H.llim = rng.stop-1

    if verbose
        Dw=Base.max(get_Dw(H)...)
        println("   iMPO After $lr orth sweep, $niter iterations Dw reduced from $previous_Dw to $Dw")
    end
    return Gs
end

#
#  Next level down we select a algorithm
#
function ITensors.orthogonalize!(H::InfiniteMPO,ul::reg_form;kwargs...)
    return qx_iterate!(H,ul;kwargs...)
end

#
#  Outer routine simply established upper or lower regular forms
#
@doc """
    orthogonalize!(H::InfiniteMPO;kwargs...)

Bring `CelledVector` representation of an infinite MPO into left or right canonical form using 
block respecting QR iteration as described in section Vi B and Alogrithm 3 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
If you intend to also call `truncate!` then do not bother calling `orthogonalize!` beforehand, as `truncate!` will do this automatically and ensure the correct handling of that gauge transforms.

# Arguments
- H::InfiniteMPO which is `CelledVector` of MPO matrices. `CelledVector` and `InfiniteMPO` are defined in the `ITensorInfiniteMPS` module.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=-11.0 indicate no rank reduction.

# Returns
- Vector{ITensor} with the gauge transforms between the input and output iMPOs

# Examples
```julia
julia> using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
julia> initstate(n) = "↑";
julia> sites = infsiteinds("S=1/2", 1;initstate, conserve_szparity=false)
1-element CelledVector{Index{Int64}, typeof(translatecelltags)}:
 (dim=2|id=326|"S=1/2,Site,c=1,n=1")
#
# This makes H directly, bypassing autoMPO.  (AutoMPO is too smart for this
# demo, it makes maximally reduced MPOs right out of the box!)
#
julia> H=make_transIsing_MPO(sites,NNN);
julia> get_Dw(H)
1-element Vector{Int64}:
 17
julia> orthogonalize!(H;orth=right,rr_cutoff=1e-15);
julia> get_Dw(H)
1-element Vector{Int64}:
 14
julia> orthogonalize!(H;orth=left,rr_cutoff=1e-15);
julia> get_Dw(H)
 1-element Vector{Int64}:
  13
julia> isortho(H,left)
true


```

"""
function ITensors.orthogonalize!(H::InfiniteMPO;kwargs...)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
        @pprint(H[1])
    end
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    lr::orth_type=get(kwargs, :orth, left)
    max_sweeps::Int64=get(kwargs,:max_sweeps,2)
    if max_sweeps<0 || max_sweeps>2
        @warn("orthogonalize!(iMPO) illegal value requested for max_sweeps=$max_sweeps.\n  Resetting to max_sweeps=2")
    end
    if  max_sweeps==2
        orthogonalize!(H,ul;kwargs...,orth=mirror(lr))
    end
    
    return orthogonalize!(H,ul;kwargs...)
end

