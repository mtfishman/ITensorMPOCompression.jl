#
#  Functions for bringing an iMPO into left or right canonical form
#
#
#  Functions for bringing an iMPO into left or right canonical form
#
function qx_step!(W::ITensor,G::ITensor,n::Int64,r::Index,c::Index,ul::reg_form,eps::Float64;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    forward = lr==left ? c : r
    Q,RL,iq=block_qx(W,n,r,c,ul;epsrr=1e-12,kwargs...) # r-Q-qx qx-RL-c
    #
    #  How far are we from RL==Id ?
    #
    if dim(c)==dim(iq)
        eta=norm(RL-δ(Float64, iq,forward))
    else
        eta=99.0 #Rank reduction occured to keep going.
    end
    #
    #  Fix up "Link,qx" indices.
    #
    ilnp=prime(settags(iq,tags(forward))) #"qx" -> "l=$n" prime
    replaceind!(RL,iq,ilnp)
    replaceind!(Q ,iq,ilnp)
    #
    #  Update the accumulated gauge transform
    #
    G=noprime(RL*G)
    @assert order(G)==2 #This will fail if the indices somehow got messed up.
    return Q,RL,G,eta
end


#
#  Loop throught the sites in correct direction
#
function qx_iterate!(H::InfiniteMPO,ul::reg_form;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    quiet::Bool=get(kwargs, :quiet, true)
    N=length(H)
    #
    #  Init gauge transform with unit matrices.
    #
    Gs=CelledVector{ITensor}(undef,N)
    for n in 1:N
        d,nsite,il,ir=parse_links(H[n],N) #get left and right indices
        if lr==left
            Gs[n]=δ(Float64,ir,ir') 
        else
            Gs[n]=δ(Float64,il,il')
        end
    end
    RLs=CelledVector{ITensor}(undef,N)
    
    
    eps=1e-13
    niter=0
    max_iter=40
    if !quiet
        @printf "niter eta\n" 
    end
    loop=true
    rng=sweep(H,lr)
    while loop
        eta=0.0
        for n in rng
            d,nsite,il,ir=parse_links(H[n],N) #get left and right indices
            H[n],RLs[n],Gs[n],etan=qx_step!(H[n],Gs[n],n,il,ir,ul,eps;kwargs...)
            eta=Base.max(eta,etan)
        end
        #
        #  H now contains all the Qs.  We need to transfer the RL's
        #  Direction?
        #
        for n in rng
            #@show inds(H[n],tags="Link") inds(RLs[n-1],tags="Link")
            H[n]=RLs[n-rng.step]*H[n] #W(n)=RL(n-1)*Q(n)
            #@show inds(H[n],tags="Link")
            @assert order(H[n])==4
            H[n]=noprime(H[n],tags="Link")
        end
        niter+=1
        loop=eta>1e-13 && niter<max_iter
        #if loop set_tags!(H) end #add back left/right markers
        if eta<1.0 && !quiet
            @printf "%4i %1.1e\n" niter eta
        end
    end
    return Gs
end

#
#  Next level down we select a algorithm
#
function i_orthogonalize!(H::InfiniteMPO,ul::reg_form;kwargs...)
    return qx_iterate!(H,ul;kwargs...)
end

#
#  Out routine simply established upper or lower regular forms
#
function i_orthogonalize!(H::MPO;kwargs...)
    @assert has_pbc(H)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    return orthogonalize!(H,ul;kwargs...)
end

