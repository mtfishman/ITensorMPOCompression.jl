#
#  Functions for bringing an iMPO into left or right canonical form
#
#
#  Functions for bringing an iMPO into left or right canonical form
#
function qx_step!(W::ITensor,n::Int64,ul::reg_form,eps::Float64;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    forward,reverese=parse_links(W,lr)
    Q,RL,iq=block_qx(W,forward,ul;epsrr=1e-12,kwargs...) # r-Q-qx qx-RL-c
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
    quiet::Bool=get(kwargs, :quiet, true)
    N=length(H)
    #
    #  Init gauge transform with unit matrices.
    #
    Gs=CelledVector{ITensor}(undef,N)
    for n in 1:N
        forward,reverse=parse_links(H[n],lr) #get left and right indices
        Gs[n]=δ(Float64,dag(forward),forward') 
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
            H[n],RLs[n],etan=qx_step!(H[n],n,ul,eps;kwargs...)
            Gs[n]=noprime(RLs[n]*Gs[n])  #  Update the accumulated gauge transform
            @assert order(Gs[n])==2 #This will fail if the indices somehow got messed up.
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
function i_orthogonalize!(H::InfiniteMPO;kwargs...)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    return i_orthogonalize!(H,ul;kwargs...)
end

