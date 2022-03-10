#
#  Functions for bringing and MPO into left or right canonical form
#
function orthogonalize!(W1::ITensor,W2::ITensor,ul::tri_type,n::Int64;kwargs...)
    W1,Lplus=block_qx(W1,ul;kwargs...) 
    W2=Lplus*W2
    il=filterinds(inds(Lplus),tags="l=$n")[1]
    iq=filterinds(inds(Lplus),tags="qx")[1]
    #pprint(iq,Lplus,il,1e-14)
    il=Index(dim(iq),tags(il))
    replaceind!(W1,iq,il)
    replaceind!(W2,iq,il)
    ITensors.@debug_check begin
        @assert is_regular_form(W1,ul,1e-14)
        @assert is_regular_form(W2,ul,1e-14)
    end
    return W1,W2 #We should not need to return these if W1 and W2 were truely passed by reference.
end

function orthogonalize!(H::MPO,ul::tri_type;kwargs...)
    pbc = has_pbc(H) 
    lr::orth_type=get(kwargs, :dir, right)
    N=length(H)
    if lr==left
        start = pbc ? 1 : 1
        rng=start:1:N-1 #sweep left to right
        link_offest=0
    else #right
        start = pbc ? N : N
        rng=start:-1:2 #sweep right to left
        link_offest=-1
    end
    for n in rng 
        nn=n+rng.step #index to neighbour
        nl=n+link_offest #index in link tag, l=$nl
        #@show n,nn,nl
        H[n],H[nn]=orthogonalize!(H[n],H[nn],ul,nl;kwargs...)
    end
end


"""
    orthogonalize!(H::MPO,ms::matrix_state)

Bring an MPO into left or right canonical form using block respecting QR decomposition
 as described in:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147

# Keywords
- `dir::orth_type = right` : choose `left` or `right` canonical form
- `sweeps::Int64` : number of sweeps to perform. If sweeps is zero or not set then sweeps continue there is no change in the internal dimensions from rank revelaing QR. 
- `epsrr::Foat64 = 1e-14` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[:,j]))<epsrr are considered zero and removed. 
"""
function orthogonalize!(H::MPO;kwargs...)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::tri_type = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    nsweep=get(kwargs,:sweeps,0)

    kwargs=Dict(kwargs) #this allows us to set the dir elements
    lr=get(kwargs,:dir,right)
    lrm=mirror(lr)
    if nsweep>0
        for isweep in 1:nsweep
            kwargs[:dir]=lrm
            orthogonalize!(H,ul;kwargs...)
            kwargs[:dir]=lr
            orthogonalize!(H,ul;kwargs...)
        end
    else
        Dws=get_Dw(H)
        while true
            kwargs[:dir]=lrm
            orthogonalize!(H,ul;kwargs...)
            kwargs[:dir]=lr
            orthogonalize!(H,ul;kwargs...)
            new_Dws=get_Dw(H)
            if new_Dws==Dws break end
            Dws=new_Dws
        end
    end
end


