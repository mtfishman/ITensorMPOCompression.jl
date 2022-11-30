#
#  Functions for bringing and MPO into left or right canonical form
#
function ITensors.orthogonalize!(W1::ITensor,W2::ITensor,ul::reg_form;kwargs...)
    W1,Lplus=block_qx(W1,ul;kwargs...) 
    W2=Lplus*W2
    @assert order(W2)<=4 #make sure there was something to contract. 
 
    iq=filterinds(inds(Lplus),tags="qx")[1]
    il=noncommonind(Lplus,iq)
    #pprint(iq,Lplus,il,1e-14)
    il=redim(il,dim(iq)) #Index(dim(iq),tags(il))
    replaceind!(W1,iq,il)
    replaceind!(W2,iq,il)
    ITensors.@debug_check begin
        @assert is_regular_form(W1,ul,1e-14)
        @assert is_regular_form(W2,ul,1e-14)
    end
    return W1,W2 #We should not need to return these if W1 and W2 were truely passed by reference.
end

function ITensors.orthogonalize!(H::MPO,ul::reg_form;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    N=length(H)
    if lr==left
        rng=1:1:N-1 #sweep left to right
    else #right
        rng=N:-1:2 #sweep right to left
    end
    for n in rng 
        nn=n+rng.step #index to neighbour
        #@show n,nn,nl
        H[n],H[nn]=orthogonalize!(H[n],H[nn],ul;kwargs...)
    end
end


@doc """
    orthogonalize!(H::MPO)

Bring an MPO into left or right canonical form using block respecting QR decomposition
 as described in:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `sweeps::Int64` : number of sweeps to perform. If sweeps is zero or not set then sweeps 
   continue until there is no change in the internal dimensions from rank revealing QR. 
- `epsrr::Float64 = 1e-14` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<epsrr are considered zero and removed. 

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
julia> orthogonalize!(H)
#
#  Wahoo .. rank reduction knocked the size of H way down, and we haven't
#  tried compressing yet!
#
julia> pprint(H[2])
I 0 0 0 0 
S S S 0 0 
0 0 0 S I 
#
#  What do all the bond dimensions of H look like?  We will need compression 
#  (truncation) in order to further bang down the size of H
#
julia> get_Dw(H)
9-element Vector{Int64}: 3 5 9 13 12 9 6 4 3
#
#  wrap up with two more checks on the structure of H
#
julia> is_lower_regular_form(H)==true
true
julia> is_orthogonal(H,left)==true
true


```

"""
function ITensors.orthogonalize!(H::MPO;kwargs...)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    nsweep=get(kwargs,:sweeps,0)
    if length(kwargs)>0
        kwargs=Dict(kwargs) #this allows us to set the dir elements
    else
        kwargs=Dict{Symbol, Any}(:orth => left)
    end
    lr=get(kwargs,:orth,left)
    lrm=mirror(lr)
    if nsweep>0
        for isweep in 1:nsweep
            kwargs[:orth]=lrm
            orthogonalize!(H,ul;kwargs...)
            kwargs[:orth]=lr
            orthogonalize!(H,ul;kwargs...)
        end
    else
        Dws=get_Dw(H)
        if !haskey(kwargs,:orth)
            @show kwargs
            get!(kwargs,:orth,lr)
        end
        while true
            kwargs[:orth]=lrm
            orthogonalize!(H,ul;kwargs...)
            kwargs[:orth]=lr
            orthogonalize!(H,ul;kwargs...)
            new_Dws=get_Dw(H)
            if new_Dws==Dws break end
            Dws=new_Dws
        end
    end
end


