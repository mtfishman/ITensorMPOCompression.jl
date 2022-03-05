#
#  Functions for bringing and MPO into left or right canonical form
#
function canonical!(W1::ITensor,W2::ITensor,ul::tri_type,n::Int64;kwargs...)
    W1,Lplus=block_qx(W1,ul;kwargs...) 
    W2=Lplus*W2
    il=filterinds(inds(Lplus),tags="l=$n")[1]
    iq=filterinds(inds(Lplus),tags="qx")[1]
    il=Index(dim(iq),tags(il))
    replaceind!(W1,iq,il)
    replaceind!(W2,iq,il)
    @assert is_regular_form(W1,ul,1e-14)
    @assert is_regular_form(W2,ul,1e-14)
    return W1,W2 #We should not to return these if W1 and W2 were truely passed by reference.
end

"""
    canonical!(H::MPO,ms::matrix_state)

Bring an MPO into left or right canonical form using block respecting QR decomposition
 as described in:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147

# Keywords
- `dir::orth_type = right` : choose `left` or `right` canonical form
- `epsrr::Foat64 = 1e-14` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[:,j]))<epsrr are considered zero and removed. 
"""
function canonical!(H::MPO;kwargs...)
    @assert has_pbc(H)
    lr::orth_type=get(kwargs, :dir, right)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("canonical!(H::MPO), H must be either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::tri_type = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    N=length(H)
    if lr==left
        for n in 1:N-1 #sweep left to right
            H[n],H[n+1]=canonical!(H[n],H[n+1],ul,n;kwargs...)
        end
    else #right
        for n in N:-1:2 #sweep right to left
            H[n],H[n-1]=canonical!(H[n],H[n-1],ul,n-1;kwargs...)
        end
    end
end


