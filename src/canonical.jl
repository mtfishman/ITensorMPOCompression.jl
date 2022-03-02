function canonical!(W1::ITensor,W2::ITensor,ms::matrix_state,n::Int64)
    W1,Lplus=block_qx(W1,ms) 
    W2=Lplus*W2
    il=filterinds(inds(Lplus),tags="l=$n")[1]
    iq=filterinds(inds(Lplus),tags="qx")[1]
    il=Index(dim(iq),tags(il))
    replaceind!(W1,iq,il)
    replaceind!(W2,iq,il)
    #@assert is_upper_lower(W1,ms.ul,1e-14)
    #assert is_upper_lower(W2,ms.ul,1e-14)
    return W1,W2 #We should not to return these if W1 and W2 were truely passed by reference.
end
#
#  Functions for bringing and MPO into left or right canonical form
#
function canonical!(H::MPO,ms::matrix_state)
    @assert has_pbc(H)
    N=length(H)
    if ms.lr==left
        for n in 1:N-1 #sweep left to right
            H[n],H[n+1]=canonical!(H[n],H[n+1],ms,n)
        end
    else
        for n in N:-1:2 #sweep right to left
            H[n],H[n-1]=canonical!(H[n],H[n-1],ms,n-1)
        end
    end
end


