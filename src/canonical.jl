#
#  Functions for bringing and MPO into left or right canonical form
#
function canonical!(H::MPO,lr::orth_type)
    @assert has_pbc(H)
    N=length(H)
    if lr==left
        for n in 1:N-1 #sweep left to right
            Lplus=block_qx(H[n],lr)
            H[n+1]=Lplus*H[n+1] 
            il=filterinds(inds(Lplus),tags="l=$n")[1]
            iq=filterinds(inds(Lplus),tags="ql")[1]
            replaceind!(H[n+1],iq,il)
        end
    else
        for n in N:-1:2 #sweep right to left
            Lplus=block_qx(H[n],lr)
            @assert detect_upper_lower(H[n],1e-14)==lower
            H[n-1]=Lplus*H[n-1]
            il=filterinds(inds(Lplus),tags="l=$(n-1)")[1]
            iq=filterinds(inds(Lplus),tags="lq")[1]
            replaceind!(H[n-1],iq,il)
            @assert detect_upper_lower(H[n-1],1e-14)==lower
        end
    end
end


