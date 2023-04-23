
struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    AL::InfiniteMPO
    C::InfiniteMPO
    AR::InfiniteMPO
end

function InfiniteCanonicalMPO(HL::reg_form_iMPO,C::CelledVector{ITensor},HR::reg_form_iMPO)
    return InfiniteCanonicalMPO(InfiniteMPO(HL),InfiniteMPO(C),InfiniteMPO(HR))
end

length(H::InfiniteCanonicalMPO)=length(H.C)

function check_ortho(H::InfiniteCanonicalMPO)::Bool
    return check_ortho(H.AL,left) && check_ortho(H.AR,right)
end

function check_gauge(H::InfiniteCanonicalMPO)::Float64
    eps2=0.0
    for n in eachindex(H)
        eps2+=norm(H.C[n - 1] * H.AR[n] - H.AL[n] * H.C[n])^2
    end
    return sqrt(eps2)
end
