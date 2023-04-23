
struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    AL::InfiniteMPO
    C::InfiniteMPO
    AR::InfiniteMPO
end

function InfiniteCanonicalMPO(HL::reg_form_iMPO,C::CelledVector{ITensor},HR::reg_form_iMPO)
    return InfiniteCanonicalMPO(InfiniteMPO(HL),InfiniteMPO(C),InfiniteMPO(HR))
end

length(H::InfiniteCanonicalMPO)=length(H.C)

