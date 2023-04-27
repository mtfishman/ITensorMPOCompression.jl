
struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    AL::InfiniteMPO
    C::InfiniteMPO
    AR::InfiniteMPO
end

function InfiniteCanonicalMPO(HL::reg_form_iMPO,C::CelledVector{ITensor},HR::reg_form_iMPO)
    return InfiniteCanonicalMPO(InfiniteMPO(HL),InfiniteMPO(C),InfiniteMPO(HR))
end

Base.length(H::InfiniteCanonicalMPO)=length(H.C)
ITensors.data(H::InfiniteCanonicalMPO)=H.AL
ITensorInfiniteMPS.isreversed(::InfiniteCanonicalMPO)=false
Base.getindex(H::InfiniteCanonicalMPO, n::Int64)=getindex(H.AL,n)

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

function orthogonalize(Hi::InfiniteMPO;kwargs...)::InfiniteCanonicalMPO
    HL=reg_form_iMPO(Hi) #not HL yet, but will be after two ortho calls.
    ac_orthogonalize!(HL, right; kwargs...)
    HR = copy(HL)
    Gs = ac_orthogonalize!(HL,left; kwargs...)
    return InfiniteCanonicalMPO(HL,Gs,HR)
end

function truncate(Hi::InfiniteMPO;kwargs...)::Tuple{InfiniteCanonicalMPO,bond_spectrums}
    HL, HR, Ss, ss = truncate!(reg_form_iMPO(Hi))
    return InfiniteCanonicalMPO(HL,Ss,HR),ss
end