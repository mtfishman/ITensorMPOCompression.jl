using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

#H = ΣⱼΣn (½ S⁺ⱼS⁻ⱼ₊n + ½ S⁻ⱼS⁺ⱼ₊n + SᶻⱼSᶻⱼ₊n)
function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNN"; NNN::Int64)
    opsum = OpSum()
    for n in 1:NNN
        J = 1.0 / n
        opsum += J * 0.5, "S+", 1, "S-", 1 + n
        opsum += J * 0.5, "S-", 1, "S+", 1 + n
        opsum += J, "Sz", 1, "Sz", 1 + n
    end
    return opsum
end

function ITensorInfiniteMPS.unit_cell_terms(::Model"hubbardNNN"; NNN::Int64)
  U::Float64 = 0.25
  t::Float64 = 1.0
  V::Float64 = 0.5
  opsum = OpSum()
  opsum += (U, "Nupdn", 1)
  for n in 1:NNN
    tj, Vj = t / n, V / n
    opsum += -tj, "Cdagup", 1, "Cup", 1 + n
    opsum += -tj, "Cdagup", 1 + n, "Cup", 1
    opsum += -tj, "Cdagdn", 1, "Cdn", 1 + n
    opsum += -tj, "Cdagdn", 1 + n, "Cdn", 1
    opsum += Vj, "Ntot", 1, "Ntot", 1 + n
  end
  return opsum
end

function orthogonalize(Hi::InfiniteMPO;kwargs...)::InfiniteCanonicalMPO
    HL=reg_form_iMPO(Hi) #not HL yet, but will be after two ortho calls.
    ac_orthogonalize!(HL, right; kwargs...)
    HR = copy(HL)
    Gs = ac_orthogonalize!(HL,left; kwargs...)
    return InfiniteCanonicalMPO(HL,Gs,HR)
end

function truncate(Hi::InfiniteMPO;kwargs...)::Tuple{InfiniteCanonicalMPO,bond_spectrums}
    HL=reg_form_iMPO(Hi) #not HL yet, but will be after two ortho calls.
    Ss, ss, HR = truncate!(HL, left)
    return InfiniteCanonicalMPO(HL,Ss,HR),ss
end

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

check_ortho(H::InfiniteMPO,lr::orth_type)=ITensorMPOCompression.check_ortho(reg_form_iMPO(H),lr)

@testset "Create InfiniteCanonicalMPO from orthogonalize function" begin
    initstate(n) = isodd(n) ? "↑" : "↓"
    Ncell,site,model,qns,NNN=2,"Electron",Model"hubbardNNN"(),false,4
    model_kwargs = (NNN=NNN,)
    s = infsiteinds(site, Ncell; initstate, conserve_qns=qns)
    Hi = InfiniteMPO(model, s; model_kwargs...)

    Ho::InfiniteCanonicalMPO = orthogonalize(Hi) #Use default cutoff, C is non-diagonal
    @test check_ortho(Ho) #AL is left ortho && AR is right ortho
    @test check_gauge(Ho) ≈ 0.0 atol = 1e-14 #ensure C[n - 1] * AR[n] - AL[n] * C[n]

    Ht,BondSpectrums = truncate(Hi) #Use default cutoff,C is now diagonal
    @test check_ortho(Ht) #AL is left ortho && AR is right ortho
    @test check_gauge(Ht) ≈ 0.0 atol = 1e-14 #ensure C[n - 1] * AR[n] - AL[n] * C[n]
    @show BondSpectrums
end




#end