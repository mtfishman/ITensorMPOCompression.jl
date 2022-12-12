using ITensors
using NDTensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.4f", f)

# @testset "Orthogonalize InfiniteMPO 2-body Hamiltonians" for ul in [lower,upper], qns in [false,true]
#     initstate(n) = "↑"
#     for N in [1,2,4], NNN in [2,4] #3 site unit cell fails for qns=true.
#         si = infsiteinds("S=1/2", N; initstate, conserve_szparity=qns)
#         H=make_transIsing_MPO(si,NNN,0.0,ul,1.0;pbc=true)
#         @test is_regular_form(H)
#         H0=InfiniteMPO(H.data)
#         HL=copy(H0)
#         @test is_regular_form(HL)
#         GL=orthogonalize!(HL;orth=left)
#         @test is_regular_form(H)
#         @test is_orthogonal(HL,left)
#         for n in 1:N
#             @test norm(HL[n]*GL[n]-GL[n-1]*H0[n]) ≈ 0.0 atol = 1e-14 
#         end
#         HR=copy(H0)
#         GR=orthogonalize!(HR;orth=right)
#         @test is_regular_form(HR)
#         @test is_orthogonal(HR,right)
#         for n in 1:N
#             @test norm(GR[n]*HR[n]-H0[n]*GR[n+1]) ≈ 0.0 atol = 1e-14
#         end   
#         HR1=copy(HL) 
#         G=orthogonalize!(HR1;orth=right)
#         @test is_regular_form(HR1)
#         @test is_orthogonal(HR1,right)
#         for n in 1:N
#             @test norm(G[n]*HR1[n]-HL[n]*G[n+1]) ≈ 0.0 atol = 1e-14
#         end   
#     end
# end

@testset "Truncate/Compress InfiniteMPO 2-body Hamiltonians" begin
    initstate(n) = "↑"
    for N in [2], NNN in [4] #3 site unit cell fails for qns=true.
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=false)
        H=make_transIsing_MPO(si,NNN,0.0,lower,1.0;pbc=true)
        @test is_regular_form(H)
        H0=InfiniteMPO(H.data)
        #
        #  Do truncate outputting left ortho Hamiltonian
        #
        HL=copy(H0)
        Ssl,ss=truncate!(HL;orth=left,cutoff=1e-15)
        # @show Ss ss 
        # @pprint(HL[1])
        @test is_regular_form(HL)
        @test is_orthogonal(HL,left)
        #
        #  Do truncate outputting right ortho Hamiltonian
        #
        HR=copy(H0)
        Ssr,ss=truncate!(HR;orth=right,cutoff=1e-15)
        #@pprint(HR[1])
        @test is_regular_form(HR)
        @test is_orthogonal(HR,right)
        #
        #  Now test guage relations using the diagonal singular value matrices
        #  as the gauge transforms.
        #
        for n in 1:N
            @show inds(Ssr[n-1]) inds(HR[n],tags="Link")
            D1=Ssr[n]*HR[n]
            @assert order(D1)==4
            D2=HL[n]*Ssl[n]
            @assert order(D2)==4
            D=tensor(D1)-tensor(D2)
            @show norm(D)
        end
    end
end

nothing

