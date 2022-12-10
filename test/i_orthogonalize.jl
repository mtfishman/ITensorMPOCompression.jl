using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)



@testset "Orthogonalize InfiniteMPO 2-body Hamiltonians" for ul in [lower]
    initstate(n) = "↑"
    for N in 1:4, NNN in [2,4]
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=false)
        H=make_transIsing_MPO(si,NNN,0.0,ul,1.0;pbc=true)
        @test is_regular_form(H,ul)
        H0=InfiniteMPO(H.data)
        HL=copy(H0)
        @test is_regular_form(HL,ul)
        GL=i_orthogonalize!(HL,lower;orth=left)
        @test is_regular_form(H,ul)
        @test is_orthogonal(HL,left)
        for n in 1:N
            @test norm(HL[n]*GL[n]-GL[n-1]*H0[n]) ≈ 0.0 atol = 1e-14 
        end
        HR=copy(H0)
        GR=i_orthogonalize!(HR,lower;orth=right)
        @test is_regular_form(HR,ul)
        @test is_orthogonal(HR,right)
        for n in 1:N
            @test norm(GR[n]*HR[n]-H0[n]*GR[n+1]) ≈ 0.0 atol = 1e-14
        end   
        HR1=copy(HL) 
        G=i_orthogonalize!(HR1,lower;orth=right)
        @test is_regular_form(HR1,ul)
        @test is_orthogonal(HR1,right)
        for n in 1:N
            @test norm(G[n]*HR1[n]-HL[n]*G[n+1]) ≈ 0.0 atol = 1e-14
        end   
    end
end

nothing

