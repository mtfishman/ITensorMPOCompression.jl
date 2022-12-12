using ITensors
#using NDTensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.4f", f)

@testset "Orthogonalize InfiniteMPO 2-body Hamiltonians" for ul in [lower,upper], qns in [false,true]
    initstate(n) = "↑"
    for N in [1,2,4], NNN in [2,4] #3 site unit cell fails for qns=true.
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=qns)
        H=make_transIsing_MPO(si,NNN,0.0,ul,1.0;pbc=true)
        @test is_regular_form(H)
        H0=InfiniteMPO(H.data)
        HL=copy(H0)
        @test is_regular_form(HL)
        GL=ITensorMPOCompression.orthogonalize!(HL;orth=left)
        @test is_regular_form(H)
        @test is_orthogonal(HL,left)
        for n in 1:N
            @test norm(HL[n]*GL[n]-GL[n-1]*H0[n]) ≈ 0.0 atol = 1e-14 
        end
        HR=copy(H0)
        GR=ITensorMPOCompression.orthogonalize!(HR;orth=right)
        @test is_regular_form(HR)
        @test is_orthogonal(HR,right)
        for n in 1:N
            @test norm(GR[n-1]*HR[n]-H0[n]*GR[n]) ≈ 0.0 atol = 1e-14
        end   
        HR1=copy(HL) 
        G=ITensorMPOCompression.orthogonalize!(HR1;orth=right)
        @test is_regular_form(HR1)
        @test is_orthogonal(HR1,right)
        for n in 1:N
            D1=G[n-1]*HR1[n]
            @assert order(D1)==4
            D2=HL[n]*G[n]
            @assert order(D2)==4
            @test norm(G[n-1]*HR1[n]-HL[n]*G[n]) ≈ 0.0 atol = 1e-14
        end   
    end
end

@testset "Truncate/Compress InfiniteMPO 2-body Hamiltonians" begin
    initstate(n) = "↑"
    for N in [1,2,4], NNN in [2,4,8] #3 site unit cell fails for qns=true.
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=false)
        H=make_transIsing_MPO(si,NNN,0.0,lower,1.0;pbc=true)
        @test is_regular_form(H)
        H0=InfiniteMPO(H.data)
        #
        #  Do truncate outputting left ortho Hamiltonian
        #
        HL=copy(H0)
        Ss,ss,HR=truncate!(HL;orth=left,cutoff=1e-15,h_mirror=true)
        # @show Ss ss 
        # @pprint(HL[1])
        @test is_regular_form(HL)
        @test is_orthogonal(HL,left)
        #
        #  Now test guage relations using the diagonal singular value matrices
        #  as the gauge transforms.
        #
        for n in 1:N
            @test norm(Ss[n-1]*HR[n]-HL[n]*Ss[n]) ≈ 0.0 atol = 1e-14
        end    
        #
        #  Do truncate from H0 outputting right ortho Hamiltonian
        #
        HR=copy(H0)
        Ss,ss,HL=truncate!(HR;orth=right,cutoff=1e-15,h_mirror=true)
        #@pprint(HR[1])
        @test is_regular_form(HR)
        @test is_orthogonal(HR,right)
        for n in 1:N
            #@show inds(Ss[n-1]) inds(HR[n],tags="Link") inds(HL[n],tags="Link") inds(Ss[n])
            @test norm(Ss[n-1]*HR[n]-HL[n]*Ss[n]) ≈ 0.0 atol = 1e-14
        end   

        
    end
end

nothing

