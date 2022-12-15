using ITensors
using ITensorMPOCompression
using Test
using Revise

@testset "ITensorMPOCompression.jl" begin
    include("blocking.jl")
    include("qx_unittests.jl")
    include("hamiltonians.jl")
    include("orthogonalize.jl")
    include("truncate.jl")
end
nothing #suppress messy dump from Test
