using ITensors
using ITensorMPOCompression
using Test
using Revise

@testset "ITensorMPOCompression.jl" begin
    include("qx_unittests.jl")
    include("blocking.jl")
    include("hamiltonians.jl")
    include("orthogonalize.jl")
    include("truncate.jl")
end
nothing #suppress messy dump from Test
