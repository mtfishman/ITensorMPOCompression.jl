using ITensors
using ITensorMPOCompression
using Test

@testset "ITensorMPOCompression.jl" begin
    include("qx_unittests.jl")
    include("blocking.jl")
    include("mpopbc.jl")
    include("canonical.jl")
end
