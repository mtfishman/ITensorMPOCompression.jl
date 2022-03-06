using ITensors
using ITensorMPOCompression
using Test
using Revise

ITensors.ITensors.disable_debug_checks()
#ITensors.ITensors.enable_debug_checks()
@testset "ITensorMPOCompression.jl" begin
    include("qx_unittests.jl")
    include("blocking.jl")
    include("mpopbc.jl")
    include("orthogonalize.jl")
    include("truncate.jl")
end
