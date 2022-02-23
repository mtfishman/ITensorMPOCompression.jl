using ITensors
using ITensorMPOCompression
using Test
using Revise

#@testset "ITensorMPOCompression.jl" begin
    include("qx_unittests.jl")
    include("blocking.jl")
    include("mpopbc.jl")
    include("canonical.jl")
    include("compress.jl")
#end
