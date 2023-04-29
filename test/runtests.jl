using ITensors
using ITensorMPOCompression
using Test
using Revise

@testset "ITensorMPOCompression.jl" begin
  @testset verbose = true "$filename" for filename in [
    "blocking.jl",
    "gauge_fix.jl",
    "qx_unittests.jl",
    "hamiltonians.jl",
    "orthogonalize.jl",
    "truncate.jl",
  ]
    print("$filename: ")
    @time include(filename)
  end
end
nothing #suppress messy dump from Test
