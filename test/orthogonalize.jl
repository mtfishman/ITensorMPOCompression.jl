using ITensorMPOCompression
using Revise
using Test
import ITensorMPOCompression.orthogonalize!

include("hamiltonians.jl")

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
println("-----------Start--------------")


@testset "Upper, lower, regular detections" begin
    N=6
    NNN=4
    hx=0.5
    J=1.0
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    #
    #  test lower triangular MPO 
    #
    H=make_transIsing_MPO(sites,NNN,J,hx,lower) 
    @test is_upper_lower(H   ,lower,eps)
    @test is_lower_regular_form(H,eps)
    W=H[2]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>1,c=>2) #stuff any op on the top row
    @test !is_lower_regular_form(W,eps)
    W=H[3]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>2,c=>dim(c)) #stuff any op on the right column
    @test !is_lower_regular_form(W,eps)
    W=H[4]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>2,c=>2) #stuff any op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular
    W=H[5]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Id=op(is,"Id")
    assign!(W,Id,r=>2,c=>2) #stuff unit op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular, but should see a warning
    # at this point the whole H should fail since we stuffed ops in the all wrong places.
    @test !is_lower_regular_form(H,eps)


end

function test_canonical(makeH,N::Int64,NNN::Int64,ms::matrix_state)
    eps=1e-14
    J=1.0
    hx=0.5
    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    H=makeH(sites,NNN,J,hx,ms.ul) 
    @test is_regular_form(H   ,ms.ul,eps)
    E0=inner(psi',H,psi)
    orthogonalize!(H;dir=ms.lr,epsrr=1e-12)
    E1=inner(psi',H,psi)
    @test abs(E0-E1)<1e-14
    @test is_regular_form(H,ms.ul,eps)
    @test  is_canonical(H,ms,eps)
    @test !is_canonical(H,mirror(ms),eps)    
end

#
@testset "Bring MPO into canonical form" begin
  
    N=10
    NNN=4
   
    test_canonical(make_transIsing_MPO    ,N,NNN,matrix_state(lower,left ))
    test_canonical(make_transIsing_MPO    ,N,NNN,matrix_state(lower,right))
    test_canonical(make_transIsing_MPO    ,N,NNN,matrix_state(upper,left ))
    test_canonical(make_transIsing_MPO    ,N,NNN,matrix_state(upper,right))
    test_canonical(make_transIsing_AutoMPO,N,NNN,matrix_state(lower,left ))
    test_canonical(make_transIsing_AutoMPO,N,NNN,matrix_state(lower,right))
    test_canonical(make_Heisenberg_AutoMPO,N,NNN,matrix_state(lower,left ))
    test_canonical(make_Heisenberg_AutoMPO,N,NNN,matrix_state(lower,right))
    
end 