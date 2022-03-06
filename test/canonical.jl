using ITensorMPOCompression
using Revise
using Test

include("hamiltonians.jl")

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
println("-----------Start--------------")


@testset "Upper, lower, pbc, regular detections" begin
    N=5
    NNN=4
    hx=0.5
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    #
    #  test lower triangular MPO with periodic boundary conditions (pbc)
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true) 
    @test has_pbc(H)
    @test is_upper_lower(H[1],lower,eps)
    @test is_upper_lower(H   ,lower,eps)
    @test is_lower_regular_form(H[1],eps)
    @test is_lower_regular_form(H,eps)
    W=H[1]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,r=>1,c=>2,Sz) #stuff any op on the top row
    @test !is_lower_regular_form(W,eps)
    W=H[2]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,r=>2,c=>dim(c),Sz) #stuff any op on the right column
    @test !is_lower_regular_form(W,eps)
    W=H[3]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,r=>2,c=>2,Sz) #stuff any op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular
    W=H[4]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Id=delta(is,is')
    assign!(W,r=>2,c=>2,Id) #stuff unit op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular, but should see a warning
    # at this point the whole H should fail since we stuffed ops in the all wrong places.
    @test !is_lower_regular_form(H,eps)


    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=false) 
    @test !has_pbc(H)
    @test is_upper_lower(H[2],lower,eps)
    @test is_upper_lower(H   ,lower,eps)
    @test is_lower_regular_form(H[2],eps)
    @test is_lower_regular_form(H[1],eps) #should handle edge row/col vectors
    @test is_lower_regular_form(H[N],eps)
    @test is_lower_regular_form(H,eps)

    H=make_transIsing_MPO(sites,NNN,hx,upper,pbc=true) 
    @test has_pbc(H)
    @test is_upper_lower(H[1],upper,eps)
    @test is_upper_lower(H   ,upper,eps)
    @test is_upper_regular_form(H[1],eps)
    @test is_upper_regular_form(H,eps)
end

function test_canonical(N::Int64,NNN::Int64,hx::Float64,ms::matrix_state)
    eps=1e-14
    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    H=make_transIsing_MPO(sites,NNN,hx,ms.ul,pbc=true) 
    @test has_pbc(H)
    @test is_upper_lower(H[1],ms.ul,eps)
    @test is_upper_lower(H   ,ms.ul,eps)
    @test is_regular_form(H[1],ms.ul,eps)
    @test is_regular_form(H   ,ms.ul,eps)
    E0=inner(psi',to_openbc(H),psi)
    orthogonalize!(H;dir=ms.lr)
    E1=inner(psi',to_openbc(H),psi)
    @test abs(E0-E1)<1e-14
    @test is_upper_lower(H,ms.ul,eps)
    @test  is_canonical(H,ms,eps)
    @test !is_canonical(H,mirror(ms),eps)    
    #
    #  two more sweeps just make sure nothing gets messed up.
    #
    orthogonalize!(H;dir=mirror(ms.lr))
    orthogonalize!(H;dir=ms.lr)
    E2=inner(psi',to_openbc(H),psi)
    @test abs(E0-E2)<1e-14
    @test is_regular_form(H,ms.ul,eps)
    @test !is_canonical(H,mirror(ms),eps)
    @test  is_canonical(H,ms,eps)
end


@testset "Bring MPO into canonical form" begin
  
    N=5
    NNN=4
    hx=0.5
    test_canonical(N,NNN,hx,matrix_state(lower,left ))
    test_canonical(N,NNN,hx,matrix_state(lower,right))
    test_canonical(N,NNN,hx,matrix_state(upper,left ))
    test_canonical(N,NNN,hx,matrix_state(upper,right))


end