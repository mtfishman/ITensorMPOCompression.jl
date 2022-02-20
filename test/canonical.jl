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
    @test detect_upper_lower(H[1],eps)==lower
    @test detect_upper_lower(H,eps)==lower
    @test is_regular_form(H[1],eps)
    @test is_regular_form(H,eps)
    W=H[1]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,r=>1,c=>2,Sz) #stuff any op on the top row
    @test !is_regular_form(W,eps)
    W=H[2]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,r=>2,c=>dim(c),Sz) #stuff any op on the right column
    @test !is_regular_form(W,eps)
    W=H[3]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,r=>2,c=>2,Sz) #stuff any op on the diag
    @test is_regular_form(W,eps) #this one should still be regular
    W=H[4]
    d,n,r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Id=delta(is,is')
    assign!(W,r=>2,c=>2,Id) #stuff unit op on the diag
    @test is_regular_form(W,eps) #this one should still be regular, but should see a warning
    # at this point the whole H should fail since we stuffed ops in the all wrong places.
    @test !is_regular_form(H,eps)


    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=false) 
    @test !has_pbc(H)
    @test detect_upper_lower(H[2],eps)==lower
    @test detect_upper_lower(H,eps)==lower
    @test is_regular_form(H[2],eps)
    @test is_regular_form(H[1],eps) #should handle edge row/col vectors
    @test is_regular_form(H[N],eps)
    @test is_regular_form(H,eps)

    H=make_transIsing_MPO(sites,NNN,hx,upper,pbc=true) 
    @test has_pbc(H)
    @test detect_upper_lower(H[1],eps)==upper
    @test detect_upper_lower(H,eps)==upper
    @test is_regular_form(H[1],eps)
    @test is_regular_form(H,eps)
   

end

@testset "Bring MPO into canonical form" begin
    N=5
    NNN=2
    hx=0.5
    eps=1e-15
    msl=matrix_state(lower,left )
    msr=matrix_state(lower,right)

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # left canonical for lower triangular MPO
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0=inner(psi',to_openbc(H),psi)
    @test detect_upper_lower(H,eps)==lower
    
    canonical!(H,left)
    
    E1=inner(psi',to_openbc(H),psi)
    @test abs(E0-E1)<1e-14
    @test detect_upper_lower(H,eps)==lower
    @test  is_canonical(H,msl,eps)
    @test !is_canonical(H,msr,eps)
    #
    # right canonical for lower triangular MPO
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0=inner(psi',to_openbc(H),psi)

    canonical!(H,right)
    @test detect_upper_lower(H,eps)==lower
    
    E1=inner(psi',to_openbc(H),psi)
    @test abs(E0-E1)<1e-14
    @test !is_canonical(H,msl,eps)
    @test  is_canonical(H,msr,eps)
    #
    #  two more sweeps just make sure nothing get messed up.
    #
    canonical!(H,left)
    canonical!(H,right)
    E2=inner(psi',to_openbc(H),psi)
    @test abs(E0-E2)<1e-14
    @test detect_upper_lower(H,eps)==lower
    @test !is_canonical(H,msl,eps)
    @test  is_canonical(H,msr,eps)


    #
    # Make sure upper triangular MPO has the same energy as the lower version had
    #
    H=make_transIsing_MPO(sites,NNN,hx,upper,pbc=true)
    @test detect_upper_lower(H,eps)==upper
    Eupper=inner(psi',to_openbc(H),psi)
    @test abs(E0-Eupper)<1e-14


end