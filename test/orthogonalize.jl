using ITensors
using ITensorMPOCompression
using Revise
using Test
import ITensorMPOCompression.orthogonalize!

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
# println("-----------Start--------------")


@testset "Upper, lower, regular detections" begin
    N=6
    NNN=4
    hx=0.5
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    #
    #  test lower triangular MPO 
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower) 
    @test is_upper_lower(H   ,lower,eps)
    @test is_lower_regular_form(H,eps)
    W=H[2]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>1,c=>2) #stuff any op on the top row
    @test !is_lower_regular_form(W,eps)
    W=H[3]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>2,c=>dim(c)) #stuff any op on the right column
    @test !is_lower_regular_form(W,eps)
    W=H[4]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>2,c=>2) #stuff any op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular
    W=H[5]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Id=op(is,"Id")
    assign!(W,Id,r=>2,c=>2) #stuff unit op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular, but should see a warning
    # at this point the whole H should fail since we stuffed ops in the all wrong places.
    @test !is_lower_regular_form(H,eps)


end

test_combos=[
    (make_transIsing_MPO,lower),
    (make_transIsing_MPO,upper),
    (make_transIsing_AutoMPO,lower),
    (make_Heisenberg_AutoMPO,lower)
]

@testset "Bring dense $(test_combo[2]) MPO into $lr canonical form" for test_combo in test_combos, lr in [left,right]
    N=10
    NNN=4
    eps=1e-14
    hx=0.0
    ms=matrix_state(test_combo[2],lr )
    makeH=test_combo[1]
    sites = siteinds("SpinHalf", N;conserve_qns=false)
    psi=randomMPS(sites)
    H=makeH(sites,NNN,hx,ms.ul) 
    #@show inds(H[1])
    @test is_regular_form(H   ,ms.ul,eps)
    E0=inner(psi',H,psi)
    orthogonalize!(H;orth=ms.lr,epsrr=1e-12)
    E1=inner(psi',H,psi)
    @test E0 ≈ E1 atol = 1e-14
    @test is_regular_form(H,ms.ul,eps)
    @test  is_canonical(H,ms,eps)
    @test !is_canonical(H,mirror(ms),eps)    
end 

test_combos=[
    (make_transIsing_MPO,lower),
    (make_transIsing_MPO,upper),
    # (make_transIsing_AutoMPO,lower),
    # (make_Heisenberg_AutoMPO,lower)
]

@testset "Bring block sparse $(test_combo[2]) MPO into $lr canonical form" for test_combo in test_combos, lr in [left,right]
    N=10
    NNN=4
    eps=1e-14
    hx=0.0
    ms=matrix_state(test_combo[2],lr )
    makeH=test_combo[1]
    sites = siteinds("SpinHalf", N;conserve_qns=true)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    H=makeH(sites,NNN,hx,ms.ul) 
    @test is_regular_form(H   ,ms.ul,eps)
    E0=inner(psi',H,psi)
    orthogonalize!(H,ms.ul;orth=ms.lr,epsrr=1e-12)
    E1=inner(psi',H,psi)
    @test E0 ≈ E1 atol = 1e-14
    @test is_regular_form(H,ms.ul,eps)
    @test  is_canonical(H,ms,eps)
    @test !is_canonical(H,mirror(ms),eps)    
end
 
@testset "Compare $ul tri rank reduction with AutoMPO, QNs=$qns" for ul in [lower,upper],qns in [false,true]
N=13
sites = siteinds("SpinHalf", N;conserve_qns=qns)
for NNN in 3:N-1
    Hauto=make_transIsing_AutoMPO(sites,NNN,0.0,ul) 
    Dw_auto=get_Dw(Hauto)
    Hr=make_transIsing_MPO(sites,NNN,0.0,ul) 
    orthogonalize!(Hr;orth=right,epsrr=1e-12) #sweep left to right
    @test is_canonical(Hr,matrix_state(ul,right),1e-12)
    @test get_Dw(Hr)==Dw_auto
    Hl=make_transIsing_MPO(sites,NNN,0.0,ul) 
    orthogonalize!(Hl;orth=left,epsrr=1e-12) #sweep right to left
    @test is_canonical(Hl,matrix_state(ul,left),1e-12)
    @test get_Dw(Hl)==Dw_auto
end  
end 

nothing
