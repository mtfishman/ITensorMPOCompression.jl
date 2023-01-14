using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf

verbose=false #verbose at the outer test level
verbose1=false #verbose inside orth algos

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
# println("-----------Start--------------")
@testset verbose=verbose "Orthogonalize" begin

@testset "Upper, lower, regular detections" begin
    N=6
    NNN=4
    model_kwargs = (hx=0.5, ul=lower)
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    #
    #  test lower triangular MPO 
    #
    H=make_transIsing_MPO(sites,NNN;model_kwargs...) 
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
    model_kwargs = (ul=test_combo[2], )
    ms=matrix_state(test_combo[2],lr )
    makeH=test_combo[1]
    sites = siteinds("SpinHalf", N;conserve_qns=false)
    psi=randomMPS(sites)
    H=makeH(sites,NNN;model_kwargs...) 
    #@show inds(H[1])
    @test is_regular_form(H   ,ms.ul,eps)
    E0=inner(psi',H,psi)
    orthogonalize!(H;verbose=verbose1,orth=ms.lr,rr_cutoff=1e-12)
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
    ms=matrix_state(test_combo[2],lr )
    makeH=test_combo[1]
    sites = siteinds("SpinHalf", N;conserve_qns=true)
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    H=makeH(sites,NNN;ul=test_combo[2]) 
    @test is_regular_form(H   ,ms.ul,eps)
    E0=inner(psi',H,psi)
    orthogonalize!(H;verbose=verbose1,orth=ms.lr,rr_cutoff=1e-12)
    E1=inner(psi',H,psi)
    @test E0 ≈ E1 atol = 1e-14
    @test is_regular_form(H,ms.ul,eps)
    @test  is_canonical(H,ms,eps)
    @test !is_canonical(H,mirror(ms),eps)    
end
 
@testset "Compare $ul tri rank reduction with AutoMPO, QNs=$qns" for ul in [lower,upper],qns in [false,true]
    N=14
    sites = siteinds("SpinHalf", N;conserve_qns=qns)
    for NNN in 3:div(N,2)
        Hauto=make_transIsing_AutoMPO(sites,NNN;ul=ul) 
        Dw_auto=get_Dw(Hauto)
        Hr=make_transIsing_MPO(sites,NNN;ul=ul) 
        orthogonalize!(Hr;verbose=verbose1,rr_cutoff=1e-12) #sweep left to right
        @test get_Dw(Hr)==Dw_auto
        Hl=make_transIsing_MPO(sites,NNN;ul=ul) 
        orthogonalize!(Hl;verbose=verbose1,rr_cutoff=1e-12) #sweep right to left
        @test get_Dw(Hl)==Dw_auto
    end  
end 

@testset "Orthogonalize iMPO Check gauge relations, ul=$ul, qbs=$qns" for ul in [lower,upper], qns in [false,true]
    initstate(n) = "↑"
    if verbose
        @printf "               Dw     Dw    Dw    Dw\n"
        @printf " Ncell  NNN  uncomp. left  right  LR\n"
    end
    for N in [1,2,4], NNN in [2,4] #3 site unit cell fails for qns=true.
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=qns)

        H0=make_transIsing_iMPO(si,NNN;ul=ul)
        @test is_regular_form(H0)
        Dw0=Base.max(get_Dw(H0)...)

        HL=copy(H0)
        @test is_regular_form(HL)
        GL=orthogonalize!(HL;verbose=verbose1,orth=left,max_sweeps=1)
        DwL=Base.max(get_Dw(HL)...)
        @test is_regular_form(HL)
        @test is_orthogonal(HL,left)
        for n in 1:N
            @test norm(HL[n]*GL[n]-GL[n-1]*H0[n]) ≈ 0.0 atol = 1e-14 
        end
        HR=copy(H0)
        GR=orthogonalize!(HR;verbose=verbose1,orth=right,max_sweeps=1)
        DwR=Base.max(get_Dw(HR)...)
        @test is_regular_form(HR)
        @test is_orthogonal(HR,right)
        for n in 1:N
            @test norm(GR[n-1]*HR[n]-H0[n]*GR[n]) ≈ 0.0 atol = 1e-14
        end   
        HR1=copy(HL) 
        G=orthogonalize!(HR1;verbose=verbose1,orth=right,max_sweeps=1)
        DwLR=Base.max(get_Dw(HR1)...)
        @test is_regular_form(HR1)
        @test is_orthogonal(HR1,right)
        for n in 1:N
            D1=G[n-1]*HR1[n]
            @assert order(D1)==4
            D2=HL[n]*G[n]
            @assert order(D2)==4
            @test norm(G[n-1]*HR1[n]-HL[n]*G[n]) ≈ 0.0 atol = 1e-14
        end
        if verbose
            @printf " %4i %4i   %4i   %4i  %4i  %4i\n" N NNN Dw0 DwL DwR DwLR
        end

    end
end

end
nothing
