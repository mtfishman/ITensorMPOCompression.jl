using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf

verbose=true #verbose at the outer test level
verbose1=false #verbose inside orth algos

 using Printf
 Base.show(io::IO, f::Float64) = @printf(io, "%1.1e", f)
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
    @test !is_upper_lower(H   ,lower,eps)
    @test !is_upper_lower(H   ,upper,eps)
    W=H[3]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>2,c=>dim(c)) #stuff any op on the right column
    @test !is_lower_regular_form(W,eps)
    @test !is_upper_lower(H   ,lower,eps)
    @test !is_upper_lower(H   ,upper,eps)
    W=H[4]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Sz=op(is,"Sz")
    assign!(W,Sz,r=>2,c=>2) #stuff any op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular
    @test  is_upper_lower(W   ,lower,eps)
    @test !is_upper_lower(W   ,upper,eps)
    W=H[5]
    r,c=parse_links(W)
    is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    Id=op(is,"Id")
    assign!(W,Id,r=>2,c=>2) #stuff unit op on the diag
    @test is_lower_regular_form(W,eps) #this one should still be regular, but should see a warning
    @test  is_upper_lower(W   ,lower,eps)
    @test !is_upper_lower(W   ,upper,eps)
    # at this point the whole H should fail since we stuffed ops in the all wrong places.
    @test !is_lower_regular_form(H,eps)


end

models=[
    [make_transIsing_MPO,"S=1/2",true],
    [make_transIsing_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1/2",true],
    [make_Heisenberg_AutoMPO,"S=1",true],
    [make_Hubbard_AutoMPO,"Electron",false],
]

@testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false,true], ul=[lower,upper]
    eps=1e-14
    pre_fixed=model[3] #Hamiltonian starts gauge fixed
    N=10 #5 sites
    NNN=7 #Include 6nd nearest neighbour interactions
    sites = siteinds(model[2],N,conserve_qns=qns);
    Hrf=reg_form_MPO(model[1](sites,NNN;ul=ul))
    state=[isodd(n) ? "Up" : "Dn" for n=1:N]
    psi=randomMPS(sites,state)
    E0=inner(psi',MPO(Hrf),psi)

    @test is_regular_form(Hrf)
    #
    #  Left->right sweep
    #
    lr=left
    @test pre_fixed == is_gauge_fixed(Hrf,eps) 
    NNN>=7 && ac_orthogonalize!(Hrf,right)
    ac_orthogonalize!(Hrf,left)
    @test is_regular_form(Hrf)
    @test check_ortho(Hrf,left)
    NNN<7 && @test is_gauge_fixed(Hrf,eps) #Now everything should be fixed, unless NNN is big
    #
    #  Expectation value check.
    #
    E1=inner(psi',MPO(Hrf),psi)
    @test E0 ≈ E1 atol = eps
    #
    #  Right->left sweep
    #
    ac_orthogonalize!(Hrf,right)
    @test is_regular_form(Hrf)
    @test check_ortho(Hrf,right)
    @test is_gauge_fixed(Hrf,eps) #Should still be gauge fixed
    #
    # #  Expectation value check.
    # #
    E2=inner(psi',MPO(Hrf),psi)
    @test E0 ≈ E2 atol = eps
end



# @testset "Bring $(test_combo[1]), $ul reg. form MPO into $lr canonical form, qns=$qns" for test_combo in test_combos, lr in [left,right], ul in [lower,upper], qns=[false,true]
#     N=10
#     NNN=7
#     eps=2e-14
#     model_kwargs = (ul=ul, cutoff=-1.0) #cutoff=-1.0 causes AutoMPO to do less compression.
#     ms=matrix_state(ul,lr )
#     makeH=test_combo[1]
#     sites = siteinds(test_combo[2], N;conserve_qns=qns)
#     state=[isodd(n) ? "Up" : "Dn" for n=1:N]
#     psi=randomMPS(sites,state)
#     H=makeH(sites,NNN;model_kwargs...) 
#     @test is_regular_form(H   ,ms.ul,eps)
#     @test !isortho(H)
#     @test !isortho(H,left)    
#     @test !isortho(H,right)    
#     E0=inner(psi',H,psi)
#     orthogonalize!(H;verbose=verbose1,orth=ms.lr)
#     E1=inner(psi',H,psi)
#     @test E0 ≈ E1 atol = eps
#     @test is_regular_form(H,ms.ul,eps)
#     @test  isortho(H,ms.lr)
#     @test !isortho(H,mirror(ms.lr))
#     @test check_ortho(H,ms.lr) #expensive does V_dagger*V=Id
# end 

# @testset "Compare $ul tri rank reduction with AutoMPO, QNs=$qns" for ul in [lower,upper],qns in [false,true]
#     N=14
#     sites = siteinds("SpinHalf", N;conserve_qns=qns)
#     # The default for rr_cutoff is 1e-15 which is too low to get reduction down
#     # AutoMPO elvels, so we need to use rr_cutoff = 2e-14
#     for NNN in 3:div(N,2)
#         Hauto=make_transIsing_AutoMPO(sites,NNN;ul=ul) 
#         Dw_auto=get_Dw(Hauto)
#         Hr=make_transIsing_MPO(sites,NNN;ul=ul) 
#         orthogonalize!(Hr;verbose=verbose1,rr_cutoff=2e-14) #sweep left to right
#         @test get_Dw(Hr)==Dw_auto
#         Hl=make_transIsing_MPO(sites,NNN;ul=ul) 
#         orthogonalize!(Hl;verbose=verbose1,rr_cutoff=2e-14) #sweep right to left
#         @test get_Dw(Hl)==Dw_auto
#     end  
# end 

# test_combos=[
#     (make_transIsing_iMPO,"S=1/2"),
#     (make_transIsing_AutoiMPO,"S=1/2"),
#     (make_Heisenberg_AutoiMPO,"S=1/2"),
#     (make_Heisenberg_AutoiMPO,"S=1"),
#     (make_Hubbard_AutoiMPO,"Electron")
# ]

# @testset "Orthogonalize iMPO Check gauge relations, H=$(test_combo[1]), ul=$ul, qbs=$qns" for test_combo in test_combos, ul in [lower,upper], qns in [false,true]
#     initstate(n) = "↑"
#     makeH=test_combo[1]
#     if verbose1
#         @printf "               Dw     Dw    Dw    Dw\n"
#         @printf " Ncell  NNN  uncomp. left  right  LR\n"
#     end
#     for N in [1,2,4], NNN in [1,2,4] #3 site unit cell fails for qns=true.
#         si = infsiteinds(test_combo[2], N; initstate, conserve_qns=qns)

#         H0=makeH(si,NNN;ul=ul)
        
#         @test is_regular_form(H0)
#         #@show H0 
#         Dw0=Base.max(get_Dw(H0)...)

#         HL=copy(H0)
#         @test is_regular_form(HL)
#         GL=orthogonalize!(HL;verbose=verbose1,orth=left,max_sweeps=1)
#         DwL=Base.max(get_Dw(HL)...)
#         @test is_regular_form(HL)
#         @test isortho(HL,left)
#         @test check_ortho(HL,left) #expensive does V_dagger*V=Id
#         for n in 1:N
#             @test norm(HL[n]*GL[n]-GL[n-1]*H0[n]) ≈ 0.0 atol = 1e-14 
#         end
#         HR=copy(H0)
#         GR=orthogonalize!(HR;verbose=verbose1,orth=right,max_sweeps=1)
#         DwR=Base.max(get_Dw(HR)...)
#         @test is_regular_form(HR)
#         @test isortho(HR,right)
#         @test check_ortho(HR,right) #expensive does V_dagger*V=Id
#         for n in 1:N
#             @test norm(GR[n-1]*HR[n]-H0[n]*GR[n]) ≈ 0.0 atol = 1e-14
#         end   
#         HR1=copy(HL) 
#         G=orthogonalize!(HR1;verbose=verbose1,orth=right,max_sweeps=1)
#         DwLR=Base.max(get_Dw(HR1)...)
#         @test is_regular_form(HR1)
#         @test isortho(HR1,right)
#         @test check_ortho(HR1,right) #expensive does V_dagger*V=Id
#         for n in 1:N
#             D1=G[n-1]*HR1[n]
#             @assert order(D1)==4
#             D2=HL[n]*G[n]
#             @assert order(D2)==4
#             @test norm(G[n-1]*HR1[n]-HL[n]*G[n]) ≈ 0.0 atol = 1e-14
#         end
#         if verbose1
#             @printf " %4i %4i   %4i   %4i  %4i  %4i\n" N NNN Dw0 DwL DwR DwLR
#         end

#     end
# end

end
nothing
