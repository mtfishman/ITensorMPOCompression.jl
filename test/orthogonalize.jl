using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise
using Test
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

verbose=false #verbose at the outer test level
verbose1=false #verbose inside orth algos

@testset verbose=verbose "Orthogonalize" begin

    # @testset "Upper, lower, regular detections" begin
    #     N=6
    #     NNN=4
    #     model_kwargs = (hx=0.5, ul=lower)
    #     eps=1e-15
    #     sites = siteinds("SpinHalf", N)
    #     #
    #     #  test lower triangular MPO 
    #     #
    #     H=make_transIsing_MPO(sites,NNN;model_kwargs...) 
    #     @test is_upper_lower(H   ,lower,eps)
    #     @test is_lower_regular_form(H,eps)
    #     W=H[2]
    #     r,c=parse_links(W)
    #     is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    #     Sz=op(is,"Sz")
    #     assign!(W,Sz,r=>1,c=>2) #stuff any op on the top row
    #     @test !is_lower_regular_form(W,eps)
    #     @test !is_upper_lower(H   ,lower,eps)
    #     @test !is_upper_lower(H   ,upper,eps)
    #     W=H[3]
    #     r,c=parse_links(W)
    #     is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    #     Sz=op(is,"Sz")
    #     assign!(W,Sz,r=>2,c=>dim(c)) #stuff any op on the right column
    #     @test !is_lower_regular_form(W,eps)
    #     @test !is_upper_lower(H   ,lower,eps)
    #     @test !is_upper_lower(H   ,upper,eps)
    #     W=H[4]
    #     r,c=parse_links(W)
    #     is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    #     Sz=op(is,"Sz")
    #     assign!(W,Sz,r=>2,c=>2) #stuff any op on the diag
    #     @test is_lower_regular_form(W,eps) #this one should still be regular
    #     @test  is_upper_lower(W   ,lower,eps)
    #     @test !is_upper_lower(W   ,upper,eps)
    #     W=H[5]
    #     r,c=parse_links(W)
    #     is=filterinds(W,tags="Site")[1] #get any site index for generating operators
    #     Id=op(is,"Id")
    #     assign!(W,Id,r=>2,c=>2) #stuff unit op on the diag
    #     @test is_lower_regular_form(W,eps) #this one should still be regular, but should see a warning
    #     @test  is_upper_lower(W   ,lower,eps)
    #     @test !is_upper_lower(W   ,upper,eps)
    #     # at this point the whole H should fail since we stuffed ops in the all wrong places.
    #     @test !is_lower_regular_form(H,eps)


    # end

    models=[
        [make_transIsing_MPO,"S=1/2",true],
        [make_transIsing_AutoMPO,"S=1/2",true],
        [make_Heisenberg_AutoMPO,"S=1/2",true],
        [make_Heisenberg_AutoMPO,"S=1",true],
        [make_Hubbard_AutoMPO,"Electron",false],
    ]

    @testset "Ac/Ab block respecting decomposition $(model[1]), qns=$qns, ul=$ul" for model in models, qns in [false, true], ul=[lower,upper]
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
        @test isortho(Hrf,left)
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
        @test isortho(Hrf,right)
        @test is_gauge_fixed(Hrf,eps) #Should still be gauge fixed
        #
        # #  Expectation value check.
        # #
        E2=inner(psi',MPO(Hrf),psi)
        @test E0 ≈ E2 atol = eps
    end

    function ITensorMPOCompression.get_Dw(H::reg_form_MPO)
        return get_Dw(MPO(H))
    end

    @testset "Compare Dws for Ac orthogonalized hand built MPO, vs Auto MPO, NNN=$NNN, ul=$ul, qns=$qns" for NNN in [1,5,8,12], ul in [lower,upper], qns in [false,true]
        N=2*NNN+4 
        sites = siteinds("S=1/2",N,conserve_qns=qns);
        Hhand=reg_form_MPO(make_transIsing_MPO(sites,NNN;ul=ul))
        Hauto=make_transIsing_AutoMPO(sites,NNN;ul=ul)
        ac_orthogonalize!(Hhand,right)
        ac_orthogonalize!(Hhand,left)
        @test get_Dw(Hhand)==get_Dw(Hauto)
    end

    models=[
        (make_transIsing_iMPO,"S=1/2"),
        (make_transIsing_AutoiMPO,"S=1/2"),
        (make_Heisenberg_AutoiMPO,"S=1/2"),
        (make_Heisenberg_AutoiMPO,"S=1"),
        (make_Hubbard_AutoiMPO,"Electron")
     ]
    
    @testset "Orthogonalize iMPO Check gauge relations, H=$(model[1]), ul=$ul, qbs=$qns, N=$N, NNN=$NNN" for model in models, ul in [lower], qns in [false,true], N in [1,2,3,4], NNN in [1,2,4,7]
       
        eps=NNN*1e-14
        initstate(n) = "↑"
        si = infsiteinds(model[2], N; initstate, conserve_qns=qns)
        H0=reg_form_iMPO(model[1](si,NNN;ul=ul))
        HL=copy(H0)
        @test is_regular_form(HL)
        GL=ac_orthogonalize!(HL,left;verbose=verbose1)
        DwL=Base.max(get_Dw(HL)...)
        @test is_regular_form(HL)
        @test isortho(HL,left)
        @test check_ortho(HL,left) #expensive does V_dagger*V=Id
        for n in 1:N
            @test norm(HL[n].W*GL[n]-GL[n-1]*H0[n].W) ≈ 0.0 atol = eps
        end

        HR=copy(H0)
        GR=ac_orthogonalize!(HR,right;verbose=verbose1)
        DwR=Base.max(get_Dw(HR)...)
        @test is_regular_form(HR)
        @test isortho(HR,right)
        @test check_ortho(HR,right) #expensive does V_dagger*V=Id
        for n in 1:N
            @test norm(GR[n-1]*HR[n].W-H0[n].W*GR[n]) ≈ 0.0 atol = eps
        end   
        HR1=copy(HL) 
        G=ac_orthogonalize!(HR1,right;verbose=verbose1)
        DwLR=Base.max(get_Dw(HR1)...)
        @test is_regular_form(HR1)
        @test isortho(HR1,right)
        @test check_ortho(HR1,right) #expensive does V_dagger*V=Id
        for n in 1:N
            # D1=G[n-1]*HR1[n].W
            # @assert order(D1)==4
            # D2=HL[n].W*G[n]
            # @assert order(D2)==4
            @test norm(G[n-1]*HR1[n].W-HL[n].W*G[n]) ≈ 0.0 atol = eps
        end


    end

end
nothing
