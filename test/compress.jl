using ITensorMPOCompression
using Revise
using Test

include("hamiltonians.jl")

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
println("-----------Start--------------")

function make_RL(r::Index,c::Index,ms::matrix_state,swap::Bool)::ITensor
    @assert ms.ul==upper || ms.ul==lower
    @assert ms.lr==left  || ms.lr==right
    A= swap ? randomITensor(c,r) : randomITensor(r,c)
    if ms.ul==lower
        dc=max(0,dim(c)-dim(r))
        for i in 1:dim(r)
            for j in i+1+dc:dim(c)
                A[r=>i,c=>j]=0.0
            end
        end
    else #must be upper
        dr=max(0,dim(r)-dim(c))
        for j in 1:dim(c)
            for i in j+1+dr:dim(r)
                A[r=>i,c=>j]=0.0
            end
        end
    end
    # now we need to zero parts of the perimeter depending on ms
    if ms.ul==lower
        if ms.lr==left 
            #zero left and right columns
            for i in 1:dim(r)
                A[r=>i,c=>1     ]=0.0
                A[r=>i,c=>dim(c)]=0.0
            end
        else #must be right
            #zero top and  bottom rows
            for i in 1:dim(c)
                A[r=>1     ,c=>i]=0.0
                A[r=>dim(r),c=>i]=0.0
            end
        end
    else #must be upper
        if ms.lr==left 
            #zero left and right columns
            for i in 1:dim(r)
                A[r=>i,c=>1     ]=0.0
                A[r=>i,c=>dim(c)]=0.0
            end
        else #must be right
           #zero top and  bottom rows
           for i in 1:dim(c)
                A[r=>1     ,c=>i]=0.0
                A[r=>dim(r),c=>i]=0.0
            end
        end

    end
    # put 1's in the diagonal corners
    A[r=>1     ,c=>1     ]=1.0
    A[r=>dim(r),c=>dim(c)]=1.0

    return A
end

function test_getM(r,c,lr::orth_type,swap::Bool)
    if lr==left
        @assert dim(r)>=dim(c)
    #    println("---------- lower left -----------")
        ms=matrix_state(lower,left)
        L=make_RL(r,c,ms,swap)
        M,L_prime,im=getM(L,ms.lr)
        if hastags(M,tags(c))
            Mplus=grow(im,M,c)
        elseif hastags(M,tags(r))
            Mplus=grow(r,M,im)
        else
            @assert false
        end
        Ltest=L-Mplus*L_prime
        @test norm(Ltest)==0.0

    #    println("---------- upper left -----------")
        ms=matrix_state(upper,left)
        R=make_RL(r,c,ms,swap)
        M,R_prime,im=getM(R,ms.lr)
        if hastags(M,tags(c))
            Mplus=grow(im,M,c)
        elseif hastags(M,tags(r))
            Mplus=grow(r,M,im)
        else
            @assert false
        end
        Rtest=R-Mplus*R_prime
        @test norm(Rtest)==0.0
    end
    if lr==right
        @assert dim(c)>=dim(r)
    #    println("---------- lower right -----------")
        ms=matrix_state(lower,right)
        L=make_RL(r,c,ms,swap)
        M,L_prime,im=getM(L,ms.lr)
        if hastags(M,tags(c))
            Mplus=grow(im,M,c)
        elseif hastags(M,tags(r))
            Mplus=grow(r,M,im)
        else
            @assert false
        end
        Ltest=L-L_prime*Mplus
        @test norm(Ltest)==0.0


    #    println("---------- upper right -----------")
        ms=matrix_state(upper,right)
        R=make_RL(r,c,ms,swap)
        M,R_prime,im=getM(R,ms.lr)
        if hastags(M,tags(c))
            Mplus=grow(im,M,c)
        elseif hastags(M,tags(r))
            Mplus=grow(r,M,im)
        else
            @assert false
        end
        Rtest=R-R_prime*Mplus
        @test norm(Rtest)==0.0
    end
    
end

@testset "getM and grow(M)" begin
    test_getM(Index(5,"Link,qx"),Index(5,"Link,l=1"),left,false)
    test_getM(Index(5,"Link,qx"),Index(5,"Link,l=1"),left,true)
    test_getM(Index(5,"Link,l=4"),Index(5,"Link,qx"),right,false)
    test_getM(Index(5,"Link,l=4"),Index(5,"Link,qx"),right,true)
    test_getM(Index(7,"Link,qx" ),Index(5,"Link,l=1"),left,false)
    test_getM(Index(7,"Link,qx" ),Index(5,"Link,l=1"),left,true)
    test_getM(Index(5,"Link,l=4"),Index(7,"Link,qx" ),right,false)
    test_getM(Index(5,"Link,l=4"),Index(7,"Link,qx" ),right,true)
end

@testset "Compress one site" begin
    N=5
    NNN=2
    hx=0.5
    eps=1e-15
    epsSVD=1e-15
    msl=matrix_state(lower,left )
    msr=matrix_state(lower,right)

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # Make right canonical, then compress to left canonical
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0l=inner(psi',to_openbc(H),psi)
    @test detect_upper_lower(H,eps)==lower
    canonical!(H,right)

    E1l=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E1l)<1e-14
    @test detect_upper_lower(H,eps)==lower
    @test  is_canonical(H,msr,eps)

    W,L=compress(H[1],left,epsSVD)
    @test detect_upper_lower(H,eps)==lower
    @test detect_upper_lower(W,eps)==lower
    @test norm(H[1]-W*L)<eps
    H[1]=W
    H[2]=L*H[2]
    @test detect_upper_lower(H[2],eps)==lower
    @test  is_canonical(H[1],msl,eps)
    # make sure the energy in unchanged
    E2l=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E2l)<1e-14
    #
    # Make left canonical, then compress to right canonical
    #
    # H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    # E0r=inner(psi',to_openbc(H),psi)
    # @test detect_upper_lower(H,eps)==lower
    # canonical!(H,left)

    # E1r=inner(psi',to_openbc(H),psi)
    # @test abs(E0r-E1r)<1e-14
    # @test detect_upper_lower(H,eps)==lower
    # @test  is_canonical(H,msl,eps)

    # W,L=compress(H[N],right,epsSVD)
    # @test detect_upper_lower(H,eps)==lower
    # @test detect_upper_lower(W,eps)==lower
    # @show inds(H[N]) inds(W) inds(L)
    # @test norm(H[N]-L*W)<eps
    # H[N]=W
    # H[N-1]=H[N-1]*L
    # @test detect_upper_lower(H[N-1],eps)==lower
    # @test  is_canonical(H[N],msl,eps)
    # # make sure the energy in unchanged
    # E2r=inner(psi',to_openbc(H),psi)
    # @test abs(E0r-E2r)<1e-14
end

# function test_one_sweep(N::Int64,NNN::Int64,hx::Float64,ul::tri_type,epsSVD::Float64,eps::Float64)
#     msl=matrix_state(lower,left )
#     msr=matrix_state(lower,right)

#     sites = siteinds("SpinHalf", N)
#     psi=randomMPS(sites)
#     #
#     # Make right canonical, then compress to left canonical
#     #
#     H=make_transIsing_MPO(sites,NNN,hx,ul,pbc=true)
#     E0l=inner(psi',to_openbc(H),psi)
#     @test is_lower_regular_form(H,eps)
#     canonical!(H,right)

#     E1l=inner(psi',to_openbc(H),psi)
#     @test abs(E0l-E1l)<1e-14
#     @test is_lower_regular_form(H,eps)
#     @test is_canonical(H,msr,eps)

#     compress!(H,left,epsSVD)
#     @test is_lower_regular_form(H,eps)
#     @test is_canonical(H,msl,eps)
#     # make sure the energy in unchanged
#     E2l=inner(psi',to_openbc(H),psi)
#     @test abs(E0l-E2l)<1e-14


    #
    # Make left canonical, then compress to right canonical
    #
    
   #=  H=make_transIsing_MPO(sites,NNN,hx,ul,pbc=true)
    E0r=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E0r)<1e-14
    @test is_lower_regular_form(H,eps)
    canonical!(H,left)

    E1r=inner(psi',to_openbc(H),psi)
    @test abs(E0r-E1r)<1e-14
    @test is_lower_regular_form(H,eps)
    @test is_canonical(H,msl,eps)

    compress!(H,right,epsSVD)
    @test is_lower_regular_form(H,eps)
    @test is_canonical(H,msr,eps)
    # make sure the energy in unchanged
    E2r=inner(psi',to_openbc(H),psi)
    @test abs(E0r-E2r)<1e-14 =#

#end

@testset "Compress full MPO" begin
    hx=0.5
    eps=1e-14
    epsSVD=1e-15

#                  V=N sites
#                    V=Num Nearest Neighbours in H
    # test_one_sweep(5,1,hx,lower,epsSVD,eps)
    # test_one_sweep(5,2,hx,lower,epsSVD,eps)
    # test_one_sweep(5,3,hx,lower,epsSVD,eps)
    # test_one_sweep(5,4,hx,lower,epsSVD,eps)
    # epsSVD=1e-5
    # test_one_sweep(50,1,hx,lower,epsSVD,eps)
    # test_one_sweep(50,2,hx,lower,epsSVD,eps)
    # test_one_sweep(50,3,hx,lower,epsSVD,eps)
    # test_one_sweep(50,4,hx,lower,epsSVD,eps)
end
