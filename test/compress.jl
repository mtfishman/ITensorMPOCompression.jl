using ITensorMPOCompression
using Revise
using Test

include("hamiltonians.jl")

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
println("-----------Start--------------")

function make_RL(r::Index,c::Index,ms::matrix_state)::ITensor
    @assert ms.ul==upper || ms.ul==lower
    @assert ms.lr==left  || ms.lr==right
    A=randomITensor(r,c)
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

function test_getM(r,c)
    if dim(r)>=dim(c)
        ms=matrix_state(lower,left)
        L=make_RL(r,c,ms)
        #@show L
        M,L_prime,im=getM(L,ms.lr)
        #@show M
        #@show L_prime
        Mplus=grow(r,M,im)
        #@show Mplus
        Ltest=L-Mplus*L_prime
        #@show Ltest
        @test norm(Ltest)==0.0

        ms=matrix_state(upper,left)
        R=make_RL(r,c,ms)
        #@show R
        M,R_prime,im=getM(R,ms.lr)
        #@show M
        #@show R_prime
        Mplus=grow(r,M,im)
        #@show Mplus
        Rtest=R-Mplus*R_prime
        #@show Rtest
        @test norm(Rtest)==0.0
    end
    if dim(c)>=dim(r)
        ms=matrix_state(lower,right)
        L=make_RL(r,c,ms)
        #@show L
        M,L_prime,im=getM(L,ms.lr)
        #@show M
        #@show L_prime
        Mplus=grow(im,M,c)
        #@show Mplus
        Ltest=L-L_prime*Mplus
        #@show Ltest
        @test norm(Ltest)==0.0


        ms=matrix_state(upper,right)
        R=make_RL(r,c,ms)
        #@show R
        M,R_prime,im=getM(R,ms.lr)
        #@show M
        #@show R_prime
        Mplus=grow(im,M,c)
        #@show Mplus
        Rtest=R-R_prime*Mplus
        #@show Rtest
        @test norm(Rtest)==0.0
    end
    
end

@testset "growM" begin
    test_getM(Index(5,"Link,l=0"),Index(5,"Link,l=1"))
    test_getM(Index(7,"Link,l=0"),Index(5,"Link,l=1"))
    test_getM(Index(5,"Link,l=0"),Index(7,"Link,l=1"))
    
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
    # left canonical for lower triangular MPO
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0=inner(psi',to_openbc(H),psi)
    @test detect_upper_lower(H,eps)==lower
    canonical!(H,right)

    E1=inner(psi',to_openbc(H),psi)
    @test abs(E0-E1)<1e-14
    @test detect_upper_lower(H,eps)==lower
    @test  is_canonical(H,msr,eps)

    W,L=compress(H[1],left,epsSVD)
    @test detect_upper_lower(H,eps)==lower
    @test detect_upper_lower(W,eps)==lower
    @test norm(H[1]-W*L)<eps
    H[1]=W
    H[2]=L*H[2]
    @test detect_upper_lower(H[2],eps)==lower
    # make sure the energy in unchanged
    E2=inner(psi',to_openbc(H),psi)
    @test abs(E0-E2)<1e-14

    #@test  is_canonical(H[1],msl,eps)

end

@testset "Compress full MPO" begin
    N=15
    NNN=4
    hx=0.5
    eps=1e-15
    epsSVD=1e-15
    msl=matrix_state(lower,left )
    msr=matrix_state(lower,right)

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # left canonical for lower triangular MPO
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0=inner(psi',to_openbc(H),psi)
    @test is_lower_regular_form(H,eps)
    canonical!(H,right)
    pprint(H,eps)

    E1=inner(psi',to_openbc(H),psi)
    @test abs(E0-E1)<1e-14
    @test is_lower_regular_form(H,eps)
    @test  is_canonical(H,msr,eps)

    compress!(H,left,epsSVD)
    @test is_lower_regular_form(H,eps)
    # make sure the energy in unchanged
    E2=inner(psi',to_openbc(H),psi)
    @test abs(E0-E2)<1e-14

    pprint(H,eps)
    pprint(H[3],eps)
#    @test  is_canonical(H,msr,eps)

end
