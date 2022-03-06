using ITensorMPOCompression
using Revise
using Test

import ITensorMPOCompression.truncate!
import ITensorMPOCompression.truncate
import ITensorMPOCompression.orthogonalize!

include("hamiltonians.jl")

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
println("-----------Start--------------")

function make_RL(r::Index,c::Index,ms::matrix_state,swap::Bool)::ITensor
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
    eps=1e-14
    if lr==left
        @assert dim(r)>=dim(c)
    #    println("---------- lower left -----------")
        ms=matrix_state(lower,left)
        L=make_RL(r,c,ms,swap)
        M,L_prime,im=getM(L,ms,eps)
        if hastags(M,tags(c))
            Mplus=grow(M,im,c)
        elseif hastags(M,tags(r))
            Mplus=grow(M,r,im)
        else
            @assert false
        end
        Ltest=L-Mplus*L_prime
        @test norm(Ltest)==0.0

    #    println("---------- upper left -----------")
        ms=matrix_state(upper,left)
        R=make_RL(r,c,ms,swap)
        M,R_prime,im=getM(R,ms,eps)
        if hastags(M,tags(c))
            Mplus=grow(M,im,c)
        elseif hastags(M,tags(r))
            Mplus=grow(M,r,im)
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
        M,L_prime,im=getM(L,ms,eps)
        if hastags(M,tags(c))
            Mplus=grow(M,im,c)
        elseif hastags(M,tags(r))
            Mplus=grow(M,r,im)
        else
            @assert false
        end
        Ltest=L-L_prime*Mplus
        @test norm(Ltest)==0.0


    #    println("---------- upper right -----------")
        ms=matrix_state(upper,right)
        R=make_RL(r,c,ms,swap)
        M,R_prime,im=getM(R,ms,eps)
        if hastags(M,tags(c))
            Mplus=grow(M,im,c)
        elseif hastags(M,tags(r))
            Mplus=grow(M,r,im)
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

#
# These test are set up not to compress (epsSVD is zero).  This is so that we can test
# the integrity of MPO forms and calculated energies to high precision.  Essentialy we
# just checking that the algorithm as coded is gauge invariant.

@testset "Compress one site" begin
    N=10
    NNN=6
    hx=0.5
    eps=1e-14
    epsSVD=.00
    ul=lower
    msl=matrix_state(ul,left )
    msr=matrix_state(ul,right)

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # Make right canonical, then compress to left canonical
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0l=inner(psi',to_openbc(H),psi)
    @test is_upper_lower(H,lower,eps)
    orthogonalize!(H,dir=right)

    E1l=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E1l)<eps
    @test is_regular_form(H,ul,eps)
    @test is_canonical(H,msr,eps)

    W,L=truncate(H[1],ul;dir=left,cutoff=epsSVD)
    @test is_regular_form(H,ul,eps)
    @test is_regular_form(W,ul,eps)
    @test norm(H[1]-W*L)<eps
    H[1]=W
    H[2]=L*H[2]
    @test is_regular_form(H[2],ul,eps)
    @test  is_canonical(H[1],msl,eps)
    # make sure the energy in unchanged
    E2l=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E2l)<eps
    #
    # Make left canonical, then compress to right canonical
    #
    H=make_transIsing_MPO(sites,NNN,hx,lower,pbc=true)
    E0r=inner(psi',to_openbc(H),psi)
    @test is_upper_lower(H,lower,eps)
    orthogonalize!(H;dir=left)

    E1r=inner(psi',to_openbc(H),psi)
    @test abs(E0r-E1r)<1e-14
    @test is_regular_form(H,lower,eps)
    @test  is_canonical(H,msl,eps)

    W,L=truncate(H[N],ul;dir=right,cutoff=epsSVD)
    @test is_regular_form(W,lower,eps)
    @test norm(H[N]-L*W)<eps
    H[N]=W
    H[N-1]=H[N-1]*L
    @test is_regular_form(H[N-1],ul,eps)
    @test  is_canonical(H[N],msr,eps)
    # make sure the energy in unchanged
    E2r=inner(psi',to_openbc(H),psi)
    @test abs(E0r-E2r)<eps
end

function test_one_sweep(N::Int64,NNN::Int64,hx::Float64,ul::tri_type,epsSVD::Float64,eps::Float64)
    msl=matrix_state(ul,left )
    msr=matrix_state(ul,right)

    sites = siteinds("SpinHalf", N)
    psi=randomMPS(sites)
    #
    # Make right canonical, then compress to left canonical
    #
    H=make_transIsing_MPO(sites,NNN,hx,ul,pbc=true)
    E0l=inner(psi',to_openbc(H),psi)
    @test is_regular_form(H,ul,eps)
    orthogonalize!(H,dir=right)
    #pprint(H,eps)

    E1l=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E1l)<eps
    @test is_regular_form(H,ul,eps)
    @test is_canonical(H,msr,eps)

    truncate!(H;dir=left,cutoff=epsSVD)
    truncate!(H;dir=right,cutoff=epsSVD)
    @test is_regular_form(H,ul,eps)
    @test is_canonical(H,msr,eps)
    # make sure the energy in unchanged
    E2l=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E2l)<10*epsSVD


    #
    # Make left canonical, then compress to right canonical
    #
    
    H=make_transIsing_MPO(sites,NNN,hx,ul,pbc=true)
    E0r=inner(psi',to_openbc(H),psi)
    @test abs(E0l-E0r)<1e-14
    @test is_regular_form(H,ul,eps)
    orthogonalize!(H,dir=left)

    E1r=inner(psi',to_openbc(H),psi)
    @test abs(E0r-E1r)<1e-14
    @test is_regular_form(H,ul,eps)
    @test is_canonical(H,msl,eps)

    truncate!(H;dir=right,cutoff=epsSVD)
    truncate!(H;dir=left,cutoff=epsSVD)
    @test is_regular_form(H,ul,eps)
    @test is_canonical(H,msl,eps)
    # make sure the energy in unchanged
    E2r=inner(psi',to_openbc(H),psi)
    relError=abs(E0r-E2r)/epsSVD
    @printf "Relative error in Energy %.1e \n" relError

end

@testset "Compress full MPO" begin
    hx=0.5
    eps=1e-14
    epsSVD=1e-14

#                  V=N sites
#                    V=Num Nearest Neighbours in H
    test_one_sweep(5,1,hx,lower,epsSVD,eps)
    test_one_sweep(5,2,hx,lower,epsSVD,eps)
    test_one_sweep(5,3,hx,lower,epsSVD,eps)
    test_one_sweep(5,4,hx,lower,epsSVD,eps)
    test_one_sweep(5,1,hx,upper,epsSVD,eps)
    test_one_sweep(5,2,hx,upper,epsSVD,eps)
    test_one_sweep(5,3,hx,upper,epsSVD,eps) #known unit on diagonal
    test_one_sweep(5,4,hx,upper,epsSVD,eps)
     epsSVD=.0000001
    test_one_sweep(10,1,hx,lower,epsSVD,eps)
    test_one_sweep(10,7,hx,lower,epsSVD,eps)
    test_one_sweep(10,8,hx,lower,epsSVD,eps) 
    test_one_sweep(10,9,hx,lower,epsSVD,eps)
    test_one_sweep(10,1,hx,upper,epsSVD,eps)
    test_one_sweep(10,7,hx,upper,epsSVD,eps)
    test_one_sweep(10,8,hx,upper,epsSVD,eps) 
    test_one_sweep(10,9,hx,upper,epsSVD,eps) 
end

#
#  Here we are pusposly truncating, so gauge invarience tests become more difficult.
#  Assuming delta(E) after compression is <epsSVD
#
@testset "Compress with higher values of epsSVD" begin
    N=10
    hx=0.5
    eps=1e-13
    epsSVD=1e-12
    test_one_sweep(10,6,hx,lower,epsSVD,eps)
    test_one_sweep(10,6,hx,upper,epsSVD,eps)
    
    # epsSVD=1e-10
    test_one_sweep(10,6,hx,lower,epsSVD,eps)  
    test_one_sweep(10,6,hx,upper,epsSVD,eps) 

end
