using ITensors
using ITensorMPOCompression
using Test
using Revise,Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

@testset "Blocking functions for dense matrices" begin

    is=Index(2,"Site,SpinHalf,n=1")
    iqx=Index(4,"Link,qx")
    V=ITensor(1.0,Index(3,"Link,l=0"),Index(3,"Link,qx"),is,is')

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,iqx,matrix_state(lower,left))
    @test reshape(array(W[is=>1:1,is'=>1:1]),4,4) == 
    [0.0 0.0 0.0 0.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0]
   
    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,iqx,matrix_state(lower,right))
    @test reshape(array(W[is=>1:1,is'=>1:1]),4,4)  == 
    [1.0 1.0 1.0 0.0; 
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0;
     0.0 0.0 0.0 0.0]
 
    lWlink=Index(4,"Link,l=1")
    L=ITensor(2.0,Index(3,"Link,ql"),Index(3,"Link,l=1"))
    Lplus,il=growRL(L,lWlink,V_offsets(1,1))
    @test matrix(Lplus) == 
    [1.0 0.0 0.0 0.0; 
     0.0 2.0 2.0 2.0;
     0.0 2.0 2.0 2.0;
     0.0 2.0 2.0 2.0]

    #
    # these test are potentially flaky because they depend on the 
    # index ordering of W.  In other words matrix(W) is ill defined, the transpose
    # of what you expect could be returned.
    #
    A=[i*1.0 for i in 1:4*4*2*2]
    V=ITensor(0.5,Index(3,"Link,l=0"),Index(3,"Link,qx"),is,is')
    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,iqx,matrix_state(lower,left))
    @test reshape(array(W[is=>1:1,is'=>1:1]),4,4) == 
    # [1.0 5.0 9.0 13.0 ; 
    #  2.0 0.5 0.5  0.5 ;
    #  3.0 0.5 0.5  0.5 ;
    #  4.0 0.5 0.5  0.5 ]
     [1.0 2.0 3.0 4.0 ; 
      0.0 0.5 0.5  0.5 ;
      0.0 0.5 0.5  0.5 ;
     0.0 0.5 0.5  0.5 ]
    
    
    
    # Now force resizing.  It shoud preserve the last row and col of W
    iqx=Index(3,"Link,qx")
    V=ITensor(0.5,Index(3,"Link,l=0"),Index(2,"Link,qx"),is,is')
    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,iqx,matrix_state(lower,left))
    @test transpose(reshape(array(W[is=>1:1,is'=>1:1]),3,4) ) == 
    [1.0 0.0  0.0 ; #top row gets zeroed out on resizing 
     2.0 0.5  0.5 ;
     3.0 0.5  0.5 ;
     4.0 0.5  0.5 ]
     
    V=ITensor(0.5,Index(3,"Link,l=0"),Index(2,"Link,qx"),is,is')
    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    
    W=setV(W,V,iqx,matrix_state(lower,right))
   
    @test reshape(array(W[is=>1:1,is'=>1:1]),3,4)  == 
    [0.5   0.5  0.5  0.0 ; 
     0.5   0.5  0.5  0.0 ;
     13.0 14.0 15.0 16.0 ]


    V=ITensor(0.5,Index(2,"Link,qx"),Index(3,"Link,l=1"),is,is')

    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,iqx,matrix_state(lower,left))
    @test reshape(array(W[is=>1:1,is'=>1:1]),3,4)  == 
    [1.0 5.0 9.0 13.0; 
     0.0 0.5 0.5  0.5;
     0.0 0.5 0.5  0.5]
    
    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    
    W=setV(W,V,iqx,matrix_state(lower,right))
    @test reshape(array(W[is=>1:1,is'=>1:1]),3,4)  == 
    [0.5 0.5  0.5 0.0; 
     0.5 0.5  0.5 0.0;
     4.0 8.0 12.0 16.0]

 
    #
    #  GrowRL tests
    #
    Lplus,il=growRL(L,lWlink,V_offsets(0,1))
    @test matrix(Lplus) == 
    [1.0 0.0 0.0 0.0; 
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 1.0]
    #
    Lplus,il=growRL(L,lWlink,V_offsets(1,0))
    @test matrix(Lplus) == 
    [1.0 2.0 2.0 2.0; 
     0.0 2.0 2.0 2.0;
     0.0 2.0 2.0 2.0;
     0.0 0.0 0.0 1.0]
    #
    Lplus,il=growRL(L,lWlink,V_offsets(0,0))
    @test matrix(Lplus) == 
    [2.0 2.0 2.0 0.0; 
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 0.0;
     0.0 0.0 0.0 1.0]
    
end


@testset "Sub tensor assign for block sparse matrices with compatable QN spaces" begin 
    qns=[QN("Sz",0)=>1,QN("Sz",0)=>3,QN("Sz",0)=>2]
    i,j=Index(qns,"i"),Index(qns,"j")
    A=randomITensor(i,j)
    nr,nc=dims(A)
    B=copy(A)
    for dr in 0:nr-1
        for dc in 0:nc-1
            B[i=>1:nr-dr,j=>1:nc-dc]=A[i=>1:nr-dr,j=>1:nc-dc]
            @test norm(B-A)==0
            B[i=>1+dr:nr,j=>1:nc-dc]=A[i=>1+dr:nr,j=>1:nc-dc]
            @test norm(B-A)==0
        end
    end
end

#
#  This is a tough nut to crack.  But it is not needed for MPO compression.
#
# @testset "Sub tensor assign for block sparse matrices with in-compatable QN spaces" begin 
#     qns=[QN("Sz",0)=>1,QN("Sz",0)=>3,QN("Sz",0)=>2]
#     qnsC=[QN("Sz",0)=>2,QN("Sz",0)=>2,QN("Sz",0)=>2] #purposely miss allgined.
#     i,j=Index(qns,"i"),Index(qns,"j")
#     A=randomITensor(i,j)
#     nr,nc=dims(A)
#     ic,jc=Index(qnsC,"i"),Index(qnsC,"j")
#     C=randomITensor(ic,jc)
#     @show dense(A) dense(C)
#     C[ic=>1:nr,jc=>1:nc]=A[i=>1:nr,j=>1:nc]
#     @show matrix(A)-matrix(C)
#     @test norm(matrix(A)-matrix(C))==0
# end


nothing