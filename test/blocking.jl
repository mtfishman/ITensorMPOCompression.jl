using ITensors
using ITensorMPOCompression
using Test
using Revise

@testset "Blocking functions" begin

    is=Index(2,"Site,n=1")
    V=ITensor(1.0,Index(3,"Link,l=0"),Index(3,"Link,l=1"),is,is')

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,V_offsets(1,1))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 0.0 0.0 0.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0]
    
    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,V_offsets(0,1))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 1.0 1.0 1.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 0.0 0.0 0.0]

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,V_offsets(1,0))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 0.0 0.0 0.0; 
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0]

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,V_offsets(0,0))
    @test matrix(slice(W,is=>1,is'=>1)) == 
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
    # Now force resizing.  It shoud preserve the last row and col of W
    #
    A=[i*1.0 for i in 1:4*4*2*2]
    V=ITensor(0.5,Index(3,"Link,l=0"),Index(2,"Link,qx"),is,is')

    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,V_offsets(1,1))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [1.0 5.0 13.0 ; 
     2.0 0.5  0.5 ;
     3.0 0.5  0.5 ;
     4.0 0.5  0.5 ]
     
    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    
    W=setV(W,V,V_offsets(0,0))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.5 0.5 13.0 ; 
     0.5 0.5 14.0 ;
     0.5 0.5 15.0 ;
     4.0 8.0 16.0 ]


    V=ITensor(0.5,Index(2,"Link,qx"),Index(3,"Link,l=1"),is,is')

    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    W=setV(W,V,V_offsets(1,1))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [1.0 5.0 9.0 13.0; 
     2.0 0.5 0.5  0.5;
     4.0 0.5 0.5  0.5]
    
    W=ITensor(eltype(A),A,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    
    W=setV(W,V,V_offsets(0,0))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.5 0.5  0.5 13.0; 
     0.5 0.5  0.5 14.0;
     4.0 8.0 12.0 16.0]

 
    #
    #  GrowRL tests
    #
    Lplus,il=growRL(L,lWlink,V_offsets(1,0))
    @test matrix(Lplus) == 
    [1.0 0.0 0.0 0.0; 
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 1.0]
    #
    Lplus,il=growRL(L,lWlink,V_offsets(0,1))
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