using ITensors
using ITensorMPOCompression
using Test
using Revise

@testset "Blocking functions" begin

    is=Index(2,"Site,n=1")
    V=ITensor(1.0,Index(3,"Link,l=0"),Index(3,"Link,l=1"),is,is')

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,V_offsets(1,1))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 0.0 0.0 0.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0]
    
    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,V_offsets(0,1))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 1.0 1.0 1.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 0.0 0.0 0.0]

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,V_offsets(1,0))
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 0.0 0.0 0.0; 
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0]

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,V_offsets(0,0))
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