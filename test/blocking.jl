
@testset "Blocking functions" begin

    is=Index(2,"Site,n=1")
    V=ITensor(1.0,Index(3,"Link,l=0"),Index(3,"Link,l=1"),is,is')

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,1,1)
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 0.0 0.0 0.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0]
    
    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,0,1)
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 1.0 1.0 1.0; 
     0.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0;
     0.0 0.0 0.0 0.0]

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,1,0)
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [0.0 0.0 0.0 0.0; 
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0]

    W=ITensor(0.0,Index(4,"Link,l=0"),Index(4,"Link,l=1"),is,is')
    setV!(W,V,0,0)
    @test matrix(slice(W,is=>1,is'=>1)) == 
    [1.0 1.0 1.0 0.0; 
     1.0 1.0 1.0 0.0;
     1.0 1.0 1.0 0.0;
     0.0 0.0 0.0 0.0]
 
    lWlink=Index(4,"Link,l=1")
    L=ITensor(2.0,Index(3,"Link,ql"),Index(3,"Link,l=1"))
    Lplus=growRL(L,lWlink,1,1)
    @test matrix(Lplus) == 
    [1.0 0.0 0.0 0.0; 
     0.0 2.0 2.0 2.0;
     0.0 2.0 2.0 2.0;
     0.0 2.0 2.0 2.0]

    #
    Lplus=growRL(L,lWlink,1,0)
    @test matrix(Lplus) == 
    [1.0 0.0 0.0 0.0; 
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 1.0]
    #
    Lplus=growRL(L,lWlink,0,1)
    @test matrix(Lplus) == 
    [1.0 2.0 2.0 2.0; 
     0.0 2.0 2.0 2.0;
     0.0 2.0 2.0 2.0;
     0.0 0.0 0.0 1.0]
    #
    Lplus=growRL(L,lWlink,0,0)
    @test matrix(Lplus) == 
    [2.0 2.0 2.0 0.0; 
     2.0 2.0 2.0 0.0;
     2.0 2.0 2.0 0.0;
     0.0 0.0 0.0 1.0]
    
    end