using ITensors
using ITensorMPOCompression
using Revise

# NNN = Number of Nearest Neighbours, for example
#    NNN=2 corresponds to nearest neighbour
#    NNN=3 corresponds to nearest and next nearest neighbour

function make_transIsing_op(indices::Vector{<:Index},NNN::Int64,hx::Float64=0.0)::ITensor
    @assert NNN>=2
    @assert length(indices)==4
    #  It turns out that
    #  Dw = 2+Sum(i,i=1..NNN) =2 + NNN*(NNN+1)/2
    #
    Dw::Int64=2+NNN*(NNN+1)/2;
    iblock=1;
    is=filterinds(indices,tags="Site")[1] #get any site index for generating operators
    ils=filterinds(indices,tags="Link") #get both link indices
    il1=Index(Dw,tags(ils[1])) #make new link indices with correct Dw
    il2=Index(Dw,tags(ils[2]))
    W=ITensor(il1,il2,is,is')
    unit=delta(is,is') #make a unit matrix
    Sz=op(is,"Sz")
    Sx=op(is,"Sx")
    assign!(W,il1=>1 ,il2=>1 ,unit)
    assign!(W,il1=>Dw,il2=>Dw,unit)
    assign!(W,il1=>Dw,il2=>1 ,hx*Sx); #add field term
    #very hard to explain this without a diagram.
    for iNN in 1:NNN
        assign!(W,il1=>iblock+1,il2=>1,Sz)
        for jNN in 1:iNN-1
            assign!(W,il1=>iblock+1+jNN,il2=>iblock+jNN,unit)
        end
        Jn=1.0/(iNN*iNN) #interactions need to decay with distance in order for H to extensive 
        assign!(W,il1=>Dw,il2=>iblock+iNN,Jn*Sz)
        iblock+=iNN
    end
    return W
end



function add_edge_links!(mpo::MPO)
    N=length(mpo)
    i1s=(inds(mpo[1])...,Index(1,"Link,l=0"))
    mpo[1]=ITensor(i1s)
    ins=(inds(mpo[N])...,Index(1,"Link,l=$(N+1)"))
    mpo[N]=ITensor(ins)
end

function V_lower_left(W::ITensor)::ITensor
    ils=filterinds(inds(W),tags="Link")
    iss=filterinds(inds(W),tags="Site")
    ilsV=(Index(dim(ils[1])-1,tags(ils[1])),Index(dim(ils[2])-1,tags(ils[2])))
    V=ITensor(ilsV...,iss...)
    for ilv in eachindval(ils)
        if ilv[1].second>1 && ilv[2].second>1
            ilvV=IndexVal(ilsV[1],ilv[1].second-1),IndexVal(ilsV[2],ilv[2].second-1)
            for isv in eachindval(iss)
                V[ilvV[1],ilvV[2],isv...]=W[ilv...,isv...]
            end
        end
    end
    return V
end



function runtest()
    N=5
    hx=0.5
    eps=1e-15
    sites = siteinds("SpinHalf", N)
    mpo=MPO(sites) #make and MPO only to get the indices
    add_edge_links!(mpo)
    for n in 1:N
        iset=IndexSet(inds(mpo[n])...)
        mpo[n]=make_transIsing_op(iset,3,hx)
        #print(mpo[n],eps)
    end

    V=V_lower_left(mpo[1])
    #print(V,eps)
    il1=filterinds(inds(V),tags="l=1")
    iothers=noncommoninds(inds(V),il1)
    #Q,R=ITensors.qr(V,iothers;positive=true)
    Q,L=ql(V,iothers;positive=true)
    pprint(Q,eps)
    @show norm(V-Q*L) 
end

runtest()