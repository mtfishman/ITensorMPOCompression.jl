using ITensors
import ITensorMPOCompression
using Revise

function fix_autoMPO(W::ITensor)::ITensor
    ils=filterinds(W,"Link")
    iss=filterinds(W,"Site")
    @assert length(ils)==2
    @assert length(iss)==2
    d,n,r,c=parse_links(W)
    Dw1,Dw2=dim(r),dim(c)
    #
    #  set up perm arrays to swap row and col 2 with N
    #
    pr=collect(1:Dw1)
    pc=collect(1:Dw2)
    pr[2],pr[Dw1]=pr[Dw1],pr[2]
    pc[2],pc[Dw2]=pc[Dw2],pc[2]
    W1=ITensor(r,c,iss...)
    for js in eachindval(iss)
        for jr in eachindval(r)
            for jc in eachindval(c)
                W1[r=>pr[jr.second],c=>pc[jc.second],js...]=W[jr,jc,js...]
            end
        end
    end
    return W1
end

function fix_autoMPO!(H::MPO)
    N=length(H)
    for n in 1:N
        H[n]=fix_autoMPO(H[n])
    end
end

function make_Heisenberg_AutoMPO(sites,NNN::Int64,hx::Float64;kwargs...)::MPO
    N=length(sites)
    ampo = OpSum()
    for j=1:N
        add!(ampo, hx   ,"Sz", j)
    end
    for dj=1:NNN
        f=1.0/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
            add!(ampo, f*0.5,"S+", j, "S-", j+dj)
            add!(ampo, f*0.5,"S-", j, "S+", j+dj)
        end
    end
    mpo=MPO(ampo,sites;kwargs...)
    if !get(kwargs,:obc,true)
        fix_autoMPO!(mpo)
    end
    return mpo
end

function make_transIsing_AutoMPO(sites,NNN::Int64,hx::Float64;kwargs...)::MPO
    N=length(sites)
    ampo = OpSum()
    for j=1:N
        add!(ampo, hx   ,"Sx", j)
    end
    for dj=1:NNN
        f=1.0/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
        end
    end
    mpo=MPO(ampo,sites;kwargs...)
    if !get(kwargs,:obc,true)
        fix_autoMPO!(mpo)
    end
    return mpo
end

function make_transIsing_MPO(sites,NNN::Int64,hx::Float64,ul::tri_type=lower;kwargs...)::MPO
    N=length(sites)
    mpo=MPO(sites) #make and MPO only to get the indices
    add_edge_links!(mpo)

    prev_link=Index(1) #Don't know Dw here so we can't make an l=0 link index
    for n in 1:N
        iset=IndexSet(inds(mpo[n])...)
        mpo[n]=make_transIsing_op(iset,prev_link,n,NNN,hx,ul)
        prev_link=filterinds(inds(mpo[n],"l=$n"))[1]
    end
    if get(kwargs,:obc,true)
        mpo=ITensorMPOCompression.to_openbc(mpo) #contract with l* and *r at the edges.
    end
    return mpo
end

function add_edge_links!(mpo::MPO)
    N=length(mpo)
    i1s=(inds(mpo[1])...,Index(1,"Link,l=0"))
    mpo[1]=ITensor(i1s)
    ins=(inds(mpo[N])...,Index(1,"Link,l=$N"))
    mpo[N]=ITensor(ins)
end


    # NNN = Number of Nearest Neighbours, for example
#    NNN=2 corresponds to nearest neighbour
#    NNN=3 corresponds to nearest and next nearest neighbour
function make_transIsing_op(indices::Vector{<:Index},prev_link::Index,nsite::Int64,NNN::Int64,hx::Float64=0.0,ul::tri_type=lower)::ITensor
    @assert NNN>=1
    @assert length(indices)==4
    #  It turns out that
    #  Dw = 2+Sum(i,i=1..NNN) =2 + NNN*(NNN+1)/2
    #
    Dw::Int64=2+NNN*(NNN+1)/2;
    iblock=1;
    is=filterinds(indices,tags="Site")[1] #get any site index for generating operators
    #@show indices
    if nsite==1
        il1=Index(Dw,"Link,l=0")
    else
        @assert dim(prev_link)>1
        il1=prev_link
    end
    indl2=filterinds(indices,tags="l=$nsite"    )
    @assert length(indl2)==1
    il2=Index(Dw,tags(indl2[1]))
    W=ITensor(il1,il2,is,is')
    unit=delta(is,is') #make a unit matrix
    Sz=op(is,"Sz")
    Sx=op(is,"Sx")
    assign!(W,il1=>1 ,il2=>1 ,unit)
    assign!(W,il1=>Dw,il2=>Dw,unit)
    #loop below is coded for lower, just swap indexes to get upper
    
    if ul==lower
        assign!(W,il1=>Dw,il2=>1 ,hx*Sx); #add field term
        #very hard to explain this without a diagram.
        for iNN in 1:NNN
            assign!(W,il1=>iblock+1,il2=>1,Sz)
            for jNN in 1:iNN-1
                assign!(W,il1=>iblock+1+jNN,il2=>iblock+jNN,unit)
            end
            Jn=1.0/(iNN) #interactions need to decay with distance in order for H to extensive 
            assign!(W,il1=>Dw,il2=>iblock+iNN,Jn*Sz)
            iblock+=iNN
        end
    else
        assign!(W,il1=>1,il2=>Dw ,hx*Sx); #add field term
        #very hard to explain this without a diagram.
        for iNN in 1:NNN
            assign!(W,il1=>1,il2=>iblock+1,Sz)
            for jNN in 1:iNN-1
                assign!(W,il1=>iblock+jNN,il2=>iblock+1+jNN,unit)
            end
            Jn=1.0/(iNN) #interactions need to decay with distance in order for H to extensive 
            assign!(W,il1=>iblock+iNN,il2=>Dw,Jn*Sz)
            iblock+=iNN
        end
    end
    return W
end

