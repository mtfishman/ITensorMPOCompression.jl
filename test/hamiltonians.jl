using ITensors
import ITensorMPOCompression
using Revise

function make_Heisenberg_AutoMPO(sites::Vector{Index{Int64}})::MPO
    N=length(sites)
    ampo = OpSum()
    for dj=1:3
        f=1.0/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
            add!(ampo, f*0.5,"S+", j, "S-", j+dj)
            add!(ampo, f*0.5,"S-", j, "S+", j+dj)
        end
    end
    return MPO(ampo,sites)  
end

function make_transIsing_AutoMPO(sites::Vector{Index{Int64}},NNN::Int64,hx::Float64)::MPO
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
    return MPO(ampo,sites)  
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
    if !get(kwargs,:pbc,false)
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

