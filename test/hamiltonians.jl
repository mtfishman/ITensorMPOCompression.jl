using ITensors
import ITensorMPOCompression
using Revise

#handle case with 1 link index at edges.
function fix_autoMPO1(W::ITensor)::ITensor
    ils=filterinds(W,"Link")
    iss=filterinds(W,"Site")
    @assert length(ils)==1
    @assert length(iss)==2
    il=ils[1] #link index
    Dw=dim(il)
    p=collect(1:Dw)
    p[2],p[Dw]=p[Dw],p[2]
    W1=ITensor(il,iss...)
    for js in eachindval(iss)
        for jl in eachindval(il)
            Sl=slice(W,jl)
            assign(W1,Sl,il=>p[jl.second])
            #W1[il=>p[jl.second],js...]=W[jl,js...]
        end
    end
    return W1
end

function fix_autoMPO(W::ITensor)::ITensor
    ils=filterinds(W,"Link")
    if length(ils)==1
        return fix_autoMPO1(W)
    end
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
#    for js in eachindval(iss)
        for jr in eachindval(r)
            for jc in eachindval(c)
                sl=slice(W,jr,jc)
                assign(W1,sl,r=>pr[jr.second],c=>pc[jc.second])
#                W1[r=>pr[jr.second],c=>pc[jc.second],js...]=W[jr,jc,js...]
            end
        end
 #   end
    return W1
end

function assign(T1::ITensor,T2::ITensor,iv1::IndexVal...)
    i1=map(first,iv1)
    iss=noncommoninds(T1,i1)
    @assert length(noncommoninds(iss,inds(T2)))==0
    @assert iss[1]==inds(T2)[1]
    @assert iss[2]==inds(T2)[2]
    assign(T1,tensor(T2),iss,iv1...)
    # for js in eachindval(iss...)
    #     T1[iv1...,js...]=T2[js...]
    # end
end
function assign(T1::ITensor,T2::DenseTensor{ElT,N},iss::Vector{Index{Int64}},iv1::IndexVal...) where {ElT,N}
    for js in eachindval(iss...)
        js2=[x.second for x in js]
        T1[iv1...,js...]=T2[js2...]
    end
end

function assign(T1::ITensor,T2::BlockSparseTensor{ElT,N},iss::Any,iv1::IndexVal...) where {ElT,N}
    #@show typeof(iss)
    for b in eachnzblock(T2)
        isv=[iss[i]=>b[i] for i in 1:length(b)]
        s=T2[b][1]
        #@show 
        T1[iv1...,isv...]=s
    end
end

function fix_autoMPO!(H::MPO)
    N=length(H)
    for n in 1:N
        H[n]=fix_autoMPO(H[n])
    end
end

make_Heisenberg_AutoMPO(sites,NNN::Int64,J::Float64,hx::Float64,ul::tri_type)::MPO = 
    make_Heisenberg_AutoMPO(sites,NNN,J,hx)

function make_Heisenberg_AutoMPO(sites,NNN::Int64,J::Float64,hx::Float64)::MPO
    N=length(sites)
    @assert(N>NNN)
    ampo = OpSum()
    for j=1:N
        add!(ampo, hx   ,"Sz", j)
    end
    for dj=1:NNN
        f=J/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
            add!(ampo, f*0.5,"S+", j, "S-", j+dj)
            add!(ampo, f*0.5,"S-", j, "S+", j+dj)
        end
    end
    mpo=MPO(ampo,sites)
    fix_autoMPO!(mpo) #swap row[2]<->row[Dw] and col[2]<->col[Dw]
    return mpo
end



make_transIsing_AutoMPO(sites,NNN::Int64,J::Float64,hx::Float64,ul::tri_type)::MPO = 
    make_transIsing_AutoMPO(sites,NNN,J,hx)

function make_transIsing_AutoMPO(sites,NNN::Int64,J::Float64,hx::Float64)::MPO
    do_field = hx!=0.0
    N=length(sites)
    @assert(N>NNN)
    ampo = OpSum()
    if do_field
        for j=1:N
            add!(ampo, hx   ,"Sx", j)
        end
    end
    for dj=1:NNN
        f=J/dj
        for j=1:N-dj
            add!(ampo, f    ,"Sz", j, "Sz", j+dj)
        end
    end
    mpo=MPO(ampo,sites)
    fix_autoMPO!(mpo) #swap row[2]<->row[Dw] and col[2]<->col[Dw]
    return mpo
end


function make_transIsing_MPO(sites,NNN::Int64,J::Float64,hx::Float64,ul::tri_type=lower)::MPO
    use_qn=hasqns(sites[1])
    N=length(sites)
    mpo=MPO(sites) #make and MPO only to get the indices
    add_edge_links!(mpo) #add in l=0 and l=N edge links ... just so we can remove them later
    Dw::Int64=transIsing_Dw(NNN)
    if (use_qn)
        prev_link=Index(QN("Sz",0)=>Dw;dir=ITensors.In,tags="Link,l=0")
    else
        prev_link=Index(Dw,"Link,l=0") 
    end

    for n in 1:N
        iset=IndexSet(inds(mpo[n])...)
        mpo[n]=make_transIsing_op(iset,prev_link,n,NNN,J,hx,ul)
        prev_link=filterinds(inds(mpo[n],"l=$n"))[1]
    end
    return ITensorMPOCompression.to_openbc(mpo) #contract with l* and *r at the edges.
end

function add_edge_links!(mpo::MPO)
    N=length(mpo)
    i1s=(inds(mpo[1])...,Index(1,"Link,l=0"))
    mpo[1]=ITensor(i1s)
    ins=(inds(mpo[N])...,Index(1,"Link,l=$N"))
    mpo[N]=ITensor(ins)
end

#  It turns out that the trans-Ising model
#  Dw = 2+Sum(i,i=1..NNN) =2 + NNN*(NNN+1)/2
#
function transIsing_Dw(NNN::Int64)::Int64
    return 2+NNN*(NNN+1)/2
end

# NNN = Number of Nearest Neighbours, for example
#    NNN=1 corresponds to nearest neighbour
#    NNN=2 corresponds to nearest and next nearest neighbour
function make_transIsing_op(indices::Vector{<:Index},prev_link::Index,nsite::Int64,NNN::Int64,J::Float64,hx::Float64=0.0,ul::tri_type=lower)::ITensor
    @assert NNN>=1
    @assert length(indices)==4
    do_field = hx!=0.0
   
    Dw::Int64=transIsing_Dw(NNN)
    iblock=1;
    is=filterinds(indices,tags="Site")[1] #get any site index for generating operators
    use_qn=hasqns(is)
    #@show indices
    
    il1=prev_link
    indl2=filterinds(indices,tags="l=$nsite"    )
    @assert length(indl2)==1
    if (use_qn)
        il2=Index(QN("Sz",0)=>Dw;dir=ITensors.In,tags=tags(indl2[1]))
    else
        il2=Index(Dw,tags(indl2[1]))
    end
    W=ITensor(il1,dag(il2),is,dag(is'))
    unit=delta(is,dag(is')) #make a unit matrix
    Sz=op(is,"Sz")
    if do_field
        Sx=op(is,"Sx")
    end
    assign!(W,il1=>1 ,il2=>1 ,unit)
    assign!(W,il1=>Dw,il2=>Dw,unit)
    #loop below is coded for lower, just swap indexes to get upper
    
    if ul==lower
        if do_field
            assign!(W,il1=>Dw,il2=>1 ,hx*Sx); #add field term
        end
        #very hard to explain this without a diagram.
        for iNN in 1:NNN
            assign!(W,il1=>iblock+1,il2=>1,Sz)
            for jNN in 1:iNN-1
                assign!(W,il1=>iblock+1+jNN,il2=>iblock+jNN,unit)
            end
            Jn=J/(iNN) #interactions need to decay with distance in order for H to extensive 
            assign!(W,il1=>Dw,il2=>iblock+iNN,Jn*Sz)
            iblock+=iNN
        end
    else
        if do_field
            assign!(W,il1=>1,il2=>Dw ,hx*Sx); #add field term
        end
        #very hard to explain this without a diagram.
        for iNN in 1:NNN
            assign!(W,il1=>1,il2=>iblock+1,Sz)
            for jNN in 1:iNN-1
                assign!(W,il1=>iblock+jNN,il2=>iblock+1+jNN,unit)
            end
            Jn=J/(iNN) #interactions need to decay with distance in order for H to extensive 
            assign!(W,il1=>iblock+iNN,il2=>Dw,Jn*Sz)
            iblock+=iNN
        end
    end
    return W
end

function fast_GS(H::MPO,sites)::Tuple{Float64,MPS}
    psi0  = randomMPS(sites,length(H))
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 2,4,8,16,32)
    setcutoff!(sweeps, 1E-10)
    E,psi= dmrg(H,psi0, sweeps;outputlevel=0)
    return E,psi
end
