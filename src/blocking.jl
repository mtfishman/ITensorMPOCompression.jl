#using Printf

function getspace(i::QNIndex,off::Int64)
    @mpoc_assert off==0 || off==1
    nb=nblocks(i)
    qnb=off==0 ? space(i)[nb] : space(i)[1]
    #@show off space(i)
    #@mpoc_assert blockdim(qnb)==1 TODO: get this working.
    return qn(qnb) 
end
function getspace(i::Index,off::Int64)
    return 1 
end
#
#  create a Vblock IndexRange using the supplied offset.
#
range(i::Index,offset::Int64) = i=>1+offset:dim(i)-1+offset
#
# functions for getting and setting V blocks required for block respecting QX and SVD
#
function getV(W::ITensor,off::V_offsets)::Tuple{ITensor, Union{QN,Int}}
    if order(W)==3
        w1=filterinds(inds(W),tags="Link")[1]
        return W[range(w1,off.o1)],getspace(w1,off.o1)
    elseif order(W)==4
        w1,w2=filterinds(inds(W),tags="Link")
        if dim(w1)==1
            return W[w1=>1:1,range(w2,off.o2)],getspace(w1,off.o1)
        elseif dim(w2)==1
            return W[range(w1,off.o1),w2=>1:1],getspace(w1,off.o1)
        else
            return W[range(w1,off.o1),range(w2,off.o2)],getspace(w1,off.o1)
        end
    else 
        @show inds(W)
        @error("getV(W::ITensor,off::V_offsets) Case with order(W)=$(order(W)) not supported.")
    end
end
# Handles W with only one link index
function setV1(W::ITensor,V::ITensor,iqx::Index,ms::matrix_state)::ITensor
    iwl,=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    ivl,=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    @mpoc_assert inds(W,tags="Site")==inds(V,tags="Site")
    off=V_offsets(ms)
    @mpoc_assert(off.o1==off.o2)
    # 
    # W & V need to have the same Link tags as W1 for the assignments below to work.
    #
    W=replacetags(W,tags(iwl),tags(iqx))
    V=replacetags(V,tags(ivl),tags(iqx)) 
    iwl,=inds(W,tags=tags(iqx))
    #
    #  Make a new tensor to copy V and one row of W into
    #
    T=eltype(V)
    W1=ITensor(T(0.0),iqx,inds(W,tags="Site"))
    rc= off.o1==1 ? 1 : dim(iwl) #preserve row/col 1 or Dw
    rc1= off.o1==1 ? 1 : dim(iqx)
    W1[iqx=>rc1:rc1]=W[iwl=>rc:rc]
    W1[range(iqx,off.o1)]=V #Finally we can do the assingment of the V-block.
    return W1
end

#
#  This function is non trivial for 3 reasons:
#   1) V could have a truncted number of rows or columns as a result of rank revealing QX
#      W then has to be re-sized accordingly
#   2) If W gets resized we need to preserve particular rows/columns outside the Vblock from the old W.
#   3) V and W should have one common link index so we need to find those and pair them together. But we
#      need to match on tags&plev because dims are different so ID matching won't work. 
#
function setV(W::ITensor,V::ITensor,iqx::Index,ms::matrix_state)::ITensor
    if order(W)==3
        return setV1(W,V,iqx,ms) #Handle row/col vectors
    end
    #
    #  Deduce all link indices
    #
    @mpoc_assert order(W)==4
    wils=filterinds(inds(W),tags="Link") #should be l=n, l=n-1
    vils=filterinds(inds(V),tags="Link") #should be qx and {l=n,l=n-1} depending on sweep direction
    @mpoc_assert length(wils)==2
    @mpoc_assert length(vils)==2
    ivqx,=inds(V,tags="Link,qx") #get the qx index.  This is the one that can change size
    ivl=noncommonind(vils,ivqx)  #get the other link l=n index on V
    iwl=wils[match_tagplev(ivl,wils...)] #now find the W link index with the same tags/plev as ivl
    iwqx=noncommonind(wils,iwl) #get the other link l=n index on W
    off=V_offsets(ms)
    @mpoc_assert off.o1==off.o2
    # 
    # W & V need to have the same Link tags as W1 for the assignments below to work.
    #
    W=replacetags(W,tags(iwqx),tags(iqx))
    V=replacetags(V,tags(ivqx),tags(iqx)) #V needs to have the same Link tags as W for the assignment below to work.
    iwqx,=inds(W,tags=tags(iqx))
    #
    #  Make a new tensor to copy V and one row of W into
    #
    iss=filterinds(inds(W),tags="Site")
    T=eltype(V)
    W1=ITensor(T(0.0),iqx,iwl,iss)
    rc= off.o1==1 ? 1 : dim(iwqx) #preserve row/col 1 or Dw
    rc1= off.o1==1 ? 1 : dim(iqx)
    #@show iwqx rc iqx rc1 iwl
    #@pprint W
    #@show W[iwqx=>rc:rc,iwl=>1:dim(iwl)]
    W1[iqx=>rc1:rc1,iwl=>1:dim(iwl)]=W[iwqx=>rc:rc,iwl=>1:dim(iwl)] #copy row/col rc/rc1
    #@pprint W1
    if dim(iqx)==1
        W1[iqx=>1:1,range(iwl,off.o2)]=V 
    elseif dim(iwl)==1
        W1[range(iqx,off.o1),iwl=>1:1]=V
    else
        W1[range(iqx,off.o1),range(iwl,off.o2)]=V #Finally we can do the assingment of the V-block.
    end
    return W1
end

#-------------------------------------------------------------------------------
#
#  Blocking functions
#
#
#  Decisions: 1) Use ilf,ilb==forward,backward  or ir,ic=row,column ?
#             2) extract_blocks gets everything.  Should it defer to get_bc_block for b and c?
#   may best to define W as
#               ul=lower         ul=upper
#                1 0 0           1 b d
#     lr=left    b A 0           0 A c
#                d c I           0 0 I
#
#                1 0 0           1 c d
#     lr=right   c A 0           0 A b
#                d b I           0 0 I
#
#  Use kwargs in extract_blocks so caller can choose what they need. Default is c only
#
mutable struct regform_blocks
    𝕀::Union{ITensor,Nothing}
    𝑨::Union{ITensor,Nothing}
    𝑨𝒄::Union{ITensor,Nothing}
    𝒃::Union{ITensor,Nothing}
    𝒄::Union{ITensor,Nothing}
    𝒅::Union{ITensor,Nothing}
    irA::Union{Index,Nothing}
    icA::Union{Index,Nothing}
    irAc::Union{Index,Nothing}
    icAc::Union{Index,Nothing}
    irb::Union{Index,Nothing}
    icb::Union{Index,Nothing}
    irc::Union{Index,Nothing}
    icc::Union{Index,Nothing}
    ird::Union{Index,Nothing}
    icd::Union{Index,Nothing}    
    regform_blocks()=new(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing)
end

d(rfb::regform_blocks)::Float64=scalar(rfb.𝕀*dag(rfb.𝕀))
b0(rfb::regform_blocks)::ITensor=rfb.𝒃*dag(rfb.𝕀)/d(rfb)
c0(rfb::regform_blocks)::ITensor=rfb.𝒄*dag(rfb.𝕀)/d(rfb)
A0(rfb::regform_blocks)::ITensor=rfb.𝑨*dag(rfb.𝕀)/d(rfb)

#
#  Transpose inds for upper, no-op for lower
#
function swap_ul(ileft::Index,iright::Index,ul::reg_form)
    return ul==lower ? (ileft,iright,dim(ileft),dim(iright)) :  (iright,ileft,dim(iright),dim(ileft))
end
# lower left or upper right
llur(ul::reg_form,lr::orth_type)= lr==left && ul==lower || lr==right&&ul==upper
llur(W::reg_form_Op,lr::orth_type)=llur(W.ul,lr)
llur(ms::matrix_state)=llur(ms.ul,ms.lr)

#  Use recognizably distinct UTF symbols for operators, and op valued vectors and matrices: 𝕀 𝑨 𝒃 𝒄 𝒅 ⌃ c₀ 𝑨𝒄
# symbols from here: https://www.compart.com/en/unicode/block/U+1D400
#extract_blocks(W::reg_form_Op,lr::orth_type;kwargs...)=extract_blocks(W.W,W.ileft,W.iright,matrix_state(W.ul,lr);kwargs...)

function extract_blocks(Wrf::reg_form_Op,lr::orth_type;all=false,c=true,b=false,d=false,A=false,Ac=false,I=true,fix_inds=false,swap_bc=true)::regform_blocks
    check(Wrf)
    @assert plev(Wrf.ileft)==0
    @assert plev(Wrf.iright)==0
    # if dir(Wrf.W,ic)!=dir(ic)
    #     ic=dag(ic)
    # end
    # if dir(W,ir)!=dir(ir)
    #     ir=dag(ir)
    # end
    # @assert !hasqns(ir) || dir(W,ir)==dir(ir)
    # @assert !hasqns(ic) || dir(W,ic)==dir(ic)
    W=Wrf.W
    ir,ic=Wrf.ileft,Wrf.iright
    if Wrf.ul==upper
        ir,ic=ic,ir #transpose
    end
    nr,nc=dim(ir),dim(ic)
    @assert nr>1 || nc>1
    if all #does not include Ac
        A=b=c=d=I=true
    end
    if fix_inds && !d
        @warn "extract_blocks: fix_inds requires d=true."
        d=true
    end
    if !llur(Wrf,lr) && swap_bc #not lower-left or upper-right
        b,c=c,b #swap flags
    end

    A = A && (nr>1 && nc>1)
    b = b &&  nr>1 
    c = c &&  nc>1

  
    Wb=regform_blocks()
    I && (Wb.𝕀= nr>1 ? slice(W,ir=>1,ic=>1) : slice(W,ir=>1,ic=>nc))
   
    if A
        Wb.𝑨= W[ir=>2:nr-1,ic=>2:nc-1]
        Wb.irA,=inds(Wb.𝑨,tags=tags(ir))
        Wb.icA,=inds(Wb.𝑨,tags=tags(ic))
    end
    if Ac
        if llur(Wrf,lr)
            Wb.𝑨𝒄= nr>1 ? W[ir=>2:nr,ic=>2:nc-1] : W[ir=>1:1,ic=>2:nc-1]
        else
            Wb.𝑨𝒄= nc>1 ? W[ir=>2:nr-1,ic=>1:nc-1]  : W[ir=>2:nr-1,ic=>1:1]
        end
        Wb.irAc,=inds(Wb.𝑨𝒄,tags=tags(ir))
        Wb.icAc,=inds(Wb.𝑨𝒄,tags=tags(ic))
    end
    if b
        Wb.𝒃= W[ir=>2:nr-1,ic=>1:1]
        Wb.irb,=inds(Wb.𝒃,tags=tags(ir))
        Wb.icb,=inds(Wb.𝒃,tags=tags(ic))
    end
    if c
        Wb.𝒄= W[ir=>nr:nr,ic=>2:nc-1]
        Wb.irc,=inds(Wb.𝒄,tags=tags(ir))
        Wb.icc,=inds(Wb.𝒄,tags=tags(ic))
    end
    if d
        Wb.𝒅= nr >1 ? W[ir=>nr:nr,ic=>1:1] : W[ir=>1:1,ic=>1:1]
        Wb.ird,=inds(Wb.𝒅,tags=tags(ir))
        Wb.icd,=inds(Wb.𝒅,tags=tags(ic))
    end

    if fix_inds
        if !isnothing(Wb.𝒄)
            Wb.𝒄=replaceind(Wb.𝒄,Wb.irc,Wb.ird)
            Wb.irc=Wb.ird
        end
        if !isnothing(Wb.𝒃)
            Wb.𝒃=replaceind(Wb.𝒃,Wb.icb,Wb.icd)
            Wb.icb=Wb.icd
        end
        if !isnothing(Wb.𝑨)
            Wb.𝑨=replaceinds(Wb.𝑨,[Wb.irA,Wb.icA],[Wb.irb,Wb.icc])
            Wb.irA,Wb.icA=Wb.irb,Wb.icc
        end
    end
    if !llur(Wrf,lr) && swap_bc #not lower-left or upper-right
        Wb.𝒃,Wb.𝒄=Wb.𝒄,Wb.𝒃
        Wb.irb,Wb.irc=Wb.irc,Wb.irb
        Wb.icb,Wb.icc=Wb.icc,Wb.icb
    end
    if !isnothing(Wb.𝑨)
        @assert hasinds(Wb.𝑨,Wb.irA,Wb.icA)
    end
    return Wb
end



function set_𝒃_block!(W::ITensor,𝒃::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    W[i1=>2:n1-1,i2=>1:1]=𝒃
end

function set_𝒄_block!(W::ITensor,𝒄::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    W[i1=>n1:n1,i2=>2:n2-1]=𝒄
end
function set_𝒃𝒄_block!(W::ITensor,𝒃𝒄::ITensor,ileft::Index,iright::Index,ms::matrix_state)
    if llur(ms)
        set_𝒃_block!(W,𝒃𝒄,ileft,iright,ms.ul)
    else
        set_𝒄_block!(W,𝒃𝒄,ileft,iright,ms.ul)
    end
end

# noop versions for when b/c are empty.  Happens in edge ops of H.
function set_𝒃𝒄_block!(::ITensor,::Nothing,::Index,::Index,::matrix_state)
end
function set_𝒃_block!(::ITensor,::Nothing,::Index,::Index,::reg_form)
end
function set_𝒄_block!(::ITensor,::Nothing,::Index,::Index,::reg_form)
end

function set_𝒅_block!(W::ITensor,𝒅::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    W[i1=>n1:n1,i2=>1:1]=𝒅
end

function set_𝕀_block!(W::ITensor,𝕀::ITensor,ileft::Index,iright::Index,ul::reg_form)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ul)
    n1>1 && assign!(W,𝕀,i1=>1,i2=>1)
    n2>1 && assign!(W,𝕀,i1=>n1,i2=>n2)
end

function set_𝑨𝒄_block(W::ITensor,𝑨𝒄::ITensor,ileft::Index,iright::Index,ms::matrix_state)
    @assert hasinds(W,ileft,iright)
    i1,i2,n1,n2=swap_ul(ileft,iright,ms.ul)
    if llur(ms) #lower left/upper right
        min1=Base.min(n1,2)
        W[i1=>min1:n1,i2=>2:n2-1]=𝑨𝒄
    else #lower right/upper left
        max2=Base.max(n2-1,1)
        W[i1=>2:n1-1,i2=>1:max2]=𝑨𝒄
    end
end






#
#  o1   add row to RL       o2  add column to RL
#   0   at bottom, Dw1+1    0   at right, Dw2+1
#   1   at top, 1           1   at left, 1
#
function growRL(RL::ITensor,iwl::Index,off::V_offsets,qn::Union{QN,Int})::Tuple{ITensor,Index}
    @mpoc_assert order(RL)==2
    @checkflux(RL)
    irl,=filterinds(inds(RL),tags=tags(iwl)) #find the link index of RL
    irqx=noncommonind(inds(RL),irl) #find the qx link of RL
    @mpoc_assert dim(iwl)==dim(irl)+1
    ipqx=redim(irqx,dim(irqx)+1,off.o1,qn) 
    T=eltype(RL)
    RLplus=ITensor(T(0.0),iwl,ipqx)
    RLplus[ipqx=>1,iwl=>1]=1.0 #add 1.0's in the corners
    RLplus[ipqx=>dim(ipqx),iwl=>dim(iwl)]=1.0
    RLplus[range(ipqx,off.o1),range(iwl,off.o2)]=RL #plop in RL in approtriate sub block.
    @checkflux(RL) 
    return RLplus,dag(ipqx)
end

#
#  factor LR such that for
#       lr=left  LR=M*RM_prime
#       lr=right LR=RL_prime*M
#  However becuase of how the ITensor index works we don't need to distinguish between left and 
#  right matrix multiplication in the code.  THis simplified code requires the RL is square.
#

function getM(RL::ITensor,iqx1::Index,ul::reg_form)::Tuple{ITensor,ITensor,Index}
    @mpoc_assert order(RL)==2
    @checkflux(RL)
    mtags=ts"Link,m"
    iqx,=inds(RL,tags="Link,qx") #Grab the qx link index
    il=noncommonind(RL,iqx) #Grab the remaining link index
    Dw=dim(iqx)
    @mpoc_assert Dw==dim(il) #make sure RL is square
    
    M=RL[iqx=>2:Dw-1,il=>2:Dw-1] #pull out the M sub block
    #
    # Now we need RL_prime such that RL=M*RL_prime.
    # RL_prime is just the perimeter of RL with 1's on the diagonal
    #
    RL=replacetags(RL,tags(iqx),mtags)  #change Link,qx to Link,m
    iqx=replacetags(iqx,tags(iqx),mtags)
    im=redim(iqx,Dw) #new common index between M_plus and RL_prime
    if dir(im)!=dir(iqx1)
        im=dag(im)
    end
    RL_prime=ITensor(0.0,im,il)
    #
    #  Copy over the perimeter of RL.
    #
    if ul==upper
        RL_prime[im=>1:1,il=>1:Dw]=RL[iqx=>1:1 ,il=>1:Dw] #first row
        RL_prime[im=>2:Dw,il=>Dw:Dw]=RL[iqx=>2:Dw,il=>Dw:Dw] #last col
    else
        RL_prime[im=>Dw:Dw,il=>2:Dw]=RL[iqx=>Dw:Dw,il=>2:Dw] #last row
        RL_prime[im=>1:Dw,il=>1:1]=RL[iqx=>1:Dw,il=>1:1] #first col
    end
    
    # Fill in interior diagonal
    #@show inds(RL_prime) im il iqx
    for j1 in 2:Dw-1 #
        RL_prime[im=>j1,il=>j1]=1.0
    end
    @checkflux(RL_prime)
    M=replacetags(M,tags(il),mtags) #change Link,l=n to Link,m
   
    return M,RL_prime,dag(im)
end


function show_blocks(T::ITensor)
    for b in nzblocks(T)
        qns=ntuple(i->space(inds(T)[i])[b[i]],length(inds(T)))
        @show b,qns
    end
end

function my_similar(::DenseTensor{ElT,N},inds...) where {ElT,N}
    return ITensor(inds...)
end

function my_similar(::BlockSparseTensor{ElT,N},inds...) where {ElT,N}
    return ITensor(inds...)
end

function my_similar(::DiagTensor{ElT,N},inds...) where {ElT,N}
    return diagITensor(inds...)
end

function my_similar(::DiagBlockSparseTensor{ElT,N},inds...) where {ElT,N}
    return diagITensor(inds...)
end

function my_similar(T::ITensor,inds...)
    return my_similar(tensor(T),inds...)
end

function warn_space(A::ITensor,ig::Index)
    ia,=inds(A,tags=tags(ig),plev=plev(ig))
    @mpoc_assert dim(ia)+2==dim(ig)
    if hasqns(A)
        sa,sg=space(ia),space(ig)
        if dir(ia)!=dir(ig)
            sa=-sa
        end
        if sa!=sg[2:nblocks(ig)-1]
            @warn "Mismatched spaces:"
            @show sa sg[2:nblocks(ig)-1] dir(ia) dir(ig)
            #@assert false
        end
    end
end
#                      |1 0 0|
#  given A, spit out G=|0 A 0| , indices of G are provided.
#                      |0 0 1|
function grow(A::ITensor,ig1::Index,ig2::Index)
    @checkflux(A) 
    @mpoc_assert order(A)==2
    warn_space(A,ig1)
    warn_space(A,ig2)
    Dw1,Dw2=dim(ig1),dim(ig2)
    G=my_similar(A,ig1,ig2)
    G[ig1=>1  ,ig2=>1  ]=1.0;
    @checkflux(G) 
    G[ig1=>Dw1,ig2=>Dw2]=1.0;
    @checkflux(G) 
    G[ig1=>2:Dw1-1,ig2=>2:Dw2-1]=A
    @checkflux(G) 
    return G
end
