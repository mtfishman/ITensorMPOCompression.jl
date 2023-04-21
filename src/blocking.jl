
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
    𝐀̂::Union{ITensor,Nothing}
    𝐀̂𝐜̂::Union{ITensor,Nothing}
    𝐕̂::Union{ITensor,Nothing}
    𝐛̂::Union{ITensor,Nothing}
    𝐜̂::Union{ITensor,Nothing}
    𝐝̂::Union{ITensor,Nothing}
    irA::Union{Index,Nothing}
    icA::Union{Index,Nothing}
    irAc::Union{Index,Nothing}
    icAc::Union{Index,Nothing}
    irV::Union{Index,Nothing}
    icV::Union{Index,Nothing}
    irb::Union{Index,Nothing}
    icb::Union{Index,Nothing}
    irc::Union{Index,Nothing}
    icc::Union{Index,Nothing}
    ird::Union{Index,Nothing}
    icd::Union{Index,Nothing}    
    regform_blocks()=new(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing)
end

d(Wb::regform_blocks)::Float64=scalar(Wb.𝕀*dag(Wb.𝕀))
b0(Wb::regform_blocks)::ITensor=Wb.𝐛̂*dag(Wb.𝕀)/d(Wb)
c0(Wb::regform_blocks)::ITensor=Wb.𝐜̂*dag(Wb.𝕀)/d(Wb)
A0(Wb::regform_blocks)::ITensor=Wb.𝐀̂*dag(Wb.𝕀)/d(Wb)

#
#  Transpose inds for upper, no-op for lower
#
function swap_ul(ileft::Index,iright::Index,ul::reg_form)
    return ul==lower ? (ileft,iright,dim(ileft),dim(iright)) :  (iright,ileft,dim(iright),dim(ileft))
end
function swap_ul(Wrf::reg_form_Op)
    return Wrf.ul==lower ? (Wrf.ileft,Wrf.iright,dim(Wrf.ileft),dim(Wrf.iright)) :  (Wrf.iright,Wrf.ileft,dim(Wrf.iright),dim(Wrf.ileft))
end
# lower left or upper right
llur(ul::reg_form,lr::orth_type)= lr==left && ul==lower || lr==right && ul==upper
llur(W::reg_form_Op,lr::orth_type)=llur(W.ul,lr)

#  Use recognizably distinct UTF symbols for operators, and op valued vectors and matrices: 
#  𝐀̂ 𝐛̂ 𝐜̂ 𝐝̂ 𝐕̂ 

function extract_blocks(Wrf::reg_form_Op,lr::orth_type;all=false,c=false,b=false,d=false,A=false,Ac=false,V=false,I=true,fix_inds=false,swap_bc=true)::regform_blocks
    check(Wrf)
    @assert plev(Wrf.ileft)==0
    @assert plev(Wrf.iright)==0
    W=Wrf.W
    ir,ic=linkinds(Wrf)
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
        Wb.𝐀̂= W[ir=>2:nr-1,ic=>2:nc-1]
        Wb.irA,=inds(Wb.𝐀̂,tags=tags(ir))
        Wb.icA,=inds(Wb.𝐀̂,tags=tags(ic))
    end
    if Ac
        if llur(Wrf,lr)
            Wb.𝐀̂𝐜̂= nr>1 ? W[ir=>2:nr,ic=>2:nc-1] : W[ir=>1:1,ic=>2:nc-1]
        else
            Wb.𝐀̂𝐜̂= nc>1 ? W[ir=>2:nr-1,ic=>1:nc-1]  : W[ir=>2:nr-1,ic=>1:1]
        end
        Wb.irAc,=inds(Wb.𝐀̂𝐜̂,tags=tags(ir))
        Wb.icAc,=inds(Wb.𝐀̂𝐜̂,tags=tags(ic))
    end
    if V
        i1,i2,n1,n2=swap_ul(Wrf)
        if llur(Wrf,lr) #lower left/upper right
            min1=Base.min(n1,2)
            min2=Base.min(n2,2)
            Wb.𝐕̂=W[i1=>min1:n1,i2=>min2:n2] #Bottom right corner
        else #lower right/upper left
            max1=Base.max(n1-1,1)
            max2=Base.max(n2-1,1)
            Wb.𝐕̂=W[i1=>1:max1,i2=>1:max2] #top left corner
        end
        Wb.irV,=inds(Wb.𝐕̂,tags=tags(ir))
        Wb.icV,=inds(Wb.𝐕̂,tags=tags(ic))
    end
    if b
        Wb.𝐛̂= W[ir=>2:nr-1,ic=>1:1]
        Wb.irb,=inds(Wb.𝐛̂,tags=tags(ir))
        Wb.icb,=inds(Wb.𝐛̂,tags=tags(ic))
    end
    if c
        Wb.𝐜̂= W[ir=>nr:nr,ic=>2:nc-1]
        Wb.irc,=inds(Wb.𝐜̂,tags=tags(ir))
        Wb.icc,=inds(Wb.𝐜̂,tags=tags(ic))
    end
    if d
        Wb.𝐝̂= nr >1 ? W[ir=>nr:nr,ic=>1:1] : W[ir=>1:1,ic=>1:1]
        Wb.ird,=inds(Wb.𝐝̂,tags=tags(ir))
        Wb.icd,=inds(Wb.𝐝̂,tags=tags(ic))
    end

    if fix_inds
        if !isnothing(Wb.𝐜̂)
            Wb.𝐜̂=replaceind(Wb.𝐜̂,Wb.irc,Wb.ird)
            Wb.irc=Wb.ird
        end
        if !isnothing(Wb.𝐛̂)
            Wb.𝐛̂=replaceind(Wb.𝐛̂,Wb.icb,Wb.icd)
            Wb.icb=Wb.icd
        end
        if !isnothing(Wb.𝐀̂)
            Wb.𝐀̂=replaceinds(Wb.𝐀̂,[Wb.irA,Wb.icA],[Wb.irb,Wb.icc])
            Wb.irA,Wb.icA=Wb.irb,Wb.icc
        end
    end
    if !llur(Wrf,lr) && swap_bc #not lower-left or upper-right
        Wb.𝐛̂,Wb.𝐜̂=Wb.𝐜̂,Wb.𝐛̂
        Wb.irb,Wb.irc=Wb.irc,Wb.irb
        Wb.icb,Wb.icc=Wb.icc,Wb.icb
    end
    if !isnothing(Wb.𝐀̂)
        @assert hasinds(Wb.𝐀̂,Wb.irA,Wb.icA)
    end
    return Wb
end


function set_𝐛̂_block!(Wrf::reg_form_Op,𝐛̂::ITensor)
    check(Wrf)
    i1,i2,n1,n2=swap_ul(Wrf)
    Wrf.W[i1=>2:n1-1,i2=>1:1]=𝐛̂
end

function set_𝐜̂_block!(Wrf::reg_form_Op,𝐜̂::ITensor)
    check(Wrf)
    i1,i2,n1,n2=swap_ul(Wrf)
    Wrf.W[i1=>n1:n1,i2=>2:n2-1]=𝐜̂
end

function set_𝐛̂𝐜̂_block!(Wrf::reg_form_Op,𝐛̂𝐜̂::ITensor,lr::orth_type)
    if llur(Wrf,lr)
        set_𝐛̂_block!(Wrf,𝐛̂𝐜̂)
    else
        set_𝐜̂_block!(Wrf,𝐛̂𝐜̂)
    end
end



function set_𝐝̂_block!(Wrf::reg_form_Op,𝐝̂::ITensor)
    check(Wrf)
    i1,i2,n1,n2=swap_ul(Wrf)
    Wrf.W[i1=>n1:n1,i2=>1:1]=𝐝̂
end

function set_𝕀_block!(Wrf::reg_form_Op,𝕀::ITensor)
    check(Wrf)
    i1,i2,n1,n2=swap_ul(Wrf)
    n1>1 && assign!(Wrf.W,𝕀,i1=>1,i2=>1)
    n2>1 && assign!(Wrf.W,𝕀,i1=>n1,i2=>n2)
end

function set_𝐀̂𝐜̂_block(Wrf::reg_form_Op,𝐀̂𝐜̂::ITensor,lr::orth_type)
    check(Wrf)
    i1,i2,n1,n2=swap_ul(Wrf)
    if llur(Wrf,lr) #lower left/upper right
        min1=Base.min(n1,2)
        Wrf.W[i1=>min1:n1,i2=>2:n2-1]=𝐀̂𝐜̂
    else #lower right/upper left
        max2=Base.max(n2-1,1)
        Wrf.W[i1=>2:n1-1,i2=>1:max2]=𝐀̂𝐜̂
    end
end
# noop versions for when b/c are empty.  Happens in edge ops of H.
function set_𝐛̂𝐜̂_block!(::reg_form_Op,::Nothing,::orth_type)
end
function set_𝐛̂_block!(::reg_form_Op,::Nothing)
end
function set_𝐜̂_block!(::reg_form_Op,::Nothing)
end

# 
#  Given R, build R⎖ such that lr=left  R=M*R⎖, lr=right R=R⎖*M
#
function build_R⎖(R::ITensor,iqx::Index,ilf::Index,ul::reg_form)::Tuple{ITensor,Index}
    @mpoc_assert order(R)==2
    @mpoc_assert hasinds(R,iqx,ilf)
    @mpoc_assert dim(iqx)==dim(ilf) #make sure RL is square
    @checkflux(R)
    im=Index(space(iqx),tags=tags(iqx),dir=dir(iqx),plev=1) #new common index between M and R⎖
    R⎖=ITensor(0.0,im,ilf) 
    #R⎖+=δ(im,ilf) #set diagonal ... blocksparse + diag is not supported yet.  So we do it manually below.
    Dw=dim(im)
    for j1 in 2:Dw-1 
        R⎖[im=>j1,ilf=>j1]=1.0 # Fill in the interior diagonal
    end
    #
    #  Copy over the perimeter of RL.
    #
    if ul==upper
        R⎖[im=>1:1,ilf=>1:Dw]=R[iqx=>1:1 ,ilf=>1:Dw] #first row
        R⎖[im=>2:Dw,ilf=>Dw:Dw]=R[iqx=>2:Dw,ilf=>Dw:Dw] #last col
    else
        R⎖[im=>Dw:Dw,ilf=>2:Dw]=R[iqx=>Dw:Dw,ilf=>2:Dw] #last row
        R⎖[im=>1:Dw,ilf=>1:1]=R[iqx=>1:Dw,ilf=>1:1] #first col
    end
    @checkflux(R⎖)
    #
    #  Fix up index tags and primes.
    #
    im=noprime(settags(im,"Link,m"))
    RL_prime=noprime(replacetags(R⎖,tags(iqx),tags(im),plev=1))
    
    return RL_prime,dag(im)
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
