#
#  Functions for characterizing operator tensors
#

#
#  Figure out the site number, and left and indicies of an MPS or MPO ITensor
#  assumes:
#       1) all tensors have two link indices (edge sites no handled yet)
#       2) link indices all have a "Link" tag
#       3) the "Link" tag is the first tag for each link index
#       4) the second tag has the for "l=nnnnn"  where nnnnn are the integer digits of the link number
#       5) the site number is the larges of the link numbers
#
#  Obviously a lot of things could go wrong with all these assumptions.
#
function parse_links(A::ITensor)::Tuple{Int64,Int64,Index,Index}
    #
    #  find any site index and try and extract the site number
    #  assume third tag is always the "n=$nsite" site number tag
    #
    is=filterinds(inds(A),tags="Site")[1]
    d=dim(is)
    ts2=String(tags(is)[3])
    @assert ts2[1:2]=="n="
    nsite::Int64=tryparse(Int64,ts2[3:end])
    #
    #  Now process the link tags
    #
    ils=filterinds(inds(A),tags="Link")
    if length(ils)==2
        tl1=String(tags(ils[1])[2])
        tl2=String(tags(ils[2])[2])
        if tl1[1:2]=="l="
            n1::Int64=tryparse(Int64,tl1[3:end]) # assume second tag is the "l=n" tag
        else
            n1=-1
        end
        if tl2[1:2]=="l="
            n2::Int64=tryparse(Int64,tl2[3:end]) # assume second tag is the "l=n" tag
        else
            n2=-1
        end
        if n1==-1 && n2==-1
            @show ils
            @assert false
        end
        if n1==-1
            if n2==nsite
                n1=nsite-1 
            elseif n2==nsite-1
                n1=nsite 
            else
                @assert false
            end
        end
        if n2==-1
            if n1==nsite
                n2=nsite-1 
            elseif n1==nsite-1
                n2=nsite 
            else
                @assert false
            end
        end     
        if n1>n2
            return d,n1,ils[2],ils[1]
        else 
            return d,n2,ils[1],ils[2]
        end
    elseif length(ils)==1
        t=tags(ils[1])
        n::Int64=tryparse(Int64,String(t[2])[3:end]) # assume second tag is the "l=n" tag
        if n==1
            return d,n,Index(1),ils[1] #row vector
        else
            return d,n,ils[1],Index(1) #col vector
        end
    else
        @assert false
    end
end

function is_canonical(r::Index,W::ITensor,c::Index,d::Int64,ms::matrix_state,eps::Float64)::Bool
    V=getV(W,V_offsets(ms))
    rv=findinds(V,tags(r))[1]
    cv=findinds(V,tags(c))[1]
    if ms.lr==left
        rc=cv
    else
        rc=rv
    end
    #@show inds(V) rv,cv,rc
    Id=V*prime(V,rc)/d
    @show Id
    Id1=delta(rc,rc)
    @show Id1
    @show Id-Id1
    if !(norm(Id-Id1)<eps)
        @show norm(Id-Id1) eps
    end
    return norm(Id-Id1)<eps
end


function is_canonical(W::ITensor,ms::matrix_state,eps::Float64)::Bool
    V=getV(W,V_offsets(ms))
    d,n,r,c=parse_links(V)
    if ms.lr==left
        rc=c
    else #right
        rc=r
    end
    Id=V*prime(V,rc)/d
    Id1=delta(rc,rc)
    return norm(Id-Id1)<eps
end

function is_canonical(H::MPO,ms::matrix_state,eps::Float64)::Bool
    N=length(H)
    if ms.lr==left
        r=1:N-1
    else # right
        r=2:N
    end
    ic=true
    for n in r
        ic=ic &&  is_canonical(H[n],ms,eps)
    end
    return ic
end

function has_pbc(H::MPO)::Bool
    N=length(H)
    nind1=length(inds(H[1]))
    nindN=length(inds(H[N]))
    nlink1=length(findinds(H[N],"Link"))
    nlinkN=length(findinds(H[N],"Link"))
    leftl=hastags(inds(H[1]),"l=0")
    rightl=hastags(inds(H[N]),"l=$N")
    obc::Bool = nind1==3 && nindN==3 &&  nlink1==1 && nlinkN==1 && leftl==0 && rightl==0
    pbc::Bool = nind1==4 && nindN==4 &&  nlink1==2 && nlinkN==2 && leftl==1 && rightl==1
    if !obc && !pbc  #if its not one or the othr something is really messed up!
        @assert false
    end
    return pbc
end


#
# It could be both or neither, so we return two Bools corresponding to
# (lower,upper) triangular
#
function detect_upper_lower(r::Index,W::ITensor,c::Index,eps::Float64)::Tuple{Bool,Bool}
    @assert hasind(W,r)
    @assert hasind(W,c)
    ord=order(W)
    @assert ord==4 || ord==2
    zero_upper=true
    zero_lower=true
    dr=max(0,dim(c)-dim(r))
    dc=max(0,dim(r)-dim(c))
    for ir in eachindval(r)
        for ic in eachindval(c)
            if ord==4
                oprc=norm(slice(W,ir,ic))
            else #ord=2
                oprc=abs(W[ir,ic])
            end
            is_zero=oprc<eps
            if (ic.second>ir.second+dr) #above diagonal
                zero_upper = zero_upper && is_zero
            end
            if (ic.second+dc<ir.second) #below diagonal
                zero_lower = zero_lower && is_zero
            end
        end
    end
    return zero_upper,zero_lower
end

function is_upper_lower(r::Index,W::ITensor,c::Index,ul::tri_type,eps::Float64)::Bool
    l,u=detect_upper_lower(r,W,c,eps)
    ul==lower ? l : u
end

is_lower(r::Index,W::ITensor,c::Index,eps::Float64) = is_upper_lower(r,W,c,lower,eps)
is_upper(r::Index,W::ITensor,c::Index,eps::Float64) = is_upper_lower(r,W,c,upper,eps)

function detect_upper_lower(W::ITensor,eps::Float64)::Tuple{Bool,Bool}
    d,n,r,c=parse_links(W)
    return detect_upper_lower(r,W,c,eps)
end

function is_upper_lower(W::ITensor,ul::tri_type,eps::Float64)::Bool 
    l,u=detect_upper_lower(W,eps)
    ul==lower ? l : u
end

is_lower(W::ITensor,eps::Float64)::Bool = is_upper_lower(W,lower,eps) 
is_upper(W::ITensor,eps::Float64)::Bool = is_upper_lower(W,upper,eps) 
    

function detect_upper_lower(H::MPO,eps::Float64)::Tuple{Bool,Bool}
    @assert length(H)>1
    l,u=detect_upper_lower(H[2],eps) #skip left and right sites in case MPO is obc
    for n in 3:length(H)-1
        il,iu=detect_upper_lower(H[n],eps)
        l = l && il
        u = u && iu
    end
    return l,u
end

function is_upper_lower(H::MPO,ul::tri_type,eps::Float64)::Bool 
    l,u=detect_upper_lower(H,eps)
    ul==lower ? l : u
end

is_lower(H::MPO,eps::Float64)::Bool = is_upper_lower(H,lower,eps) 
is_upper(H::MPO,eps::Float64)::Bool = is_upper_lower(H,upper,eps) 

#
# This test is complicated by two things
#   1) It is not clear to me (JR) that the A block of an MPO matrix must be upper or lower triangular for
#      block respecting compression to work properly.  Parker et al. make no definitive statement about this.
#      It is my intention to test this empirically using auto MPO generated Hamiltonians 
#      which tend to non-triangular.
#   2) As a consequence of 1, we cannot decide in advance whether to test for upper or lower regular forms.
#      We must therefore test for both and return true if either one is true. 
#
function is_regular_form(W::ITensor,eps::Float64)::Tuple{Bool,Bool}
    d,n,r,c=parse_links(W)
    Dw1,Dw2=dim(r),dim(c)
    #handle edge row and col vectors
    if Dw1==1 #left edge row vector
        return is_unit(slice(W,c=>Dw2),eps),is_unit(slice(W,c=>1),eps)
    end
    if Dw2==1 #right edge col vector
        return is_unit(slice(W,r=>1),eps),is_unit(slice(W,r=>Dw1),eps)
    end
    #ul=detect_upper_lower(W) is the a requirement??
    irf=true #is regular form
    # There must be unit matrices in the top left and bottom right corners.
    irf=irf && is_unit(slice(W,r=>1  ,c=>1  ),eps)
    irf=irf && is_unit(slice(W,r=>Dw1,c=>Dw2),eps)
    # now look for zeroed partial rows and columns, avoiding the corner unit matricies.
    top_row_zero=true #lower
    bot_row_zero=true #upper tri
    left__col_zero=true #upper tri
    right_col_zero=true #lower tri
    #lower tri zero tests
    for ic in 2:Dw2
        top_row_zero=top_row_zero     && norm(slice(W,r=>1  ,c=>ic))<eps
    end
    for ir in 1:Dw1-1
        right_col_zero=right_col_zero && norm(slice(W,r=>ir,c=>Dw2))<eps
    end
    #upper tri zero tests
    for ic in 1:Dw2-1
        bot_row_zero=bot_row_zero     && norm(slice(W,r=>Dw1,c=>ic))<eps
    end
    for ir in 2:Dw1
        left__col_zero=left__col_zero && norm(slice(W,r=>ir,c=>1  ))<eps
    end
    reg_lower=irf && top_row_zero && right_col_zero
    reg_upper=irf && bot_row_zero && left__col_zero
    # before returning we should also check for any unit matricies along the diagonal
    # this gets a tricky for non-square matrices.
    diag_unit = false
    if Dw1>=Dw2
        for ic in 2:Dw2-1
            diag_unit = diag_unit || is_unit(slice(W,r=>ic,c=>ic  ),eps)
        end
    else
        dr=Dw2-Dw1
        for ir in 2:Dw1-1
            diag_unit = diag_unit || is_unit(slice(W,r=>ir,c=>ir+dr),eps)
        end
    end
    if diag_unit
        #pprint(W,eps)
        println("ITensorMPOCompression.is_regular_form\n  Warning: found unit operator along the diagonal of an MPO")
    end    
   
    return reg_lower,reg_upper
end

function is_regular_form(W::ITensor,ul::tri_type,eps::Float64)::Bool
    i = ul==lower ? 1 : 2
    return is_regular_form(W,eps)[i]
end

function is_lower_regular_form(W::ITensor,eps::Float64)::Bool
    return is_regular_form(W,eps)[1]
end

function is_upper_regular_form(W::ITensor,eps::Float64)::Bool
    return is_regular_form(W,eps)[2]
end


function is_regular_form(H::MPO,eps::Float64)::Bool
    N=length(H)
    irf=true,true
    for n in 1:N
        irf=irf && is_regular_form(H[n],eps)
    end
    return irf[1] || irf[2]
end

function is_regular_form(H::MPO,ul::tri_type,eps::Float64)::Bool
    N=length(H)
    irf=true
    for n in 1:N
        irf=irf && is_regular_form(H[n],ul,eps)
    end
    return irf
end

function is_lower_regular_form(H::MPO,eps::Float64)::Bool
    return is_regular_form(H,lower,eps)
end

function is_upper_regular_form(H::MPO,eps::Float64)::Bool
    return is_regular_form(H,upper,eps)
end

function get_traits(W::ITensor,eps::Float64)
    d,n,r,c=parse_links(W)
    Dw1,Dw2=dim(r),dim(c)
    bl,bu = is_regular_form(W,eps)
    l= bl ? 'L' : ' '
    u= bu ? 'U' : ' '

    tri = bl ? lower : upper
    msl=matrix_state(tri,left)
    msr=matrix_state(tri,right)
    is__left=is_canonical(W,msl,eps)
    is_right=is_canonical(W,msr,eps)
    if is__left && is_right
        lr='I'
    elseif is__left && !is_right
        lr='L'
    elseif !is__left && is_right
        lr='R'
    else
        lr='M'
    end

    return Dw1,Dw2,d,l,u,lr
end
    
function pprint(H::MPO,eps::Float64)
    N=length(H)
    println(" n   Dw1   Dw2   d   U/L   L/R")
    for n in 1:N
        Dw1,Dw2,d,l,u,lr=get_traits(H[n],eps)
        println(" $n    $Dw1     $Dw2    $d    $l$u     $lr")
    end
end
