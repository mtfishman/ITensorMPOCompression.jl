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
    d=dim(filterinds(inds(A),tags="Site")[1])
    ils=filterinds(inds(A),tags="Link")
    if length(ils)==2
        t1=tags(ils[1])
        t2=tags(ils[2])
        n1::Int64=tryparse(Int64,String(t1[2])[3:end]) # assume second tag is the "l=n" tag
        n2::Int64=tryparse(Int64,String(t2[2])[3:end]) # assume second tag is the "l=n" tag
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

function is_canonical(W::ITensor,ms::matrix_state,eps::Float64)::Bool
    if ms.lr==left
        V=getV(W,1,1)
    elseif ms.lr==right
        V=getV(W,0,0)
    else
        assert(false)
    end
    d,n,r,c=parse_links(V)
    if ms.lr==left
        rc=c
    elseif ms.lr==right
        rc=r
    else
        assert(false)
    end
    Id=V*prime(V,rc)/d
    Id1=delta(rc,rc)
    return norm(Id-Id1)<eps
end

function is_canonical(H::MPO,ms::matrix_state,eps::Float64)::Bool
    N=length(H)
    if ms.lr==left
        r=1:N-1
    elseif ms.lr==right
        r=2:N
    else
        assert(false)
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

function detect_upper_lower(W::ITensor,eps::Float64)::tri_type
    d,n,r,c=parse_links(W)
    zero_upper=true
    zero_lower=true
    for ir in eachindval(r)
        for ic in eachindval(c)
            oprc=norm(slice(W,ir,ic))
            is_zero=norm(oprc)<eps
            if (ic.second>ir.second) #above diagonal
                zero_upper = zero_upper && is_zero
            end
            if (ic.second<ir.second) #below diagonal
                zero_lower = zero_lower && is_zero
            end
        end
    end
    if zero_upper && zero_lower
        ret=diagonal
    elseif zero_upper
        ret=lower
    elseif zero_lower
        ret=upper
    else
        ret=full
    end
    return ret
end


            


function detect_upper_lower(H::MPO,eps::Float64)::tri_type
    @assert length(H)>1
    ul=detect_upper_lower(H[2],eps) #skip left and right sites in case MPO is obc
    for n in 3:length(H)-1
        iln=detect_upper_lower(H[n],eps)
        if (iln!=ul && iln!=diagonal) 
            if ul==diagonal
                ul=iln
            else
                ul=full
            end
        end
    end
    return ul
end

#
# This test is complicated by two things
#   1) Itis not clear to me (JR) that the A block of an MPO matrix must be upper or lower triangular for
#      block respecting compression to work properly.  Parker et al. make no definitive statement about this.
#      It is my intention to test this empirically using auto MPO generated which tend to non-triangular.
#   2) As a consequence of 1, we cannot decide in advance whether to test of upper or lower regular forms.
#      We must therefore test for both and return true if either one is true. 
#
function is_regular_form(W::ITensor,eps::Float64)::Bool
    d,n,r,c=parse_links(W)
    Dw1,Dw2=dim(r),dim(c)
    #handle edge row and col vectors
    if Dw1==1 #left edge row vector
        return is_unit(slice(W,c=>Dw2),eps)
    end
    if Dw2==1 #right edge col vector
        return is_unit(slice(W,r=>1),eps)
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
    irf=irf && ((top_row_zero&&right_col_zero) || (bot_row_zero && left__col_zero))
    # before returning we should also check for any unit matricies along the diagonal
    # this gets a tricky for non-square matrices.
    diag_unit = false
    if Dw1>=Dw2
        for ic in 2:Dw2-1
            diag_unit = diag_unit || abs(norm(slice(W,r=>ic,c=>ic  ))-sqrt(d))<eps
        end
    else
        dr=Dw2-Dw1
        for ir in 2:Dw1-1
            diag_unit = diag_unit || abs(norm(slice(W,r=>ir,c=>ir+dr))-sqrt(d))<eps
        end
    end
    if diag_unit
        println("ITensorMPOCompression.is_regular_for\n  Warning: found unit operator along the diagonal of an MPO")
    end    
   
    return irf
end

function is_regular_form(H::MPO,eps::Float64)::Bool
    N=length(H)
    irf=true
    for n in 1:N
        irf=irf && is_regular_form(H[n],eps)
        if !irf break end
    end
    return irf
end
