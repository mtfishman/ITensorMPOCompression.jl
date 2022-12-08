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
    nsite,space=parse_site(is)
    d=dim(is)
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
        n=tryparse(Int64,String(t[2])[3:end]) # assume second tag is the "l=n" tag
        if n==Nothing
            return d,nsite,Index(1),ils[1] #probably a qx link, don;t if row or col
        elseif n==1
            return d,nsite,Index(1),ils[1] #row vector
        else
            return d,nsite,ils[1],Index(1) #col vector
        end
    else
        @assert false
    end
end

function parse_site(is::Index)
    @assert hastags(is,"Site")
    nsite=-1
    for t in tags(is)
        ts=String(t)
        if ts[1:2]=="n="
            nsite::Int64=tryparse(Int64,ts[3:end])
        end
        if length(ts)>=4 && ts[1:4]=="Spin"
            space=ts
        end
        if ts[1:2]=="S="
            space=ts[3:end]
        end
    end
    @assert nsite>=0
    return nsite,space
end

function parse_link(il::Index)::Int64
    @assert hastags(il,"Link")
    nsite=-1
    for t in tags(il)
        ts=String(t)
        if ts[1:2]=="l="
            nsite::Int64=tryparse(Int64,ts[3:end])
            break
        end
    end
    @assert nsite>=0
    return nsite
end

#----------------------------------------------------------------------------
#
#  Detection of canonical (orthogonal) forms
#
function is_canonical(r::Index,W::ITensor,c::Index,d::Int64,ms::matrix_state,eps::Float64=default_eps)::Bool
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
    Id1=delta(rc,rc')
    @show Id1
    @show Id-Id1
    if !(norm(Id-Id1)<eps)
        @show norm(Id-Id1) eps
    end
    return norm(Id-Id1)<eps
end


function is_canonical(W::ITensor,ms::matrix_state,eps::Float64=default_eps)::Bool
    V=getV(W,V_offsets(ms))
    d,n,r,c=parse_links(V)
    if ms.lr==left
        rc=c
    else #right
        rc=r
    end
    Id=V*prime(dag(V),rc)/d
    if order(Id)==2
        is_can = norm(dense(Id)-delta(rc,dag(rc')))<eps
    elseif order(Id)==0
        is_can = abs(scalar(Id)-d)<eps
    end
    return is_can    
end

function is_canonical(H::MPO,ms::matrix_state,eps::Float64=default_eps)::Bool
    N=length(H)
    ic=true
    for n in 2:N-1 #skip the edge row/col opertors
        ic=ic &&  is_canonical(H[n],ms,eps)
    end
    return ic
end

function is_canonical(H::MPO,lr::orth_type,eps::Float64=default_eps)::Bool
    l,u=detect_regular_form(H)
    @assert u || l
    ul = u ? upper : lower
    return is_canonical(H,matrix_state(ul,lr),eps)
end

@doc """
is_orthogonal(H,lr[,eps])::Bool

Test if all sites in an MPO statisfty the condition for `lr` orthogonal (canonical) form.

# Arguments
- `H:MPO` : MPO to be characterized.
- `lr::orth_type` : choose `left` or `right` orthogonality condition to test for.
- `eps::Float64 = 1e-14` : operators inside H with norm(W[i,j])<eps are assumed to be zero.

Returns `true` if the MPO is in `lr` orthogonal (canonical) form
"""
is_orthogonal(H::MPO,lr::orth_type,eps::Float64=default_eps)::Bool = is_canonical(H,lr,eps)

#-

#----------------------------------------------------------------------------
#
#  Detection of upper and lower triangular forms
#
#

# It could be both or neither, so we return two Bools corresponding to
# (lower,upper) triangular
#
function detect_upper_lower(r::Index,W::ITensor,c::Index,eps::Float64=default_eps)::Tuple{Bool,Bool,Char}
    if dim(r)==1 && dim(c)==1
        return true,true,'1'
    elseif dim(r)==1
        return true,true,'R'
    elseif dim(c)==1
        return true,true,'C'
    end
    ord=order(W)
    zero_upper=true
    zero_lower=true
    dr=Base.max(0,dim(c)-dim(r))
    dc=Base.max(0,dim(r)-dim(c))
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
    return zero_upper,zero_lower,' '
end

function is_upper_lower(r::Index,W::ITensor,c::Index,ul::reg_form,eps::Float64=default_eps)::Bool
    l,u=detect_upper_lower(r,W,c,eps)
    ul==lower ? l : u
end

is_lower(r::Index,W::ITensor,c::Index,eps::Float64=default_eps) = is_upper_lower(r,W,c,lower,eps)
is_upper(r::Index,W::ITensor,c::Index,eps::Float64=default_eps) = is_upper_lower(r,W,c,upper,eps)

function detect_upper_lower(W::ITensor,eps::Float64=default_eps)::Tuple{Bool,Bool,Char}
    d,n,r,c=parse_links(W)
    return detect_upper_lower(r,W,c,eps)
end

function is_upper_lower(W::ITensor,ul::reg_form,eps::Float64=default_eps)::Bool 
    l,u=detect_upper_lower(W,eps)
    ul==lower ? l : u
end

is_lower(W::ITensor,eps::Float64=default_eps)::Bool = is_upper_lower(W,lower,eps) 
is_upper(W::ITensor,eps::Float64=default_eps)::Bool = is_upper_lower(W,upper,eps) 
    

function detect_upper_lower(H::MPO,eps::Float64=default_eps)::Tuple{Bool,Bool}
    @assert length(H)>1
    l,u=detect_upper_lower(H[2],eps) #skip left and right sites in case MPO is obc
    for n in 3:length(H)-1
        il,iu=detect_upper_lower(H[n],eps)
        l = l && il
        u = u && iu
    end
    return l,u
end

function is_upper_lower(H::MPO,ul::reg_form,eps::Float64=default_eps)::Bool 
    l,u=detect_upper_lower(H,eps)
    ul==lower ? l : u
end

is_lower(H::MPO,eps::Float64=default_eps)::Bool = is_upper_lower(H,lower,eps) 
is_upper(H::MPO,eps::Float64=default_eps)::Bool = is_upper_lower(H,upper,eps) 


#----------------------------------------------------------------------------
#
#  Detection of upper and lower regular forms
#


#
# This test is complicated by two things
#   1) It is not clear to me (JR) that the A block of an MPO matrix must be upper or lower triangular for
#      block respecting compression to work properly.  Parker et al. make no definitive statement about this.
#      It is my intention to test this empirically using auto MPO generated Hamiltonians 
#      which tend to be non-triangular.
#   2) As a consequence of 1, we cannot decide in advance whether to test for upper or lower regular forms.
#      We must therefore test for both and return true if either one is true. 
#

@doc """
    detect_regular_form(W[,eps])::Tuple{Bool,Bool}
    
Inspect the structure of an operator-valued matrix W to see if it satisfies the regular form 
conditions as specified in Section III, definition 3 of
> *Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147*

# Arguments
- `W::ITensor` : operator-valued matrix to be characterized. W is expected to have 2 "Site" indices and 1 or 2 "Link" indices
- `eps::Float64 = 1e-14` : operators inside W with norm(W[i,j])<eps are assumed to be zero.

# Returns a Tuple containing
- `reg_lower::Bool` Indicates W is in lower regular form.
- `reg_upper::Bool` Indicates W is in upper regular form.
The function returns two Bools in order to handle cases where W is not in regular form, returning 
(false,false) and W is in a special pseudo diagonal regular form, returning (true,true).
    
"""
function detect_regular_form(W::ITensor,eps::Float64=default_eps)::Tuple{Bool,Bool}
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
    # this gets a bit tricky for non-square matrices.
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
    #
    #  Not sure how to hanlde this right now ... unit ops seem to appear with alarming
    #  frequency after compression.
    #
    # if diag_unit
    #     pprint(W,eps)
    #     println("ITensorMPOCompression.is_regular_form\n  Warning: found unit operator along the diagonal of an MPO")
    # end    
   
    return reg_lower,reg_upper
end

@doc """
    is_regular_form(W,ul[,eps])::Bool

Determine if an operator-valued matrix, `W`, is in `ul` regular form.

# Arguments
- `W::ITensor` : operator-valued matrix to be characterized. `W` is expected to have 2 site indices and 
    1 or 2 link indices
- 'ul::reg_form' : choose `lower` or `upper`.
- `eps::Float64 = 1e-14` : operators inside `W` with norm(W[i,j])<eps are assumed to be zero.

# Returns 
- `true` if `W` is in `ul` regular form.

"""
function is_regular_form(W::ITensor,ul::reg_form,eps::Float64=default_eps)::Bool
    i = ul==lower ? 1 : 2
    return detect_regular_form(W,eps)[i]
end

@doc """
    is_lower_regular_form(W[,eps])::Bool

Determine if an operator-valued matrix, `W`, is in lower regular form.

# Arguments
- `W::ITensor` : operator-valued matrix to be characterized. `W` is expected to have 2 site indices and 
    1 or 2 link indices
- `eps::Float64 = 1e-14` : operators inside `W` with norm(W[i,j])<eps are assumed to be zero.

# Returns 
- `true` if `W` is in lower regular form.

"""
function is_lower_regular_form(W::ITensor,eps::Float64=default_eps)::Bool
    return detect_regular_form(W,eps)[1]
end

@doc """
    is_upper_regular_form(W[,eps])::Bool

Determine if an operator-valued matrix, `W`, is in upper regular form.

# Arguments
- `W::ITensor` : operator-valued matrix to be characterized. `W` is expected to have 2 site indices and 
    1 or 2 link indices
- `eps::Float64 = 1e-14` : operators inside `W` with norm(W[i,j])<eps are assumed to be zero.

# Returns
- `true` if `W` is in upper regular form.

"""
function is_upper_regular_form(W::ITensor,eps::Float64=default_eps)::Bool
    return detect_regular_form(W,eps)[2]
end

@doc """
    is_regular_form(H[,eps])::Bool

Determine if an MPO, `H`, is in either lower xor upper regular form. All sites in H must be
in the same (lower or upper) regular form in order to return true.  In other words mixtures
of lower and upper will fail.

# Arguments
- `H::MPO` : MPO to be characterized. 
- `eps::Float64 = 1e-14` : operators inside `H` with norm(W[i,j])<eps are assumed to be zero.

# Returns 
- `true` if *all* sites in `H` are in either lower xor upper regular form.

"""
function is_regular_form(H::MPO,eps::Float64=default_eps)::Bool
    N=length(H)
    lrf,urf=true,true
    for n in 1:N
        l,u=detect_regular_form(H[n],eps) #there must be some trick to get all this onto one line
        lrf=lrf && l
        urf=urf && u
    end
    return lrf || url
end

@doc """
    is_regular_form(H,ul[,eps])::Bool

Determine is a MPO, `H`, is in `ul` regular form. All sites in H must
in the same `ul` regular form in order to return true.

# Arguments
- `H::MPO` : MPO to be characterized. 
- 'ul::reg_form' : choose `lower` or `upper`.
- `eps::Float64 = 1e-14` : operators inside `H` with norm(W[i,j])<eps are assumed to be zero.

# Returns 
- `true` if *all* sites in `H` are in `ul` regular form.

"""
function is_regular_form(H::MPO,ul::reg_form,eps::Float64=default_eps)::Bool
    N=length(H)
    irf=true
    for n in 1:N
        irf=irf && is_regular_form(H[n],ul,eps)
    end
    return irf
end

@doc """
    detect_regular_form(H[,eps])::Tuple{Bool,Bool}
    
Inspect the structure of an MPO `H` to see if it satisfies the regular form conditions.

# Arguments
- `H::MPO` : MPO to be characterized.
- `eps::Float64 = 1e-14` : operators inside `W` with norm(W[i,j])<eps are assumed to be zero.

# Returns a Tuple containing
- `reg_lower::Bool` Indicates all sites in `H` are in lower regular form.
- `reg_upper::Bool` Indicates all sites in `H` are in upper regular form.
The function returns two Bools in order to handle cases where `H` is not regular form, returning (`false`,`false`) and `H` is in a special pseudo-diagonal regular form, returning (`true`,`true`).
    
"""
function detect_regular_form(H::MPO,eps::Float64=default_eps)::Tuple{Bool,Bool}
    N=length(H)
    l,u=true,true
    for n in 1:N
        ln,un=detect_regular_form(H[n],eps)
        l= l && ln
        u= u && un
    end
    return l,u
end

@doc """
    is_lower_regular_form(H[,eps])::Bool

Determine if all sites in an MPO, `H`, are in lower regular form.

# Arguments
- `H::MPO` : MPO to be characterized. 
- `eps::Float64 = 1e-14` : operators inside `H` with norm(W[i,j])<eps are assumed to be zero.

# Returns 
- `true` if *all* sites in `H` are in lower regular form.

"""
function is_lower_regular_form(H::MPO,eps::Float64=default_eps)::Bool
    return is_regular_form(H,lower,eps)
end

@doc """
    is_upper_regular_form(H[,eps])::Bool

Determine if all sites in an MPO, `H`, are in upper regular form.

# Arguments
- `H::MPO` : MPO to be characterized. 
- `eps::Float64 = 1e-14` : operators inside `H` with norm(W[i,j])<eps are assumed to be zero.

# Returns 
- `true` if *all* sites in `H` are in upper regular form.

"""
function is_upper_regular_form(H::MPO,eps::Float64=default_eps)::Bool
    return is_regular_form(H,upper,eps)
end

function get_Dw(H::MPO)::Vector{Int64}
    N=length(H)
    Dws=Vector{Int64}(undef,N-1)
    for n in 1:N-1
        d,nsite,r,c=parse_links(H[n])
        Dws[n]=dim(c)
    end
    return Dws
end
    
function get_traits(W::ITensor,eps::Float64)
    d,n,r,c=parse_links(W)
    Dw1,Dw2=dim(r),dim(c)
    bl,bu = detect_regular_form(W,eps)
    l= bl ? 'L' : ' '
    u= bu ? 'U' : ' '
    if bl && bu
        l='D'
        u=' '
    elseif !(bl || bu)
        l='N'
        u='o'
    end

    tri = bl ? lower : upper
    msl=matrix_state(tri,left)
    msr=matrix_state(tri,right)
    is__left=is_canonical(W,msl,eps)
    is_right=is_canonical(W,msr,eps)
    if is__left && is_right
        lr='B'
    elseif is__left && !is_right
        lr='L'
    elseif !is__left && is_right
        lr='R'
    else
        lr='M'
    end

    bl,bu,RC=detect_upper_lower(W)
    lt= bl ? 'L' : ' '
    ut= bu ? 'U' : ' '
    if RC!=' '
        lt=RC
        ut=' '
    elseif bl && bu
        lt='D'
        ut=' '
    elseif !(bl || bu)
        lt='F'
        ut=' '
    end
    return Dw1,Dw2,d,l,u,lr,lt,ut
end
    
