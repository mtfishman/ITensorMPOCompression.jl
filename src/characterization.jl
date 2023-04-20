#
#  Functions for characterizing operator tensors
#

#
#  Figure out the site number, and left and indicies of an MPS or MPO ITensor
#  assumes:
#       1) link indices all have a "Link" tag
#       2) the "Link" tag is the first tag for each link index
#       3) one other tag has the form "l=$n"  where n is the link number
#       4) the site number is the largest of the link numbers, i.e. left link is l=n-1
#
#  Obviously a lot of things could go wrong with all these assumptions.
#
#  returns forward,reverse relatvie to sweep direction indexes
#  if lr=left, then the sweep direction is o the right, and i_right is the forward index.
function parse_links(A::ITensor,lr::orth_type)::Tuple{Index,Index}
    i_left,i_right = parse_links(A)
    return lr==left ? (i_right,i_left) : (i_left,i_right)
end

function parse_links(A::ITensor)::Tuple{Index,Index}
    #
    #  find any site index and try and extract the site number
    #
    d,nsite,space=parse_site(A)
    #
    #  Now process the link tags
    #
    ils=filterinds(inds(A),tags="Link")
    if length(ils)==2
        n1,c1= parse_link(ils[1])
        n2,c2= parse_link(ils[2]) #find the "l=$n" tags. -1 if not such tag
        n1,n2=infer_site_numbers(n1,n2,nsite) #handle qx links with no l=$n tags.
        if c1>c2
            return ils[2],ils[1]
        elseif c2>c1
            return ils[1],ils[2]
        elseif n1>n2
            return ils[2],ils[1]
        elseif n2>n1
            return ils[1],ils[2]
        else
            @mpoc_assert false #no way to dermine order of links
        end
    elseif length(ils)==1
        n,c=parse_link(ils[1])
        if n==Nothing
            return Index(1),ils[1] #probably a qx link, don't know if row or col
        elseif nsite==1
            return Index(1),ils[1] #row vector
        else
            return ils[1],Index(1) #col vector
        end
    else
        @show ils
        @mpoc_assert false
    end
end
undef_int=-99999

function parse_site(W::ITensor)
    is=inds(W,tags="Site")[1]
    return parse_site(is)
end

function parse_site(is::Index)
    @mpoc_assert hastags(is,"Site")
    nsite=undef_int #sentenal value
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
    @mpoc_assert nsite>=0
    return dim(is),nsite,space
end

function parse_link(il::Index)::Tuple{Int64,Int64}
    @mpoc_assert hastags(il,"Link")
    nsite=ncell=undef_int #sentinel values
    for t in tags(il)
        ts=String(t)
        if ts[1:2]=="l=" || ts[1:2]=="n="
            nsite::Int64=tryparse(Int64,ts[3:end])
        end
        if ts[1:2]=="c="
            ncell::Int64=tryparse(Int64,ts[3:end])
        end
    end
    return nsite,ncell
end


#
# if one ot links is "Link,qx" then we don't get any site info from it.
# All this messy logic below tries to infer the site # of qx link from site index
# and link number of other index.
#
function infer_site_numbers(n1::Int64,n2::Int64,nsite::Int64)::Tuple{Int64,Int64}
    if n1==undef_int && n2==undef_int
        @mpoc_assert false
    end
    if n1==undef_int
        if n2==nsite
            n1=nsite-1 
        elseif n2==nsite-1
            n1=nsite 
        else
            @mpoc_assert false
        end
    end
    if n2==undef_int
        if n1==nsite
            n2=nsite-1 
        elseif n1==nsite-1
            n2=nsite 
        else
            @mpoc_assert false
        end
    end    
    return n1,n2
end

#
#  Handles direction and leaving out the last element in the sweep.
#
function sweep(H::AbstractMPS,lr::orth_type)::StepRange{Int64, Int64}
    N=length(H)
    return lr==left ? (1:1:N-1) : (N:-1:2)
end

#
#  Handles direction only.  For iMPOs we include the last site in the unit cell.
#
function sweep(H::AbstractInfiniteMPS,lr::orth_type)::StepRange{Int64, Int64}
    N=length(H)
    return lr==left ? (1:1:N) : (N:-1:1)
end

function linkind(M::AbstractInfiniteMPS, j::Integer)
    return commonind(M[j], M[j + 1])
end
  

#----------------------------------------------------------------------------
#
#  Detection of canonical (orthogonal) forms
#

@doc """
check_ortho(H,lr[,eps])::Bool

Test if all sites in an MPO statisfty the condition for `lr` orthogonal (canonical) form.

# Arguments
- `H:MPO` : MPO to be characterized.
- `lr::orth_type` : choose `left` or `right` orthogonality condition to test for.
- `eps::Float64 = 1e-14` : operators inside H with norm(W[i,j])<eps are assumed to be zero.

Returns `true` if the MPO is in `lr` orthogonal (canonical) form.  This is an expensive operation which scales as N*Dw^3 which should mostly be used only for unit testing code or in debug mode.  In production code use isortho which looks at the cached ortho state.

"""


@doc """
isortho(H,lr)::Bool

Test if anMPO is in `lr` orthogonal (canonical) form by checking the cached orthogonality center.

# Arguments
- `H:MPO` : MPO to be characterized.
- `lr::orth_type` : choose `left` or `right` orthogonality condition to test for.

Returns `true` if the MPO is in `lr` orthogonal (canonical) form.  This is a fast operation and should be safe to use in time critical production code.  The one exception is for iMPO with Ncell=1, where currently the ortho center does not distinguis between left/right or un-orthognal states.
"""
function isortho(H::AbstractMPS,lr::orth_type)::Bool 
    io=false
    # if length(H)==1
    #     io=check_ortho(H,lr) #Expensive!!
    # else
        if isortho(H)
            if lr==left
                io= orthocenter(H)==length(H)
            else
                io=orthocenter(H) == 1
            end
        end
    #end
    return  io
end

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
    zero_upper=true
    zero_lower=true
    dr=Base.max(0,dim(c)-dim(r))
    dc=Base.max(0,dim(r)-dim(c))
    for ir in eachval(r)
        zero_upper = zero_upper && norm(W[r=>ir:ir,c=>ir+dr+1:dim(c)])<eps
        zero_lower = zero_lower && norm(W[r=>ir:ir,c=>1:ir-1-dc])<eps
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
    r,c=parse_links(W)
    return detect_upper_lower(r,W,c,eps)
end

function is_upper_lower(W::ITensor,ul::reg_form,eps::Float64=default_eps)::Bool 
    l,u=detect_upper_lower(W,eps)
    ul==lower ? l : u
end

is_lower(W::ITensor,eps::Float64=default_eps)::Bool = is_upper_lower(W,lower,eps) 
is_upper(W::ITensor,eps::Float64=default_eps)::Bool = is_upper_lower(W,upper,eps) 
    

function detect_upper_lower(H::MPO,eps::Float64=default_eps)::Tuple{Bool,Bool}
    @mpoc_assert length(H)>1
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
function is_regular_form(H::AbstractMPS,ul::reg_form,eps::Float64=default_eps)::Bool
    il=dag(linkind(H,1))
    for n in 2:length(H)-1
        ir=linkind(H,n)
        #@show il ir inds(H[n])
        Wrf=reg_form_Op(H[n],il,ir,ul)
        !is_regular_form(Wrf,eps) && return false
        il=dag(ir)
    end
    return true
end

function detect_regular_form(H::AbstractMPS,eps::Float64=default_eps)::Tuple{Bool,Bool}
   return is_regular_form(H,lower,eps),is_regular_form(H,upper,eps)
end


function get_Dw(H::MPO)::Vector{Int64}
    N=length(H)
    Dws=Vector{Int64}(undef,N-1)
    for n in 1:N-1
        l=commonind(H[n],H[n+1])
        Dws[n]=dim(l)
    end
    return Dws
end

function get_Dw(H::InfiniteMPO)::Vector{Int64}
    N=length(H)
    Dws=Vector{Int64}(undef,N)
    for n in 1:N
        l=commonind(H[n],H[n+1])
        Dws[n]=dim(l)
    end
    return Dws
end
    
function get_traits(W::ITensor,eps::Float64)
    r,c=parse_links(W)
    d,n,space=parse_site(W)
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
    is__left=check_ortho(W,msl,eps)
    is_right=check_ortho(W,msr,eps)
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
    
