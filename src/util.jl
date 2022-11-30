using Printf

function is_unit(O::ITensor,eps::Float64)::Bool
    s=inds(O)
    @ITensors.debug_check begin
        @assert(length(s)==2)
    end
    Id=delta(s[1],s[2])
    if hasqns(s)
        nm=0.0
        for b in eachnzblock(Id)
            isv=[s[i]=>b[i] for i in 1:length(b)]
            nm+=abs(O[isv...]-Id[isv...])
        end
        return nm<eps
    else
        return norm(O-Id)<eps
    end
end

function to_char(O::ITensor,eps::Float64)::Char
    c='0'
    if is_unit(O,eps)
        c='I'
    elseif norm(O)>eps
        c='S'
    end
    c
end
function to_char(O::Float64,eps::Float64)::Char
    c='0'
    if abs(O-1.0)<eps
        c='I'
    elseif abs(O)>eps
        c='S'
    end
    c
end

@doc """
pprint(W[,eps])

Show (pretty print) a schematic view of an operator-valued matrix.  In order to display any `ITensor`
as a matrix one must decide which index to use as the row index, and similarly for the column index. 
`pprint` does this by inspecting the indices with "Site" tags to get the site number, and uses the 
"Link" indices `l=\$n` as the column and `l=\$(n-1)` as the row. The output symbols I,0,S have the obvious 
interpretations and S just means any (non zero) operator that is not I.  This function can obviously 
be enhanced to recognize and display other symbols like X,Y,Z ... instead of just S.

# Arguments
- `W::ITensor` : Operator-valued matrix for display.
- `eps::Float64 = 1e-14` : operators inside W with norm(W[i,j])<eps are assumed to be zero.

# Examples
```julia
julia> pprint(W) 
I 0 0 0 0 
S 0 0 0 0 
S 0 0 0 0 
0 0 I 0 0 
0 S 0 S I 
```
"""
function pprint(W::ITensor,eps::Float64=default_eps)
    d,n,r,c=parse_links(W)
    pprint(r,W,c,eps)
end

function pprint(W::ITensor,r::Index,eps::Float64=default_eps)
    c=noncommoninds(W,r)
    @assert length(c)==1
    pprint(r,W,c[1],eps)
end

macro pprint(W)
    quote
    println($(string(W))," = ")
    pprint($(esc(W)))
    end
end

function pprint(r::Index,W::ITensor,c::Index,eps::Float64=default_eps)
    # @assert hasind(W,r) can't assume this becuse parse_links could return a Dw=1 dummy index.
    # @assert hasind(W,c)
    isl=filterinds(W,tags="Link")
    ord=order(W)
    if length(isl)==2
        for ir in  eachindval(r)
            for ic in  eachindval(c)
                if ord==4
                    Oij=slice(W,ir,ic)
                else
                    @assert ord==2
                    Oij=abs(W[ir,ic])
                end
                Base.print(to_char(Oij,eps))
                Base.print(" ")
            end
            Base.print("\n")
        end
    elseif length(isl)==1
        for i in  eachindval(isl[1])
        Oij=slice(W,i)
        Base.print(to_char(Oij,eps))
        Base.print(" ")
        end
        Base.print("\n")
    end
    Base.print("\n")
end

@doc """
    pprint(H[,eps])

Display the structure of an MPO, one line for each lattice site. For each site the table lists
- n   : lattice site number
- Dw1 : dimension of the row index (left link index)
- Dw2 : dimension of the column index (right link index)
- d   : dimension of the local Hilbert space
- Reg. Form  : U=upper, L=lower, D=diagonal, No = Not reg. form.
- Orth. Form : L=left, R=right, M=not orthogonal, B=both left and right.
- Tri. Form  : U=upper triangular, L=lower triangular, F=full (not triangular), D=diagonal, R=row, C=column

Be careful when reading the L symbols. L stands for Left in the Orth. column but is stands 
for Lower in the other two columns.

    # Arguments
- `H::MPO` : MPO for display.
- `eps::Float64 = 1e-14` : operators inside H with norm(W[i,j])<eps are assumed to be zero.

# Examples
```julia
julia> using ITensors
julia> using ITensorMPOCompression
julia> N=10; #10 sites
julia> NNN=7; #Include up to 7th nearest neighbour interactions
julia> sites = siteinds("S=1/2",N);
julia> H=make_transIsing_MPO(sites,NNN);
julia> pprint(H)
n    Dw1  Dw2   d   Reg.  Orth.  Tri.
                    Form  Form   Form
 1    1   30    2    L      M     R 
 2   30   30    2    L      M     L 
 3   30   30    2    L      M     L 
 4   30   30    2    L      M     L 
 5   30   30    2    L      M     L 
 6   30   30    2    L      M     L 
 7   30   30    2    L      M     L 
 8   30   30    2    L      M     L 
 9   30   30    2    L      M     L 
10   30    1    2    L      M     C 

julia> orthogonalize!(H,orth=right)
julia> pprint(H)
n    Dw1  Dw2   d   Reg.  Orth.  Tri.
                    Form  Form   Form
 1    1    3    2    L      M     R 
 2    3    4    2    L      R     L 
 3    4    5    2    L      R     L 
 4    5    6    2    L      R     L 
 5    6    7    2    L      R     L 
 6    7    6    2    L      R     L 
 7    6    5    2    L      R     L 
 8    5    4    2    L      R     L 
 9    4    3    2    L      R     L 
10    3    1    2    L      R     C 
julia> truncate!(H,orth=left)
julia> pprint(H)
n    Dw1  Dw2   d   Reg.  Orth.  Tri.
                    Form  Form   Form
 1    1    3    2    L      L     R 
 2    3    4    2    L      L     L 
 3    4    5    2    L      L     F 
 4    5    6    2    L      L     F 
 5    6    7    2    L      L     F 
 6    7    6    2    L      L     F 
 7    6    5    2    L      L     F 
 8    5    4    2    L      L     F 
 9    4    3    2    L      L     L 
10    3    1    2    L      M     C 
```
"""
function pprint(H::MPO,eps::Float64=default_eps)
    N=length(H)
    println("  n    Dw1  Dw2   d   Reg.  Orth.  Tri.")
    println("                      Form  Form   Form")
    
    for n in 1:N
        Dw1,Dw2,d,l,u,lr,lt,ut=get_traits(H[n],eps)
        #println(" $n    $Dw1    $Dw2    $d    $l$u     $lr   $lt$ut")
        @printf "%4i %4i %4i %4i    %s%s     %s     %s%s\n" n Dw1 Dw2 d l u lr lt ut
    end
end

#
#  In general it is useful to pass kwargs from a parent function to a child
#  in addition in the parent function we may want add or replace and existing key=>val parse_links
#  for some reason this seems to be a nasty uphill battle, especially if kwargs is empty.
#
function add_or_replace(kwargs,key::Any,val::Any)
    if length(kwargs)>0
        kwargs=Dict{Symbol, Any}(kwargs) #this allows us to set an element
        kwargs[key]=val
    else
        kwargs=Dict{Symbol, Any}(key => val)
    end
    return kwargs
end

export pprint,is_unit,slice
