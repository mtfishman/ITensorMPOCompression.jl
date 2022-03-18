
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
as a matrix one must decide which index to use as the row index, and simiarly for the column index. 
`pprint` does this by inspecting the indices with "Site" tags to get the site number, and uses the 
"Link" indices `l=\$n` as the column and `\$(n-1)` as the row. The output symbols I,0,S have the obvious 
interpretations and S just means any (non zero) operator that is not I.  This function can obviously 
be enchanced to recognize and display other symbols like X,Y,Z ... instead of just S.

# Arguments
- `W::ITensor` : Operator-valued matrix for display.
- `eps::Float64 = 1e-14` : operators inside W with norm(W[i,j])<eps are assumed to be zero.

# Examples
```julia
julia>pprint(W) 
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

function pprint(r::Index,W::ITensor,c::Index,eps::Float64=default_eps)
    @assert hasind(W,r)
    @assert hasind(W,c)
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
