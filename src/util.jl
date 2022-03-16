
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
 
function pprint(W::ITensor,eps::Float64)
    d,n,r,c=parse_links(W)
    pprint(r,W,c,eps)
end

function pprint(r::Index,W::ITensor,c::Index,eps::Float64)
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

export pprint,is_unit,slice
