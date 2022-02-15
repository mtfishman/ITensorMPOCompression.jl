using ITensors

function slice(A::ITensor,iv::IndexVal...)::ITensor
    iss=Tuple(noncommoninds(inds(A),map(first,iv)))
    Sl=ITensor(iss)
    for isv in eachindval(iss)
        Sl[isv...]=A[iv...,isv...]
    end
    return Sl
end

function is_unit(O::ITensor,eps::Float64)::Bool
    s=inds(O)
    @ITensors.debug_check begin
        @assert(length(s)==2)
    end
    norm(O-delta(s[1],s[2]))<eps
end

function to_char(O::ITensor,eps::Float64)::Char
    c='0'
    if is_unit(O,eps)
        c='I'
    elseif (norm(O))>eps
        c='S'
    end
    c
end
  
function pprint(W::ITensor,eps::Float64)
    isl=filterinds(W,tags="Link")
    if length(isl)==2
        for i in  eachindval(isl[1])
        for j in  eachindval(isl[2])
            Oij=slice(W,i,j)
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
