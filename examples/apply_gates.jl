# Define the nearest neighbor term `S⋅S` for the Heisenberg model
using ITensors
using ITensorMPOCompression
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

function ITensors.op(::OpName"expS⋅S", ::SiteType"S=1/2",
    s1::Index, s2::Index; τ::Number)
    O = 0.5 * op("S+", s1) * op("S-", s2) +
        0.5 * op("S-", s1) * op("S+", s2) +
              op("Sz", s1) * op("Sz", s2)
    return exp(τ * O)
end

function fix_link_tags!(W::ITensor)
    is=filterinds(inds(W),tags="Site")[1]
    ts2=String(tags(is)[3])
    @mpoc_assert ts2[1:2]=="n="
    nsite::Int64=tryparse(Int64,ts2[3:end])
    ils=filterinds(inds(W),"Link")
    n1::Int64=tryparse(Int64,String(tags(ils[1])[2])[3:end])
    if length(ils)>1
        n2::Int64=tryparse(Int64,String(tags(ils[2])[2])[3:end])
    else
        n2=0
    end
    #@show nsite n1 n2
    for i in 1:length(ils)
        tn=String(tags(ils[i])[2])
        if tn[1]=='n'
            if i==1
                if n2==nsite
                    tl="l=$(nsite-1)"
                end
                if n2==nsite-1
                    tl="l=$nsite"
                end
                if n2==0
                    if n1==nsite
                        tl="l=$nsite"
                    else
                        tl="l=$(nsite-1)"
                    end
                end          
            end
            if i==2
                if n1==nsite
                    tl="l=$(nsite-1)"
                end
                if n1==nsite-1
                    tl="l=$nsite"
                end
               
            end
            replacetags!(W,tn,tl,tags="Link")
        end
    end
end
function fix_link_tags!(H::MPO)
    for W in H
        fix_link_tags!(W)
    end
end

N=10
τ = -0.01im
even = [("expS⋅S", (n, n+1), (τ = τ,)) for n in 1:2:N-1]
odd  = [("expS⋅S", (n, n+1), (τ = τ,)) for n in 2:2:N-1]
s = siteinds("S=1/2", N)
even_expτH = ops(even, s)
odd__expτH = ops(odd , s)

H=make_Heisenberg_AutoMPO(s,N)
orthogonalize!(H,orth=right)
pprint(H)
He = apply(even_expτH, H)
fix_link_tags!(He)
orthogonalize!(He,orth=right)
get_Dw(He)
pprint(He)
# W=He[1]
# pprint(W)
# d,n,r,c,lnr,lnc=parse_links(W)
# for j in 1:dim(c)
#     @show slice(W,c=>j)
# end
# # Ho = apply(odd__expτH, He)
# # pprint(Ho)
# Nothing
