#
#  Handle MPOs with full matricies at the edges.  Support contraction of to
#  edges down to row and column vectors found in a standard MPO
#
mutable struct MPOpbc
    mpo::MPO
    l::ITensor #contract  leftmost site with this to get row vector op
    r::ITensor #contract rightmost site with this to get col vector op
    MPOpbc(mpo::MPO)=new(copy(mpo),get_lr_lower(mpo)...) 
end

function to_openbc(mpo::MPOpbc)::MPO
    N=length(mpo.mpo)
    #@show mpo.mpo[1]
    #@show mpo.l
    
    mpo.mpo[1]=mpo.l*mpo.mpo[1]
    mpo.mpo[N]=mpo.mpo[N]*mpo.r
    @assert length(filterinds(inds(mpo.mpo[1]),tags="Link"))==1
    @assert length(filterinds(inds(mpo.mpo[N]),tags="Link"))==1
    return mpo.mpo
end

function to_openbc(mpo::MPO)::MPO
    pbc=MPOpbc(mpo)
    return to_openbc(pbc)
end

function get_lr_lower(mpo::MPO)::Tuple{ITensor,ITensor}
    ul::tri_type = is_lower_regular_form(mpo,1e-14) ? lower : upper
 
    N=length(mpo)
    W1=mpo[1]
    llink=filterinds(inds(W1),tags="l=0")[1]
    l=ITensor(0.0,dag(llink))

    WN=mpo[N]
    rlink=filterinds(inds(WN),tags="l=$N")[1]
    r=ITensor(0.0,dag(rlink))
    if ul==lower
        l[llink=>dim(llink)]=1.0
        r[rlink=>1]=1.0
    else
        l[llink=>1]=1.0
        r[rlink=>dim(rlink)]=1.0
    end

    return l,r
end
