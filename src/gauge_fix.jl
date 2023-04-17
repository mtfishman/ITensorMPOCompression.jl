
#
#  Find the first dim==1 index and remove it, then return a Vector.
#
function vector_o2(T::ITensor)
    @assert order(T)==2
    i1=inds(T)[findfirst(d->d==1,dims(T))]
    return vector(T*dag(onehot(i1=>1)))
end

function is_gauge_fixed(W::reg_form_Op,eps::Float64;b=true,c=true)::Bool where {T}
    igf=true
    Wb=extract_blocks(W,left;c=true,b=true)
    if b && dim(W.ileft)>1
        igf=igf && norm(b0(Wb))<eps
    end
    if c && dim(W.iright)>1
        igf=igf && norm(c0(Wb))<eps
    end
    return igf
end

function is_gauge_fixed(Hrf::reg_form_MPO,eps::Float64;kwargs...)::Bool where {T}
    for W in Hrf
        !is_gauge_fixed(W,eps;kwargs...) && return false
    end
    return true
end

function gauge_fix!(W::reg_form_Op,tₙ₋₁::Vector{Float64},lr::orth_type)
    @assert is_regular_form(W)
    Wb=extract_blocks(W,lr;all=true,fix_inds=true)
    𝕀,𝑨,𝒃,𝒄,𝒅=Wb.𝕀,Wb.𝑨,Wb.𝒃,Wb.𝒄,Wb.𝒅 #for readability below.
    nr,nc=dim(W.ileft),dim(W.iright)
    nb,nf = lr==left ? (nr,nc) : (nc,nr)
    #
    #  Make in ITensor with suitable indices from the 𝒕ₙ₋₁ vector.
    #
    if nb>1
        ibd,ibb = llur(matrix_state(W.ul,lr)) ?  (Wb.ird, Wb.irb) : (Wb.icd, Wb.icb)
        𝒕ₙ₋₁=ITensor(tₙ₋₁,dag(ibb),ibd)
    end
    𝒄⎖=nothing
    #
    #  First two if blocks are special handling for row and column vector at the edges of the MPO
    #
    if nb==1 #col/row at start of sweep.
        𝒕ₙ=c0(Wb) 
        𝒄⎖=𝒄-𝕀*𝒕ₙ
        𝒅⎖=𝒅
    elseif nf==1 ##col/row at the end of the sweep
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝒃
        𝒕ₙ=ITensor(1.0,Index(1),Index(1)) #Not used, but required for the return statement.
    else
        𝒕ₙ=𝒕ₙ₋₁*A0(Wb)+c0(Wb)
        𝒄⎖=𝒄+𝒕ₙ₋₁*𝑨-𝒕ₙ*𝕀
        𝒅⎖=𝒅+𝒕ₙ₋₁*𝒃
    end
    
    set_𝒅_block!(W,𝒅⎖)
    @assert is_regular_form(W)

    if !isnothing(𝒄⎖)
        if llur(matrix_state(W.ul,lr))
            set_𝒄_block!(W,𝒄⎖)
        else
            set_𝒃_block!(W,𝒄⎖)
        end
    end
    @assert is_regular_form(W)

    # 𝒕ₙ is always a 1xN tensor so we need to remove that dim==1 index in order for vector(𝒕ₙ) to work.
    return vector_o2(𝒕ₙ)
end

function gauge_fix!(H::reg_form_MPO) where {T}
    tₙ=Vector{Float64}(undef,1)
    for W in H
        tₙ=gauge_fix!(W,tₙ,left)
        @assert is_regular_form(W)
    end
    #tₙ=Vector{Float64}(undef,1) end of sweep above already returns this.
    for W in reverse(H)
        tₙ=gauge_fix!(W,tₙ,right)
        @assert is_regular_form(W)
    end
end