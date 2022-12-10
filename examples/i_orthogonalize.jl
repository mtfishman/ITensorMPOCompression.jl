using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Printf
using Test


function get_site_number(W::ITensor)::Tuple{Int64,Int64}
    is=filterinds(inds(W),tags="Site")[1]
    nsite,space=parse_site(is)
    d=dim(is)
    return d,nsite
end

function iparse_link(il::Index)
    @assert hastags(il,"Link")
    nsite=ncell=-99999 #sentinel values
    for t in tags(il)
        ts=String(t)
        if ts[1:2]=="l="
            nsite::Int64=tryparse(Int64,ts[3:end])
        end
        if ts[1:2]=="c="
            ncell::Int64=tryparse(Int64,ts[3:end])
        end
    end
    @assert nsite!=-99999
    @assert ncell!=-99999
    return nsite,ncell
end
function iparse_links(W::ITensor,N::Int64)::Tuple{Index,Index}
    ils=filterinds(W,tags="Link")
    @assert length(ils)==2
    n1,c1=iparse_link(ils[1])
    n2,c2=iparse_link(ils[2])
    nc1=c1*N+n1
    nc2=c2*N+n2
    @assert nc1!=nc2
    if nc1<nc2
        return ils[1],ils[2]
    else
        return ils[2],ils[1]
    end
end

# function set_tags!(H::InfiniteMPO)
#     N=length(H)
#     d,nsite=get_site_number(H[1])
#     dn=nsite-1
#     for n in 1:N
#         nl = dn + (n==1 ? N+n-1 : n-1)
#         nr = dn + n
#         H[n]=addtags(H[n],"left" ,tags="l=$nl")
#         H[n]=addtags(H[n],"right",tags="l=$nr")
#     end
# end
#
#  Functions for bringing an iMPO into left or right canonical form
#
function qx_step!(W::ITensor,G::ITensor,n::Int64,r::Index,c::Index,ul::reg_form,eps::Float64;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    forward = lr==left ? c : r
    #@show lr n r c forward
    #d,n,r1,c1=iparse_links(W) #tag(r)= "l=$(n-1)", tag(c)="l=$n"
    #@show n,r,c
    #@show inds(W)
    Q,RL,iq=block_qx(W,n,r,c,ul;epsrr=1e-12,kwargs...) # r-Q-qx qx-RL-c
   
    if dim(c)==dim(iq)
        eta=norm(RL-δ(Float64, iq,forward))
    else
        eta=99.0
    end
    #@show tags(c)
    #tln=String(tags(c)[2])
    ilnp=prime(settags(iq,tags(forward))) #"qx" -> "l=$n" prime

    #@show iq ilnp
    #@show iq ilnp inds(RL) inds(G)
    replaceind!(RL,iq,ilnp)
    replaceind!(Q ,iq,ilnp)
   
    #@show inds(RL) inds(G)
    G=RL*G
    #@show inds(G)
    @assert order(G)==2
    G=noprime(G)
    #G=addtags(G,"right" ,tags=tln)
    #@show n inds(Q) inds(RL) inds(G)
    return Q,RL,G,eta
end


#
#  Loop throught the sites in correct direction
#
function qx_iterate!(H::InfiniteMPO,ul::reg_form;kwargs...)
    lr::orth_type=get(kwargs, :orth, left)
    quiet::Bool=get(kwargs, :quiet, true)
    N=length(H)
    #
    #  Init gauge transform with unit matrices.
    #
    Gs=CelledVector{ITensor}(undef,N)
    for n in 1:N
        il,ir=iparse_links(H[n],N) #get left and right indices
        if lr==left
            Gs[n]=δ(Float64,ir,ir') 
        else
            Gs[n]=δ(Float64,il,il')
        end
    end
    RLs=CelledVector{ITensor}(undef,N)
    
    
    eps=1e-13
    niter=0
    max_iter=40
    if !quiet
        @printf "niter eta\n" 
    end
    loop=true
    rng=sweep(H,lr)
    while loop
        eta=0.0
        for n in rng
            il,ir=iparse_links(H[n],N) #get left and right indices
            H[n],RLs[n],Gs[n],etan=qx_step!(H[n],Gs[n],n,il,ir,ul,eps;kwargs...)
            eta=Base.max(eta,etan)
        end
        #
        #  H now contains all the Qs.  We need to transfer the RL's
        #  Direction?
        #
        for n in rng
            #@show inds(H[n],tags="Link") inds(RLs[n-1],tags="Link")
            H[n]=RLs[n-rng.step]*H[n] #W(n)=RL(n-1)*Q(n)
            #@show inds(H[n],tags="Link")
            @assert order(H[n])==4
            H[n]=noprime(H[n],tags="Link")
        end
        niter+=1
        loop=eta>1e-13 && niter<max_iter
        #if loop set_tags!(H) end #add back left/right markers
        if eta<1.0 && !quiet
            @printf "%4i %1.1e\n" niter eta
        end
    end
    return Gs
end

#
#  Next level down we select a algorithm
#
function i_orthogonalize!(H::InfiniteMPO,ul::reg_form;kwargs...)
    return qx_iterate!(H,ul;kwargs...)
end

#
#  Out routine simply established upper or lower regular forms
#
function i_orthogonalize!(H::MPO;kwargs...)
    @assert has_pbc(H)
    (bl,bu)=detect_regular_form(H,1e-14)
    if !(bl || bu)
        throw(ErrorException("orthogonalize!(H::MPO), H must be in either lower or upper regular form"))
    end
    @assert !(bl && bu)
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    return orthogonalize!(H,ul;kwargs...)
end

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f)



@testset "Orthogonalize InfiniteMPO 2-body Hamiltonians" begin
    initstate(n) = "↑"
    ul=lower
    for N in 1:4, NNN in [2,4,8]
        si = infsiteinds("S=1/2", N; initstate, conserve_szparity=false)
        H=make_transIsing_MPO(si,NNN,0.0,ul,1.0;pbc=true)
        @test is_regular_form(H,ul)
        H0=InfiniteMPO(H.data)
        HL=copy(H0)
        @test is_regular_form(HL,ul)
        GL=i_orthogonalize!(HL,lower;orth=left)
        @test is_regular_form(H,ul)
        @test is_orthogonal(HL,left)
        for n in 1:N
            @test norm(HL[n]*GL[n]-GL[n-1]*H0[n]) ≈ 0.0 atol = 1e-14 
        end
        HR=copy(H0)
        GR=i_orthogonalize!(HR,lower;orth=right)
        @test is_regular_form(HR,ul)
        @test is_orthogonal(HR,right)
        for n in 1:N
            @test norm(GR[n]*HR[n]-H0[n]*GR[n+1]) ≈ 0.0 atol = 1e-14
        end   
        HR1=copy(HL) 
        G=i_orthogonalize!(HR1,lower;orth=right)
        @test is_regular_form(HR1,ul)
        @test is_orthogonal(HR1,right)
        for n in 1:N
            @test norm(G[n]*HR1[n]-HL[n]*G[n+1]) ≈ 0.0 atol = 1e-14
        end   
    end
    # @pprint(Hi[1])
    # @pprint(Hi[2])
    # @pprint(H[2])
end

nothing

