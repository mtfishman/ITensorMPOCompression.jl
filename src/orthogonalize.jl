#-----------------------------------------------------------------
#
#   Ac block respecting orthogonalization.
#
@doc """
    orthogonalize!(H::MPO)

Bring an MPO into left or right canonical form using block respecting QR decomposition
 as described in:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=`1.0 indicates no rank reduction.

# Examples
```julia
julia> using ITensors
julia> using ITensorMPOCompression
julia> N=10; #10 sites
julia> NNN=7; #Include up to 7th nearest neighbour interactions
julia> sites = siteinds("S=1/2",N);
#
# This makes H directly, bypassing autoMPO.  (AutoMPO is too smart for this
# demo, it makes maximally reduced MPOs right out of the box!)
#
julia> H=make_transIsing_MPO(sites,NNN);
#
#  Make sure we have a regular form or orhtogonalize! won't work.
#
julia> is_lower_regular_form(H)==true
true
#
#  Let's see what the second site for the MPO looks like.
#  I = unit operator, and S = any other operator
#
julia> pprint(H[2])
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
.
.
.
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 I 0 0 
0 S 0 S 0 0 S 0 0 0 S 0 0 0 0 S 0 0 0 0 0 S 0 0 0 0 0 0 S I 
#
#  Now we can orthogonalize or bring it into canonical form.
#  Defaults are left orthogonal with rank reduction.
#
julia> orthogonalize!(H;rr_cutoff=1e-14)
#
#  Wahoo .. rank reduction knocked the size of H way down, and we haven't
#  tried compressing yet!
#
julia> pprint(H[2])
I 0 0 0 
S I 0 0 
0 0 S I 
#
#  What do all the bond dimensions of H look like?  We will need compression 
#  (truncation) in order to further bang down the size of H
#
julia> get_Dw(H)
9-element Vector{Int64}: 3 4 5 6 7 6 5 4 3
#
#  wrap up with two more checks on the structure of H
#
julia> is_lower_regular_form(H)==true
true
julia> isortho(H,left)==true
true


```

"""
function ac_orthogonalize!(H::reg_form_MPO,lr::orth_type;eps::Float64=1e-14,kwargs...) 
    if !is_gauge_fixed(H,eps)
        gauge_fix!(H)
    end
    rng=sweep(H,lr)
    for n in rng
        nn=n+rng.step
        H[n],R,iqp=ac_qx(H[n],lr)
        H[nn].W=R*H[nn].W
        H[nn][lr]=dag(iqp)
        check(H[n])
        check(H[nn])
    end
    H.rlim = rng.stop+rng.step+1
    H.llim = rng.stop+rng.step-1
end

function ac_orthogonalize!(H::MPO,lr::orth_type;kwargs...) 
    Hrf=reg_form_MPO(H)
    ac_orthogonalize!(Hrf,lr;kwargs)
    return MPO(Hrf)
end

#--------------------------------------------------------------------------------------------
#
#  Functions for bringing an iMPO into left or right canonical form
#

#
#  Outer routine simply established upper or lower regular forms
#
@doc """
    orthogonalize!(H::InfiniteMPO;kwargs...)

Bring `CelledVector` representation of an infinite MPO into left or right canonical form using 
block respecting QR iteration as described in section Vi B and Alogrithm 3 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
If you intend to also call `truncate!` then do not bother calling `orthogonalize!` beforehand, as `truncate!` will do this automatically and ensure the correct handling of that gauge transforms.

# Arguments
- H::InfiniteMPO which is `CelledVector` of MPO matrices. `CelledVector` and `InfiniteMPO` are defined in the `ITensorInfiniteMPS` module.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=-11.0 indicate no rank reduction.

# Returns
- Vector{ITensor} with the gauge transforms between the input and output iMPOs

# Examples
```julia
julia> using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
julia> initstate(n) = "↑";
julia> sites = infsiteinds("S=1/2", 1;initstate, conserve_szparity=false)
1-element CelledVector{Index{Int64}, typeof(translatecelltags)}:
 (dim=2|id=326|"S=1/2,Site,c=1,n=1")
#
# This makes H directly, bypassing autoMPO.  (AutoMPO is too smart for this
# demo, it makes maximally reduced MPOs right out of the box!)
#
julia> H=make_transIsing_MPO(sites,NNN);
julia> get_Dw(H)
1-element Vector{Int64}:
 17
julia> orthogonalize!(H;orth=right,rr_cutoff=1e-15);
julia> get_Dw(H)
1-element Vector{Int64}:
 14
julia> orthogonalize!(H;orth=left,rr_cutoff=1e-15);
julia> get_Dw(H)
 1-element Vector{Int64}:
  13
julia> isortho(H,left)
true


```

"""
function ac_orthogonalize!(H::reg_form_iMPO,lr::orth_type;verbose=false,kwargs...)
    if !is_gauge_fixed(H,1e-14)
        gauge_fix!(H)
    end
    N=length(H)
    #
    #  Init gauge transform with unit matrices.
    #
    Gs=CelledVector{ITensor}(undef,N)
    for n in 1:N
        ln=lr==left ? H[n].iright : dag(H[n].iright) #get the forward link index
        Gs[n]=δ(Float64,dag(ln),ln') 
    end
    RLs=CelledVector{ITensor}(undef,N)
    
    eps=1e-13
    niter=0
    max_iter=40
    
    if verbose
        previous_Dw=Base.max(get_Dw(H)...)
        @printf "niter  Dw  eta\n" 
    end
    loop=true
    rng=sweep(H,lr)
    while loop
        eta=0.0
        for n in rng
            H[n],RLs[n],etan=ac_qx_step!(H[n],lr,eps;kwargs...)
            if lr==left
                Gs[n]=noprime(RLs[n]*Gs[n])  #  Update the accumulated gauge transform
            else
                Gs[n-1]=noprime(RLs[n]*Gs[n-1])  #  Update the accumulated gauge transform
            end
            @mpoc_assert order(Gs[n])==2 #This will fail if the indices somehow got messed up.
            eta=Base.max(eta,etan)
        end
        #
        #  H now contains all the Qs.  We need to transfer the RL's
        #
        for n in rng
            R=noprime(RLs[n-rng.step])
            ic=commonind(R,H[n].W)
            il=noncommonind(R,ic)
            RH=R*H[n].W
            if lr==left
                H[n]=reg_form_Op(RH,noprime(il),H[n].iright,H[n].ul)
            else
                H[n]=reg_form_Op(RH,H[n].ileft,noprime(il),H[n].ul)
            end
            check(H[n])
        end
        niter+=1
        if verbose
            @printf "%4i %4i %1.1e\n" niter Base.max(get_Dw(H)...) eta
        end
        loop=eta>1e-13 && niter<max_iter
    end
    H.rlim = rng.stop+1
    H.llim = rng.stop-1

    # if verbose
    #     Dw=Base.max(get_Dw(H)...)
    #     println("   iMPO After $lr orth sweep, $niter iterations Dw reduced from $previous_Dw to $Dw")
    # end
    
    return Gs
end

function ac_qx_step!(W::reg_form_Op,lr::orth_type,eps::Float64;kwargs...)
    Q,R,iq,p=ac_qx(W,lr;cutoff=1e-14,qprime=true,kwargs...) # r-Q-qx qx-RL-c
    #
    #  How far are we from RL==Id ?
    #
    if dim(forward(W,lr))==dim(iq)
        eta = RmI(R,p) #Different handling for dense and block-sparse
    else
        eta=99.0 #Rank reduction occured so keep going.
    end
    return Q,R,eta
end

#
#  Evaluate norm(R-I)
#
RmI(R::ITensor,perms)=RmI(tensor(R),perms)
function RmI(R::DenseTensor,perm::Vector{Int64})
    Rmp=matrix(R)[:,perm]
    return norm(Rmp-Matrix(LinearAlgebra.I,size(Rmp)))
end

function RmI(R::BlockSparseTensor,perms::Vector{Vector{Int64}})
    @assert nnzblocks(R)==length(perms)
    eta2=0.0
    for (n,b) in enumerate(nzblocks(R))
        bv=ITensors.blockview(R,b)
        Rp=bv[:,perms[n]] #un-permute the columns
        eta2+=norm(Rp-Matrix(LinearAlgebra.I,size(Rp)))^2
    end
    return sqrt(eta2) 
end
