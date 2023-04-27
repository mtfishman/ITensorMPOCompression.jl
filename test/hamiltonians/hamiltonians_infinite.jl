import ITensorInfiniteMPS: InfiniteSum, InfiniteMPO
#
#  impo can be an Ncell==1 uniform iMPO.  But if the interaction extends out NNN neightbours
#  then the infinite sum needs to hold h=l*W^1*W^2...W^(NNN)*W^(NNN+1)*r with the standard capping
#  vectors l=(1,0...0) and r=(0,0...0,1) 
#  Unfortunately we cannot easily deduce NNN from the contents of impo.  Especially if It
#  is orthogonalized and or compressed.
#  
#
function InfiniteSum{MPO}(impo::InfiniteMPO, NNN::Int64)
  @mpoc_assert length(impo) == 1  #for now.
  N = NNN + 1
  mpo = MPO(N)
  for n in 1:N
    il, ir = parse_links(impo[n]) #get left and right links before we wipe out the cell numbers
    if n == 1
      il1 = removetags(il, "c=$(n-1)") #wipe out cell numbers
      il1 = replacetags(il1, "l=1", "l=$(n-1)")
    else
      l, il1 = parse_links(mpo[n - 1])
    end
    ir1 = removetags(ir, "c=$n")
    ir1 = replacetags(ir1, "l=1", "l=$n")
    ir2 = new_id(ir1)
    mpo[n] = replaceinds(impo[n], (il, ir), (il1, ir2))
  end
  #
  #  Cap the ends so there are no dangling links.
  #
  l, r = get_lr(mpo)
  mpo[1] = l * mpo[1]
  mpo[N] = mpo[N] * r
  return InfiniteSum{MPO}([mpo])
end

@doc """
    transIsing_iMPO(sites,NNN;kwargs...)
 
Infinite lattice of a transverse Ising model Hamiltonian with up to `NNN` neighbour 2-body 
interactions.  The interactions are hard coded to decay like `J/(i-j)`. between sites `i` and `j`.
One unit cell of the iMPO is stored, but `CelledVector` indexing allows code to access any unit cell.
     
# Arguments
- `sites` : Site set defining the lattice of sites.
- `NNN::Int64=1` : Number of neighbouring 2-body interactions to include in `H`
# Keywords
- `hx::Float64=0.0` : External magnetic field in `x` direction.
- `ul::reg_form=lower` : build H with `lower` or `upper` regular form.
- `J::Float64=1.0` : Nearest neighbour interaction strength.  Further neighbours decay like `J/(i-j)`..

"""
function transIsing_iMPO(sites, NNN::Int64; kwargs...)
  mpo = transIsing_MPO(sites, NNN; pbc=true, kwargs...)
  return InfiniteMPO(mpo.data)
end

# n is starting site number
function daisychain_links!(H::MPO, n::Int64)
  Ncell = length(H)
  H[1] = replacetags(H[1], "Link,l=$(n+1)", "Link,c=1,l=1")
  H[1] = replacetags(H[1], "Site,n=$(n+1)", "Site,c=1,n=1")
  nsave = n
  for i in 2:Ncell
    n += 1
    H[i] = replacetags(H[i], "Link,l=$n", "Link,c=1,l=$(i-1)")
    H[i] = replacetags(H[i], "Link,l=$(n+1)", "Link,c=1,l=$i")
    H[i] = replacetags(H[i], "Site,n=$(n+1)", "Site,c=1,n=$i")
  end
  #do the left link at the end, other wise Ncell=n+1 causes problems.
  H[1] = replacetags(H[1], "Link,l=$nsave", "Link,c=0,l=$Ncell")
  i0 = inds(H[1]; tags="Link,c=0,l=$Ncell")[1]
  i0 = replacetags(i0, "c=0", "c=1")
  iN = inds(H[Ncell]; tags="Link,c=1,l=$Ncell")[1]
  return H[Ncell] = replaceind(H[Ncell], iN, i0)
end

function transIsing_AutoiMPO(isites, NNN::Int64; kwargs...)
  return AutoiMPO(transIsing_AutoMPO, isites, NNN; kwargs...)
end

function Heisenberg_AutoiMPO(isites, NNN::Int64; kwargs...)
  return AutoiMPO(Heisenberg_AutoMPO, isites, NNN; kwargs...)
end
function Hubbard_AutoiMPO(isites, NNN::Int64; kwargs...)
  return AutoiMPO(Hubbard_AutoMPO, isites, NNN; kwargs...)
end

function AutoiMPO(MakeH::Function, isites, NNN::Int64; kwargs...)
  #
  # Make a finte lattice MPO large enough that there are no edge effects for Ncell sites in the center.
  #
  Ncell = length(isites) #unit cell size in the inf lattice.
  #
  # Choose a finite lattice size that should have Ncell+2 sites
  # in the middle with no edge effects.
  #
  N = 2 * (NNN + 2) + Ncell
  #
  #  Make a finite lattice of sites and corresponding MPO
  #
  ts = String(tags(isites[1])[1]) #get the site type.
  qns = hasqns(isites[1])
  fsites = siteinds(ts, N; conserve_qns=qns)
  for i in 1:Ncell
    fsites[i + NNN + 1] = isites[i]
  end
  fmpo = MakeH(fsites, NNN; pbc=true, kwargs...)
  #
  # Check that Dw is constant in the range we intend to use.
  #
  Dws = get_Dw(fmpo)
  for i in (NNN + 1):(NNN + Ncell + 2)
    @assert Dws[i] == Dws[i + 1]
  end
  #@show get_Dw(fmpo) NNN+2:NNN+2+Ncell
  #
  #  Creat and Ncell sized mpo from the middle chunk where Dw is constant.
  #
  impo = MPO(Ncell)
  for i in 1:Ncell
    impo[i] = fmpo[i + NNN + 1]
  end
  daisychain_links!(impo, NNN + 1)
  return InfiniteMPO(impo.data)
end

function new_id(i::Index)::Index
  if hasqns(i)
    return Index(space(i); dir=dir(i), tags=tags(i), plev=plev(i))
  else
    return Index(dim(i), tags(i); plev=plev(i))
  end
end
