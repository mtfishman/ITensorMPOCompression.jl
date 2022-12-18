
#
#  impo can be an Ncell==1 uniform iMPO.  But if the interaction extends out NNN neightbours
#  then the infinite sum needs to hold h=l*W^1*W^2...W^(NNN)*W^(NNN+1)*r with the standard capping
#  vectors l=(1,0...0) and r=(0,0...0,1) 
#  Unfortunately we cannot easily deduce NNN from the contents of impo.  Especially if It
#  is orthogonalized and or compressed.
#  
#
function InfiniteSum{MPO}(impo::InfiniteMPO,NNN::Int64)
    @assert length(impo)==1  #for now.
    N=NNN+1
    mpo=MPO(N)
    for n in 1:N
      il,ir=parse_links(impo[n]) #get left and right links before we wipe out the cell numbers
      if n==1
        il1=removetags(il,"c=$(n-1)") #wipe out cell numbers
        il1=replacetags(il1,"l=1","l=$(n-1)")
      else
        l,il1=parse_links(mpo[n-1])
      end
      ir1=removetags(ir,"c=$n")
      ir1=replacetags(ir1,"l=1","l=$n")
      ir2=new_id(ir1)
      mpo[n]=replaceinds(impo[n],(il,ir),(il1,ir2))
    end
    #
    #  Cap the ends so there are no dangling links.
    #
    l,r=get_lr(mpo)
    mpo[1]=l*mpo[1]
    mpo[N]=mpo[N]*r
    return  InfiniteSum{MPO}([mpo])
end
  

@doc """
    make_transIsing_iMPO(sites,NNN;kwargs...)
 
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
function make_transIsing_iMPO(sites,NNN::Int64;kwargs...)
    mpo=make_transIsing_MPO(sites,NNN;pbc=true,kwargs...)
    return InfiniteMPO(mpo.data)
end

function new_id(i::Index)::Index
    if hasqns(i)
        return Index(space(i);dir=dir(i),tags=tags(i),plev=plev(i))
    else
        return Index(dim(i),tags(i),plev=plev(i))
    end
end
