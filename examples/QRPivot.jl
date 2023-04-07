using ITensors
using ITensorMPOCompression

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)


N=5 #5 sites
NNN=3 #Include 2nd nearest neighbour interactions
sites = siteinds("S=1/2",N);
H=make_transIsing_MPO(sites,NNN);
W,il,ir=H[2],linkind(H,1),linkind(H,2)
is=noncommoninds(W,il,ir)
@show il ir
d=dim(is[1])
Dwl,Dwr=dim(il),dim(ir)
c=W[il=>Dwl:Dwl,ir=>2:Dwr-1]
I=slice(W,il=>1,ir=>1)
@assert norm(c*I)<1e-15
Ac=W[il=>2:Dwl,ir=>2:Dwr-1]
ila,ira=inds(Ac,tags=tags(il)),inds(Ac,tags=tags(ir))
linds=noncommoninds(Ac,ira)
Q,R,iq=qr(Ac,linds;positive=true,rr_cutoff=1e-14,tags=tags(irp))
Q*=sqrt(d)
R/=sqrt(d)
Dwrp=dim(iq)+2
irp=redim(ir,Dwrp)
Wp=ITensor(0.0,il,irp,is)
Wp[il=>1:Dwl,irp=>1:1]=W[il=>1:Dwl,ir=>1:1]
Wp[il=>2:Dwl,irp=>2:Dwrp-1]=Q
Wp[il=>Dwl:Dwl,irp=>Dwrp:Dwrp]=W[il=>Dwl:Dwl,ir=>Dwr:Dwr]

iRr=noncommonind(R,iq)
R=prime(R,iRr)
iRpl,iRpr=dag(irp),prime(redim(iRr,dim(iRr)+2,1))
Rp=noprime(ITensorMPOCompression.grow(R,iRpl,ir'))

W1=Wp*Rp
@assert norm(W-W1)<1e-15
check_ortho(Wp,matrix_state(lower,left))

