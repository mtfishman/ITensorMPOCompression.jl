using ITensors, ITensorMPS, ITensorMPOCompression

import ITensorMPOCompression: assign!, slice

N=2
s = siteinds("S=1/2", N;conserve_szparity=true)
qns=qn.(space(s[1]))
l = Index([qns[1]=>2,qns[2]=>2],"l")
Id= denseblocks(δ(dag(l), prime(l)))
il= Index([QN()=>1,QN()=>1,qns[1]=>1,qns[2]=>1,QN()=>1],"Link")
L₁=ITensor(0.0,l,dag(l'),il)
assign!(L₁,Id,il=>5)
# Idl=Id*onehot(Float64,il=>1)
# L₁[il=>5:5]=Idl
display(array(slice(L₁,il=>5)))

