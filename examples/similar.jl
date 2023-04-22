using ITensors
using ITensors.NDTensors
using ITensorMPOCompression

A = randomITensor(Index(5, "i"), Index(4, "j"))
Aq = randomITensor(Index(QN(0) => 5; tags="i"), Index(QN(0) => 4; tags="j"))
Ad = diagITensor(Index(5, "i"), Index(5, "j"))
Aqd = diagITensor(Index(QN(0) => 5; tags="i"), Index(QN(0) => 4; tags="j"))

ib, jb = Index(4, "i"), Index(3, "j")
ibq, jbq = Index(QN(0) => 3; tags="i"), Index(QN(0) => 3; tags="j")
B = similar(A, ib, jb)
Bq = similar(Aq, ibq, jbq)
Bd = similar(Ad, Index(4, "i"), Index(4, "j"))
Bqd = similar(Aqd, ibq, jbq)
