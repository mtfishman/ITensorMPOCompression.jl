using ITensors
i = Index(5)
A = ITensor(ones(5, 5), i, i')
Q, R = qr(A, i)
@show dims(Q) dims(R) norm(A - Q * R)
Q, R = qr(A, i; rr_cutoff=1e-15)
@show dims(Q) dims(R) norm(A - Q * R)
