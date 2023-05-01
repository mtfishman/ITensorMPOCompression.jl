# ITensorMPOCompression
ITensorMPOCompression is a Julia language module based in the [ITensors](https://itensor.org/) library.  In general, compression of Hamiltonian MPOs must be treated differently than standard MPS compression.  The root of the problem can be traced to the combination of both extensive and intensive degrees of freedom in Hamiltonian operators.  If conventional compression methods are applied to Hamiltonian MPOs, one finds that two of the singular values will diverge as the lattice size increases, resulting in severe numerical instabilities.
The recent paper 
> *Local Matrix Product Operators:Canonical Form, Compression, and Control Theory* Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel **Phys. Rev. B** 102, 035147
contains a number of important insights for the handling of finite and infinite lattice MPOs. They show that the intensive degrees of freedom can be isolated from the extensive ones. If one only compresses the intensive portions of the Hamiltonian then the divergent singular values are removed from the problem.  This module attempts to implement the algorithms described in the *Parker et. al.* paper.  
The techincal details are presented in a document provided with this module: [TechnicalDetails.pdf](../TechnicalDetails.pdf). A brief summary of the key functions of this module follows.

## Block respecting QX decomposition
### Finite MPO
A finite lattice MPO with *N* sites can expressed as
```math
\hat{H}=\hat{W}^{1}\hat{W}^{2}\hat{W}^{3}\cdots\hat{W}^{N-1}\hat{W}^{N}
```
where each ``\\\hat{W}^{n}`` is an operator-valued matrix on site *n*
### Regular Forms
MPOs must be in the so called regular form in order for orthogonalization and compression to succeed. These forms are defined as follows:
```math
\hat{W}_{upper}=\begin{bmatrix}\hat{\mathbb{I}} & \hat{\boldsymbol{c}} & \hat{d}\\
0 & \hat{\boldsymbol{A}} & \hat{\boldsymbol{b}}\\
0 & 0 & \hat{\mathbb{I}}
\end{bmatrix}
```
```math
\hat{W}_{lower}=\begin{bmatrix}\hat{\mathbb{I}} & 0 & 0\\
\hat{\boldsymbol{b}} & \hat{\boldsymbol{A}} & 0\\
\hat{d} & \hat{\boldsymbol{c}} & \hat{\mathbb{I}}
\end{bmatrix}
```
Where:
``\\\hat{\boldsymbol{b}}\quad and\quad\hat{\boldsymbol{c}}`` are operator-valued vectors and
``\\\hat{\boldsymbol{A}}`` is an operator-valued matrix which is *not nessecarily triangular*. 


# Truncation (SVD compression)
Truncation (or compression) of an MPO starts with 2 orthogonalization sweeps using column pivoting, rank reducing QR decomposoitions.  These sweeps will significantly reduce the bond dimensions of the MPO.  A final sweep using SVD decomposition and compression of the internal degrees of freedom will then yield a bond spectrum at each bond site.  The users can control the rank reduction and SVD compression using parameters described below.
```@docs
ITensorMPOCompression.truncate!
```
# Orthogonalization (Canonical forms)
Most users will not need to deal directly with orthogonalization, as the truncation routine will do this automaticlly.  This is achieved by simply sweeping through the lattice and carrying out block respecting *QX* steps described in the technical notes.  For left canoncical form one starts at the left and sweeps right, and the converse applies for right canonical form.
```@docs
ITensorMPOCompression.orthogonalize!
```

# Characterizations

The module has a number of functions for viewing and characterization of MPOs and operator-valued matrices. 

## Regular forms

Regular forms are defined above in section [Regular Forms](@ref)

```@docs
ITensorMPOCompression.reg_form
detect_regular_form
is_regular_form
```

## Orthogonal forms

The definition of orthogonal forms for MPOs or operator-valued matrices are more complicated than those for MPSs.  First we must define the inner product of two operator-valued matrices. For the case of orthogonal columns this looks like:

```math
\sum_{a}\left\langle \hat{W}_{ab}^{\dagger},\hat{W}_{ac}\right\rangle =\frac{\sum_{a}^{}Tr\left[\hat{W}_{ab}\hat{W}_{ac}\right]}{Tr\left[\hat{\mathbb{I}}\right]}=\delta_{bc}
```
Where the summation limits depend on where the V-block is for `left`/`right` and `lower`/`upper`.  The specifics for all four cases are shown in table 6 in the [Technical Notes](../TechnicalDetails.pdf)

```@docs
ITensorMPOCompression.orth_type
ITensors.isortho
ITensorMPOCompression.check_ortho
```

# Some utility functions
One of the difficult aspects of working with operator-valued matrices is that they have four indices and if one just does a naive @show W to see what's in there, you see a voluminous output that is hard to read because of the default slicing selected by the @show overload. The pprint(W) (pretty print) function attempts to solve this problem by simply showing you where the zero, unit and other operators reside.
```@docs
pprint
```
# Test Hamiltonians
For development and demo purposes it is useful to have a quick way to make Hamiltonian
MPOs for various models.  Right now we have four Hamiltonians available
1. Direct transverse Ising model with arbitrary neighbour 2-body interactions
2. autoMPO transverse Ising model with arbitrary neighbour 2-body interactions
3. autoMPO Heisenberg model with arbitrary neighbour 2-body interactions
4. The 3-body model in eq. 34 of the Parker paper, built with autoMPO.
The autoMPO MPOs come pre-truncated so they are not as useful for testing truncation. The automated truncation in AutoMPO can **partially** disabled by providing the the keyword argument `cutoff=-1.0` which gets passed down in the svd/truncate call used when building the MPO

