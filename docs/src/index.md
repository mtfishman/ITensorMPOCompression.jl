# ITensorMPOCompression
ITensorMPOCompression is a Julia language module based in the [ITensors](https://itensor.org/) library.  In general, compression of Hamiltonian MPOs must be treated differently than standard MPS compression.  The root of the problem can be traced to the combination of both extensive and intensive degrees of freedom in Hamiltonian operators.  If conventional compression methods are applied of Hamiltonain MPOs, one finds that two of the singular values will diverge as the lattice size increases, resulting in severe numerical instabilities.
The recent paper 
> *Local Matrix Product Operators:Canonical Form, Compression, and Control Theory* Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel **Phys. Rev. B** 102, 035147
contains a number of important insights for the handling of finite and infinite lattice MPOs. They show that the intensive degrees of freedom can be isolated from the extensive ones. If one only compresses the intensive portions of the Hamiltonian then the divergent singular values are removed from the problem.  This module attempts to implement the algorithms described in the *Parker et. al.* paper.  The initial release will support orthogonalization (canonical form) and truncation (SVD compression) of the finite lattice MPOs.  A future release will add support for handling iMPOs, or infinite lattice MPOs with a repeating unit cell.
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

In order to handle all combinations of upper and lower regular forms in conjunction with left and right canonical (orthogonal) forms we must extend ITensors *QR* decomposition capabilities to also include *QL*, *RQ* and *LQ* decomposition.  Below and in the code these will be generically referred to as *QX* decomposition.
### The V-block
For lower regular form the V-blocks are:
```math
\hat{V}_{L}=\begin{bmatrix}\hat{\boldsymbol{A}}_{L} & 0\\
\hat{\boldsymbol{c}}_{L} & \hat{\mathbb{I}}
\end{bmatrix}
```
```math
\hat{V}_{R}=\begin{bmatrix}\hat{\mathbb{I}} & 0\\
\hat{\boldsymbol{b}}_{L} & \hat{\boldsymbol{A}}_{L}
\end{bmatrix}
```
where the `L` and `R` subscripts refer to `left` and `right` orthogonal forms being targeted.
### QX decomposition of the V-block
Block respecting *QL* decomposition is defined as *QL* decomposition of the of corresponding V-block:

`1.` Reshape 
```math 
\hat{V}_{ab}\rightarrow V_{ab}^{mn}\rightarrow V_{\left(mna\right)b}
```
`2.` *QL* decompose 
```math
V_{\left(mna\right)b}=\sum_{k=1}^{\chi+1}Q_{\left(mna\right)k}L_{kb}
```
`3.` Reshape 
```math 
Q_{\left(mna\right)k}\rightarrow\hat{Q}_{ak}
```
`4.` Stuff *Q* back into correct block of *W*.

`5.` Transfer *L* to the next site: 
```math
\hat{W}^{\left(i+1\right)}\rightarrow L\hat{W}^{\left(i+1\right)}
``` 
Fortunately, under the hood,  ITensor takes care of all the reshaping for us. After this process *W* will now be in left canonical form.  Other cases and some additional details are described in tech notes.
```@docs
block_qx
```
# Orthogonalization (Canonical forms)
This is achieved by simply sweeping through the lattice and carrying out block respecting *QX* steps described above.  For left canoncical form one starts at the left and sweeps right, and the converse applies for right canonical form.
```@docs
ITensorMPOCompression.orthogonalize!
```
# Truncation (SVD compression)
Prior to truncation the MPO must first be rendered into canoncial form using the orthogonalize! function described above.  If for example the MPO is right-lower canonical form then a truncation sweep starts by doing a block repsecting *QL* decomposition on site 1:
```math
\hat{W}^{1}\rightarrow\hat{Q}^{1}L^{1}
```
We now must further factor *L* as follows
```math
L=\begin{bmatrix}1 & 0 & 0\\
0 & \mathsf{L} & 0\\
0 & t & 1
\end{bmatrix}=ML^{\prime}=\begin{bmatrix}1 & 0 & 0\\
0 & \mathsf{M} & 0\\
0 & 0 & 1
\end{bmatrix}\begin{bmatrix}1 & 0 & 0\\
0 & \mathsf{\mathbb{I}} & 0\\
0 & t & 1
\end{bmatrix}
```
where internal sans-M matrix is what gets decomposed with SVD.  Picking out this internal matrix is really the secret sauce for avoiding diverging singular values when the lattice size grows.  After decomposition `M=UsV` the `U` matrix is absorbed to the left such that `W=QU` which is left canoncial. *sV* gets combined with L' and transfered to next site to the right.  There are many details to consider and these are explained in the technical notes.

```@docs
truncate!
```

# Characterizations

The module has a number of functions for characterization of MPOs and operator-valued matrices. Some
points to keep in mind:
1. An MPO must be in one of the regular forms in order for orthogonalization to work.
2. An MPO must be in one ot the orthonormal (canonical) forms prior to SVD truncation (compression). However the truncate! function will detect this and do the orthogonalization if needed.
3. Lower (upper) regular form does not mean that the MPO is lower (upper) triangular.  As explained in the [Technical Notes](../TechnicalDetails.pdf) the `A`-block does not need to be triangular.
4. Having said all that, most common hand constructed MPOs are either lower or upper triangular. Lower happens to be the more common convention.
5. The orthogonalize! operation just happens to preserve lower (upper) trianglur form.  However truncation (SVD) does not preserve triangular form.
6. At the moment autoMPO outputs (after a simple 2<-->Dw row & col swap [`fix_autoMPO!`](@ref)) lower *regular* form, but not lower *triangular* form (the `A` block is not triangular).

## Regular forms

Regular forms are defined above in section [Regular Forms](@ref)

```@docs
reg_form
detect_regular_form
is_regular_form
is_lower_regular_form
is_upper_regular_form
```

## Orthogonal forms

The definition of orthogonal forms for MPOs or operator-valued matrices are more complicated than those for MPSs.  First we must define the inner product of two operator-valued matrices. For the case of orthogonal columns this looks like:

```math
\sum_{a}\left\langle \hat{W}_{ab}^{\dagger},\hat{W}_{ac}\right\rangle =\frac{\sum_{a}^{}Tr\left[\hat{W}_{ab}\hat{W}_{ac}\right]}{Tr\left[\hat{\mathbb{I}}\right]}=\delta_{bc}
```
Where the summation limits depend on where the V-block is for `left`/`right` and `lower`/`upper`.  The specifics all four cases are shown in table 6 in the [Technical Notes](../TechnicalDetails.pdf)

```@docs
orth_type
is_orthogonal
```

# Some utility functions
One of the difficult aspects of working with operator-valued matrices is that they have four indices and if one just does a naive @show W to see what's in there, you see a voluminous output that is hard to read because of the default slicing selected by the @show overload. The pprint(W) (pretty print) function attempts to solve this problem by simply showing you where the zero, unit and other operators reside.
```@docs
pprint
```
# Test Hamiltonians
For development and demo purposes it is useful to have a quick way to make Hamiltonian
MPOs for various models.  Right now we have three Hamiltonians available
1. Direct transverse Ising model with arbitrary neighbour interactions
2. autoMPO transverse Ising model with arbitrary neighbour interactions
3. autoMPO Heisenberg model with arbitrary neighbour interactions
The autoMPO MPOs come pre-truncated so they are not useful for testing truncation.

```@docs
make_transIsing_MPO
make_transIsing_AutoMPO
make_Heisenberg_AutoMPO
fix_autoMPO!(::MPO)
```