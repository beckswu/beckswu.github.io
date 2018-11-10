---
layout:     post
title:      "Linear Algebra + PCA - Summary "
subtitle:   "Linear Algebra 线性代数 总结"
date:       2018-09-23 20:00:00
author:     "Becks"
header-img: "img/post/math.jpg"
catalog:    true
tags:
    - Linear Algebra
    - 学习笔记
    - PCA
    - 总结
---

> note from youtube khan Academy / Introduction to Linear Algebra \[Gilbert Strang\]

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


## Vector

Communicative: &nbsp;&nbsp;&nbsp;  $$ \vec v \cdot \vec w  =  \vec w \cdot \vec v $$ <br/>
Distributive: &nbsp;&nbsp;&nbsp;  $$ \left(\vec w + \vec v \right)\cdot \vec x  =  \vec w \cdot \vec x + \vec v \cdot \vec x  $$<br/>
Associative over scaler multiplication: &nbsp;&nbsp;&nbsp; $$ \left( c \vec v \right) \cdot \vec w  =  c \left( \vec v \cdot \vec w \right)  $$<br/>
Dot product self is the length square: &nbsp;&nbsp;&nbsp; $$ \vec v \cdot \vec v  =  ||v||^2 = v_1^2 + v_2^2 + ... + v_n^2  $$<br/>
Cosine: &nbsp;&nbsp;&nbsp; $$  \vec a \cdot \vec b  =  ||a|| ||b|| cos\theta  $$<br/>
Sine: &nbsp;&nbsp;&nbsp; $$  \vec a \times \vec b  =  ||a|| ||b|| sin\theta  $$<br/>
Cauchy Schwarz Inequality: &nbsp;&nbsp;&nbsp; $$  \| \vec a \cdot \vec b \|  <=  ||a||||b||  $$<br/>
scaler projection: &nbsp;&nbsp;&nbsp; $$  \vec proj_{L} \left(\vec x\right) =   \frac{ \vec x \cdot \vec v  }{ ||\vec v|| } $$<br/>
vector projection: &nbsp;&nbsp;&nbsp; $$  \vec proj_{L} \left(\vec x\right) =  c \vec v =  \frac{ \vec x \cdot \vec v  }{ \vec v \cdot \vec v } \vec v =  \frac{ \vec v \vec v^T  }{ \| \vec v \|^2 } \vec x =  $$<br/>
two vector $$  \vec  x $$ and $$  \vec y $$ are orthorgonal: &nbsp;&nbsp;&nbsp; $$  \vec  x \cdot \vec y  = 0 $$   <br/>
(Optional) : $$  \vec a \times \left( \vec b \times \vec c \right)  =  \vec b \left(\vec a \cdot \vec c \right) - \vec c \left(\vec a \cdot \vec b \right)  $$

cross product: <br/>
 $$  \vec a =  \begin{bmatrix} a_1 \\ a_2  \\ a_3 \\ \end{bmatrix}, \vec b =  \begin{bmatrix} b_1 \\ b_2  \\ b_3 \\ \end{bmatrix}, \vec a \times \vec b  = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \\ \end{bmatrix} $$<br/>
Cross product 还可以算平行四边形的面积

<span style="background-color: #FFFF00">Dot product tells: product of lengths of vectors move together at same direction with b.</span> When $$\vec a $$ ∙ $$\vec b $$ = 0, perpendicular, $$\vec a $$ onto $$\vec b $$ is zero <br/>
<span style="background-color: #FFFF00">Cross product tells: product of lengths of vectors move perpendicular direction with b.</span> When $$\vec a \times \vec b = || \vec a || || \vec b || $$, perpendicular, 获得最大值, 当 $$\vec a $$ 和 $$ \vec b $$ colinear, $$\vec a \times \vec b $$ = 0 no perpendicular vector


注: <span style="color: red">difference between perpendicular and orthogonal</span>:  $$  \vec  x \cdot \vec y  = \vec 0 $$   only means orthogonal, zero vector is orthogonal to everything but zero vector not perpendicular to everything; perpendicular is orthogonal, 但是othogonal 不一定是perpendicular, 因为 $$  \vec  x \cdot \vec 0  = 0 $$ 不是perpendicular

求basis的coordinate的时，若basis vector orthogonal to each other, 可以用scaler projection, 看$$  \vec x$$到$$  \vec v_i$$的projection $$ \frac{ \vec x \cdot \vec v  }{ \|\vec v\|^2 } $$ 即是coordinate

## Basis

Basis is a set of n vectors that 
- are not linear combinations of each other (linear independent): <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for all $$ a_1, …, a_n \subseteq F $$, if $$ a_1v_1 + … + a_nv_n = 0 $$, then necessarily  $$ a_1 = … = a_n = 0;  $$
- span the space: <br/> 
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; every (vector) $$\vec x$$ in V it is possible to choose $$ a_1, …, a_n \subseteq  F $$ such that $$ x = a_1 \vec v_1 + … + a_n \vec vn. $$
- The space is then n-dimensional 

## Inverse

 $$ A =  \begin{bmatrix} a & b  \\ c & d \\ \end{bmatrix}, A^{-1} =  \frac{ 1  }{ det\left(A\right) } \begin{bmatrix} d & -c  \\ -b & a \\ \end{bmatrix} = \frac{ 1  }{ ad - bc } \begin{bmatrix} d & -c  \\ -b & a \\ \end{bmatrix}  $$

多维的inverse 可以row operation, 需要inverse matrix 在左面，indentity matrix在右侧，把左侧的matrix通过row operation变成indentiy matrix，一样的row operation 也apply 在右侧，<span style="color: red">最后得到右侧的matrix</span>就是inverse 

$$ \left[ \begin{array}{ccc \| ccc}  1 & 0 &1 &1 & 0 &0  \\ 0 & 2 & 1 &0 & 1 &0  \\ 1&1&1 &0 & 0 &1 \end{array} \right]  =>  \left[ \begin{array}{ccc\|ccc}  1 & 0 &0 & -1 & -1 &2   \\  0 & 1 &0 & -1 & 0 & 1  \\ 0 & 0 &1 & 2&1&0  \end{array} \right] $$

A square matrix is __not invertible__ is called __singular__ or **degenerate**. A square matrix is singular if and only if its __determinant__ is 0. <span style="background-color: #FFFF00">Non-square matrices</span> do not have inverse(因为it will always be the case that either the rows or columns (whichever is larger in number) are linearly dependent for non-square matrix).

<span style="background-color: #FFFF00">Invertibility  <=> unique solution to $$ A \vec x = \vec 0$$ </span>

 $$ \left( A^{-1} \right)^{-1} = A  $$ <br/>
 $$ \left( kA \right)^{-1} = k^{-1} A^{-1}  $$ for nonzero scalar K <br/>
 $$ \left( AB \right)^{-1} =  B^{-1}A^{-1}  $$  <br/>
 $$ \left( A^T \right)^{-1} =  \left( A^{-1} \right)^T  $$  <br/>
 $$ det\left( A^{-1} \right) =  det\left( A \right)^{-1}  $$  <br/>

 inverse of diagonal matrix 是每个diagonal 数的倒数

 $$ A = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{bmatrix},  A^{-1} = \begin{bmatrix} 0.5 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{bmatrix} $$

 if A be a square n by n matrix is invertible, following must be __all true of all false__
 - A is inveritble, A has an inverse, is nonsingular, or is nondegenerate
 - A is row-equivalent to the n-by-n identity matrix In.
 - A is column-equivalent to the n-by-n identity matrix In.
 - A has n pivot positions.
 - A has full rank; that is, rank A = n.
 - The equation Ax = 0 has only the trivial solution x = 0.
 - The kernel of A is trivial, i.e., it contains only the null vector as an element, ker(A) = {0}.
 - Null A = {0}.
 - The equation Ax = b has exactly one solution for each b in $$R^N$$
 - The columns of A are linearly independent.
 - The columns of A span $$R^N$$.
 - Col A = $$R^N$$.
 - The columns of A form a basis of $$R^N$$.
 - The linear transformation mapping x to Ax is a bijection(onto and one-to-one) from $$R^N$$ to $$R^N$$.
 - There is an n-by-n matrix B such that AB =  $$I_n$$ = BA.
 - The transpose  $$A^T$$ is an invertible matrix (hence rows of A are linearly independent, span $$R^N$$, and form a basis of  $$R^N$$).
 - The number 0 is not an eigenvalue of A.

## Transpose

 $$ \left( A^T \right)^{T} = A  $$ <br/>
 $$ \left( A + B \right)^T = A^T + B^T $$ for nonzero scalar K <br/>
 $$ \left( AB \right)^{T} =  B^T A^T  $$   <br/>
 $$ \left( cA \right)^T = cA^T  $$ for scalar c <br/>
 $$ det\left( A^{T} \right) =  det\left( A \right)  $$  <br/>
 $$ a \cdot b = a^Tb   $$ for column vectors a and b  <br/>
 $$ Rank \left(A\right) = Rank \left(A^T\right) $$ <br/>

$$\left( A\vec x \right)  \cdot \vec y = \left( A\vec x \right)^T \vec y = \vec x^T A^T \vec y =  \vec x^T \left( A^T \vec y \right) = \vec x \cdot \left( A^T \vec y \right) $$

if $$  A^T = A $$, then A is called __symmetric matrix__

## Linear Independence and Span

a set of vectors is __linearly dependent__ if one of the vectors in the st can be defined as a linear combination of the others. 比如$$ \begin{bmatrix} 2 \\ 3 \\ \end{bmatrix}, \begin{bmatrix} 4 \\ 6 \\ \end{bmatrix} $$ is linear dependent  If no vector in the set can be written in this way, then vectors are __linearly independent__

n vectors in $$R^N$$ are __linearly independent__ if and only if the <span style="background-color: #FFFF00">determinant </span>of the matrix formed by the vectors as its column is non-zero. ($$det\left(A \right)\neq 0 $$ or $$A\left(\vec x \right) = \vec 0$$ only has trivial solution)

$$span \left( v_1, v_2, \cdots, v_n \right) = \{ c_1v_1 + c_2v_2 +\cdots + c_nv_n \mid c_i \in R  \}  $$ The space of all of the combination of vectors $$v_1, v_2, ...,v_n$$

## Subspace 

if V is subspace of $$R^N$$
1. The zero vector 0 is in V
2. if  $$ \vec x $$ in V, any scaler c, then the scalar product $$ c\vec x $$ is in V (closure under scaler multiplication)
3. if  $$ \vec a $$ in V and $$ \vec b $$ in V , then $$ \vec a + \vec b $$ also in V (closure under addition)

![](\img\post\Linear-Algebra\pic1.PNG)


## Null/Column Space

__Dimension__: the number (cardinality) of a basis of V

the __kernel(nullspace)__ of a linear map L : V → W between two vector spaces V and W, is the set of all elements v of V for which L(v) = 0, where 0 denotes the zero vector in W: <br/>
$$ ker\left(L\right) = \{ v \in V | L\left( v \right) = \vec 0 \} $$ <br/>
or in $$R^N:  N = \{ \vec x \in R^n | A \vec x =  \vec 0  \} $$  

Null space is valid subspace 满足: <br/>
1. $$ \vec 0 $$ in subspace
2. if $$ \vec v_1, \vec v_2 \in N $$, then  $$ A \left( \vec v_1 +  \vec v_2 \right) = A\vec v_1 + A\vec v_2 = \vec 0 \in N $$
3. if $$ \vec v \in N $$, $$ A \left(c \vec v \right) = c \left( A\vec v \right) \in N $$

__Nullity__: Dimension of Null space = number of free variables <span style="color: red">free variables (non-pivot) </span> in reduced echelon form in Matrix A
$$N\left( A \right) =\left( \vec 0 \right) $$ if and only if column vectors of A are linearly independent (only apply when A is n by n matrix)

 The __column space (also called the range or image)__ of a matrix A is the span (set of all possible linear combinations) of its column vectors. The column space of a matrix is the __image__ or __range__ of the corresponding matrix transformation.

 $$A = \left[\vec v_1, \vec v_2, \cdots, \vec v_n \right] = span \left(\vec v_1, \vec v_2, \cdots, \vec v_n \right) $$

Column space is valid subspace 满足: <br/>
1. $$ \vec 0 $$ in subspace
2. if $$ \vec b, \vec c \in C\left(A\right) $$, then $$  \vec b + \vec c = \left( b_1 +  c_1  \right) \vec v_1 + \left( b_2 +  c_2  \right) \vec v_2 + \cdots + \left( b_n +  c_n  \right) \vec v_n  \in  C\left(A\right) $$
3. if $$ \vec v \in C\left(A\right) $$, $$ A \left(c \vec v \right) = c \left( A\vec v \right) \in C\left(A\right)$$

__Rank__: Dimension of Column space = number of <span style="color: red">pivot variables </span> in reduced echelon form in Matrix A (the number of linear independent column vector)

__Rank Nullity Theorem__<br/>
$$ nullity\left( A \right) + rank\left(  A \right) = dim\left( V \right) $$


## Onto and One-to-One

#### Onto
__Onto (surjective)__: every elements in co-domain y ∈ Y, there exist at least one x ∈ X such that f(x) = y. 每一个Y都至少都有一个X与之相对应, 一个X对应多个Y 或 有个X 没有对应Y 无所谓

following statements are quivalent (A is m by n matrix): <br/>
1. T is onto
2. T(x) = b has at least one solution for every b in $$R^m$$ 
3. The columns of A span $$R^m$$ (否则不能span $$R^m$$)
4. A has a pivot in every row

#### One-to-One

__One-to-one (injective)__: for every y that map to, there at most at most one x map to it. 每个一个y只有一个x map, 每个x map to unqiue y: f(x)=y, 有的Y没有x对应上 无所谓, 可以有y 没有x map to

following statements are quivalent (A is m by n matrix): <br/>
1. T is one-to-one
2. T(x) = b has at most one solution for b in $$R^m$$
3. The columns of A are linearly independent (因为每一个y最后只有一个x)
4. A has pivot in every column
5. Ax = 0 has only the trival solution

$$f: x -> y $$ is invertible if and only if f is onto and one-to-one <br/>
Invertible means For every y ∈ Y f(x) = y has a unique solution, that means one-to-one, 如果有y ∈ Y 但是没 有相应的 x 对应，就不是 invertible 了,所以 invertible means onto

| T is one-to-one | T is onto |
|------|------|
| T(X) =b has at most one solution for every b | T(x) = b has least one solution for every b | 
| The columns of A are linearly independent |  The columns of A span $$R^m$$ | 
| A has a pivot in every column | A has a pivot in every row | 



## Linear Transformation:

 __linear transformation__ (linear mapping, linear map) is a mapping V → W between two modules (including vector spaces) that preserves the operations of addition and scalar multiplication. <br/>
$$ T: R^n -> R^n $$  if and only if  $$\vec a , \vec b \in R^n 1. T\left( \vec a + \vec b \right) = T\left( \vec a \right) + T\left( \vec b \right) 2. T\left(c\vec a\right)  = cT\left(\vec a \right) $$

#### matrix product properties

__Associative__ :  (AB)C = A(BC).   <br/>
__Distributive__:  A(B+C) = AB + BC,  (B+C)A = BA + CA <br/>
__Not Communicative__: $$AB \neq BA $$

## LU Decomposition

1. A is n by n nonsingular matrix. A = LU where L and U are triangular matrices. 
2. <span style="color: red">U is a upper triangular matrix with the pivots on its diagonal</span> . Gaussian Elmination is from A to U.
3. <span style="color: red">L is lower triangular matrix. The entries of L are exactly the multiplier $$l_{ij}$$ - 表示 Gaussian elimination 第i行 -= $$l_{ij}$$第j行, 从reduced form to A, 第i行 += $$l_{ij}$$第j行. 
4. Every inverse matrix $$E^{-1}$$ is lower triangular. $$E_{ij}$$ 表示 j 行 += $$E_{ij}$$i 行 to make ij entry as zero *The lower triangular product of inverse is L*. <span style="color: red"> Every $$E^{-1}$$ has 1 down its diagonal </span>
5. If U diagonal is 1, then we can write $$A = LU \space \space or \space \space A = LDU$$  <span style="color: red"> where D is diagonal matrix contains pivot and U has 1 on the diagonal </span>
6. Solve $$Ax = LUx = b $$ can be solve by two step: $$Lc = b$$ (foward) then solve $$Ux = c$$ (backword)
   - before we augment to $$\left[A b \right]$$, but most computer codes keep two sides seprate. The memory of elimination is held in L and U, to process b whenever we want to.

e.g. $$A = \begin{bmatrix} 2 & 1 \\ 6 & 8 \end{bmatrix}$$

$$Forward \space from \space A \space to \space U  \space \space E_{21}A = \begin{bmatrix} 1 & 0 \\ -3 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 6 & 8 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 0 & 5 \end{bmatrix}  = U $$

$$Back \space from \space U \space to \space A  \space \space E_{21}^{-1} A = \begin{bmatrix} 1 & 0 \\ 3 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 0 & 5 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 6 & 8 \end{bmatrix}  = A$$

$$Row \space 3 \space of \space U = \left(Row \space 3 \space of \space A \right) - l_{31}\left(Row \space 1 \space of \space U \right) - l_{32} \left(Row \space 2 \space of \space U \right) $$

$$Row \space 3 \space of \space A = \left(Row \space 3 \space of \space U \right) + l_{31}\left(Row \space 1 \space of \space U \right) - l_{32} \left(Row \space 2 \space of \space U \right)$$

$$ Row \space 1 \space of \space U = Row \space 1 \space of \space A $$

Gaussian Elimination 每一步从A 到 U 是乘以 matrix $$E_{ij}$$ to produce zero in (i,j) position, 3 by 3 matrix  $$\left(E_{32} E_{31} E_{21} \right) A = U $$ and $$A = \left(E_{21}^{-1} E_{31}^{-1} E_{32}^{-1} \right) U $$ 

show (3) e.g. Elimination subtracts $$\frac{1}{2}$$ times row 1 from row 2. The last step substract $$\frac{2}{3}$$ times row 2 from 3. The (3,1) multiplier is zero because the (3,1) Entry in A is zero. No operation needed

$$A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ \frac{1}{2} & 1 & 0 \\ 0 & \frac{2}{3} & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 & 0 \\ 0 & \frac{3}{2} & 1 \\ 0 & 0 & \frac{4}{3} \end{bmatrix}   $$

show (5): *Divide U by a diagonal matrix D that contains the pivots*. That leaves a new triangular matrix with 1's on the diagonal

$$A = LDU = L \begin{bmatrix} d_1 & & & \\ & d_2 & & \\ & & \ddots &  \\ & & & d_n \end{bmatrix}  \begin{bmatrix} 1 &  u_{12} /d_1 & u_{13} /d_1  & . \\  & 1 & u_{23} /d_2 & .  \\  &  & \ddots & \vdots  \\ & & & 1 \end{bmatrix}  $$


## Projection on plane

vector projection $$\vec x \cdot \vec v $$因为是scaler 可以放后面

 $$  \vec proj_{L} \left(\vec x\right) =  c \vec v =  \frac{ \vec x \cdot \vec v  }{ \vec v \cdot \vec v } \vec v =  \frac{ \vec v  \left(\vec v \cdot \vec x \right)}{ \vec v \cdot \vec v }  =  \frac{ \vec v \space \vec v^T \vec x }{ \vec v \cdot \vec v } = \frac{ \vec v \space \vec v^T }{ \vec v \cdot \vec v }  \vec x $$

if A is matrix consists of basis of V, then :<br/>
$$Proj_v \vec x = A \left( A^T A \right)^{-1} A^T \vec x $$

__Prove__: if A is n by k matrix of basis for V $$\{ \vec b_1 , \vec b_2 , \cdots, \vec b_k \}$$, then all vectors on V is linear combination of A's columns,  such that $$ \vec y \in R^k, A\vec y = \{ \vec b_1 , \vec b_2 , \cdots, \vec b_k \} \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_k \end{bmatrix} = y_1 \vec b_1 + y_2 \vec b_2 + \cdots + y_k \vec b_k $$, <br/>
根据projection定义, $$\vec x = Proj_v \vec x  + \vec w $$(构成直角三角形), $$ Proj_v \vec x = A\vec y$$ 与 $$\vec w$$ orthogonal, $$\vec w$$ is member of $$V^{\bot} = C\left( A \right)^{\bot} =  N\left( A^T \right) $$. 根据null space 定义  

$$ A^T\left(\vec x - Proj_v \vec x \right) = \vec 0 $$ 

$$ A^T \vec x - A^T Proj_v \vec x  = \vec 0 $$ 

$$  A^T \vec x = A^T A \vec y  $$

$$  \vec y = \left(A^T A \right)^{-1}  A^T \vec x $$ 

$$  Proj_v \vec x  = A \left(A^T A \right)^{-1}  A^T \vec x   $$

 <span style="color: red">注:</span> If A is square matrix with independent columns, $$  Proj_v = Indentiy \space matrix $$ 


## Orthogonal

__Orthogonal Complement__ of V: for some V, $$V^{\bot} = \{ \vec x \in R^n  \space \mid \vec x \cdot \vec v = 0 \space for \space  every \space \vec v \in V \}$$ . Orthogonal complements is valid subspace

null space is orthogonal complement of row space <br/>
$$N\left(A \right) = \left(  C\left( A^T \right) \right)^{\bot}, \left(   N\left(A \right) \right)^{\bot}= C\left( A^T \right) $$ <br/>
left null space is orthgonal complement of column space <br/>
$$N\left(A^T \right)  = \left(  C\left( A \right) \right)^{\bot},  \left( N\left(A^T \right) \right)^{\bot}  =  C\left( A \right) $$ <br/>

$$Dim\left( V \right) + Dim\left( V^{\bot} \right) = nums \space of \space columns $$

#### Orthonormal Basis

A subset $$\{ \vec v_1, \vec v_2, \cdots, \vec v_k \}$$ of a vector space V, with inner product <,>, is called __orthonormal__ if $$<\vec v_i, \vec v_j> = 0 \space when \space i \neq j $$. That is, the vector are mutually orthogonal. Moverover, each length is one  $$<\vec v_i, \vec v_i> = 1. $$

$$ B = \{ \vec v_1, \vec v2, \cdots, \vec v_k \} $$ is __orthonormal set__ 需满足 1. $$\| \vec v_i \| = 1$$ each vector has length = 1. &nbsp;&nbsp; 2. Each vector is orthogonal to each other. 

An orthonormal set must be  <span style="color: red">linearly independent</span>, and so it is a vector basis for the space it spans. Such a basis is called an __orthonormal basis__

properties: 
1. The column vectors are linearly independent.
2. if $$ \vec x = c_1 \vec v_1 +  c_2 \vec v_2 + \cdots +  c_k \vec v_k$$, then $$\vec v_i \cdot \vec x = c_1 \vec v_1 \cdot \vec v_i +  c_2 \vec v_2 \cdot \vec v_i + \cdots +  c_k \vec v_k \cdot \vec v_i = c_i \vec v_i \cdot \vec v_i = c_i $$
3. $$ \left[ \vec x \right]_B = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_k \end{bmatrix} =  \begin{bmatrix} \vec v_1 \cdot \vec x \\ \vec v_2 \cdot \vec x  \\ \vdots \\ \vec v_k \cdot \vec x  \end{bmatrix} $$
4. If orthonormal basis 组成matrix A， 则 $$ A^T A = I_k $$ the identity matrix <br/>
    prove: $$ A^T A = \begin{bmatrix} -- \vec v_1^T -- \\  --\vec v_2^T-- \\ --\vdots-- \\ --\vec v_k^T-- \end{bmatrix}  \begin{bmatrix} \mid &\mid &\cdots &\mid \\  \vec v_1 & \vec v_2 & \cdots  &\vec v_k \\ \mid &\mid &\cdots &\mid \end{bmatrix} =  \begin{bmatrix} 1 & 0 &\cdots & 0 \\  0 & 1 & \cdots & 0  \\ \vdots &  \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}  $$
5. __Left Invertible__: A is n by n matrix whose columns form an orthonormal set, then $$A^{-1} = A^T $$ (因为是 n by n, 所以是invertible)
6. If orthonormal basis 组成的matrix A,  then $$Proj_v \vec x = A \left( A^T A \right)^{-1} A \vec x = A A^T \vec x $$
7. If orthonormal basis 组成的matrix A, then when do transformation, it will preserve length and angle for transformation. It aslo preserve inner product. prove: 


   $$ \| C \vec x \|^2 = C\vec x \cdot C \vec x = \left( C\vec x \right)^T C \vec x = \vec x^T C^T C \vec x =\vec x^T \vec x =  \| C \vec x \|   $$ 

   $$cos\theta =  \frac{ C\vec w \cdot C \vec v }{ \| C\vec v \| \| C\vec v \| } =  \frac{ \left(C\vec w \right)^T C \vec v }{ \| \vec v \| \| \vec v \| } =  \frac{ \vec w^T C^T C \vec v }{ \| \vec v \| \| \vec v \| }  = \frac{ \vec w^T \vec v }{ \| \vec v \| \| \vec v \| } $$

  $$ \left( A x \right)^T \left(A y \right) = x^T A^T A y = x^T y  $$ 

#### Orthogonal Matrix

A nonsingular matrix Q is called an <span style="background-color: #FFFF00">__orthogonal matrix__ if  $$Q^{-1} = Q^T$$</span>. Q is orthogonal matrix if and only if the columns of Q form an orthonormal set of vectors in $$R^n$$. (a square real matrix with orthonormal columns is called orthogonal) 

 <span style="color: red"> 注</span>: Orthogonal matrix 必须是square，否则non-square matrix 没有inverse

<span style="background-color: #FFFF00"> 只要column vectors (dimension n) 是orthonormal(each length = 1 and mutually perpendicular),就可以组成orthogonal matrix (n by n), 并且row vectors 同样是orthonormal </span>

Wiki: An __orthogonal matrix__ is a square matrix whose columns and rows are orthogonal unit vectors i.e. $$Q^TQ = QQ^T = I$$

prove orthonormal columns form orthogonal matrix:

方法一: orthonormal columns is left invertible. $$Q^T Q = I => Q^{-1} = Q $$, because $$Q \space Q^{-1} = I => Q \space Q^T = I $$, columns 组成的matrix row 也是orthonormal => matrix is orthogonal 

方法二: Since $$Q^TQ = I$$, then Projection matrix is $$Q \left(Q^T Q \right)^{-1} Q^T = Q Q^T$$, 我们知道Square matrix (n by n) with n independent columns, Projection matrix is indentity matrix, so we have $$ Q Q^T = I$$.

properties: 
1. The column vectors and row vectors forms orthonormal basis
2. transpose = inverse: $$Q^T = Q^{-1}$$
3. The determinant of any orthogonal matrix is either +1 or -1

#### Gram-Schmidt process

__Gram-Schmidt process__, is a procedure which takes a nonorthogonal set of linearly independent functions to constructs an orthogonal basis

![](\img\post\Linear-Algebra\pic2.png)

#### QR Decomposition

Any  square matrix A may be decomposed as A = QR where Q is an __orthogonal matrix__ (its columns are orthogonal unit vectors $$O^T Q = QQ^T = I $$ ) and R is an __upper triangular matrix__ (right triangular matrix).

If A is invertible, then the factorization is unique if we require the diagonal elements of R to be positive.

$$AR^{-1} = Q $$ where a is square matrix and <span style="color: red"> Q is the orthogonal matrix(columns, rows are orthonormal) from Gram-schmidt process</span>, $$R^{-1}$$ 是upper triangular matrix 表示row operation 得到orthogonal matrix (example in PDF)
 
$$ A = QR = > Q^{-1} A = Q^{-1} Q R  => Q^T A = R $$

## Eigenvector & Eigenvalue

<span style="color: red">注:</span> eigenvalues & eigenvectors only apply to square matrix. For non-square matrices we use singular values

If T is a linear transformation from a vector space V over a field F into itself(V) and $$\vec v$$ is a not a zero vector, then  $$\vec v$$ is an __eigenvector__ of T if  $$T\left(\vec v\right)$$  is a scalar multile of v. It can be written: 

$$ T\left(\vec v \right) = \lambda \vec v $$

where $$\lambda$$ is a scalar in the field F, known as __eigenvalue__ associated with __eigenvector__ $$\vec v$$

$$ A \vec v = \lambda \vec v $$

$$ A \vec v - \lambda  I_n \vec v = \vec 0 $$

$$ \left( A - \lambda  I_n  \right) \vec v = \vec 0 $$

因为 $$\vec v $$ is non-zero vector, $$ \left( A - \lambda  I_n  \right) $$ 必须是linear dependent matrix, 否则只有trivial soluion {0}, so $$ det\left( A - \lambda  I_n  \right) = 0 $$ 

- Eigenvalue can be 0. Then $$A \vec v = \lambda \vec 0$$ means that $$\vec v$$ is in nullspace
- we can multiply eigenvectors by any nonzero constants. $$A\left(c \vec v \right) = \lambda\left(c \vec v \right)$$ is still true
- If A is Indetity matrix, then eigenvalue is 1, $$A \vec v = \vec v$$
- If $$A \vec v = \lambda \vec v $$ then $$A^2 \vec v = \lambda^2 \vec v $$  and $$A^{-1} \vec v = \lambda^{-1} \vec v $$ for the same $$\vec v$$
- <span style="color: red">If A is singular, then 0 is an eigenvalue</span>.  Prove: A is singular => det(A) = 0  =>  $$det\left(A- \lambda I \right) = 0 => det\left(A - 0I \right) = 0 $$;  因为A若不是singluar, 需要$$A - \lambda I$$让它变成singular, 若已经是singular, 无须减去$$\lambda$$
- Row operation (Elimination) don't always preserve eigenvalue (比如第二行 += 第一行). 
- <span style="color: red">Triangular matrix has Eigenvalue on its diagonal </span> (因为triangular的determinant = diagonal 数的乘积)
- <span style="background-color: #FFFF00">The product of eigenvalues = determinant </span>
- <span style="background-color: #FFFF00"> The sum of eigenvalues = the sum of n diagonal entries </span>. The sum of the entries along the main diagonal is called the __trace__ of A
- $$A^TA $$ and  $$AA^T$$ are both symmetric matrix, <span style="color: red">both share the same eigenvalues</span>

$$\lambda_{1} + \lambda_{2} + \cdots + \lambda_{n} = a_{11} + a_{22} + \cdots + a_{nn} $$

prove The product of eigenvalues = determinant: 

suppose $$\lambda_1 , \lambda_2,  \cdots , \lambda_n$$ are the eigenvalues of A, Then $$ \lambda s $$$ are aslo the roots of the characteristic polynomial

$$\begin{align} det\left(A - \lambda  I \right) &= \left(-1\right)^n \left( \lambda - \lambda_1 \right)\left(\lambda - \lambda_2 \right) \cdots \left( \lambda - \lambda_n \right) \\ &= \left(\lambda_1 - \lambda \right)\left(\lambda_2 - \lambda \right) \cdots \left(\lambda_n - \lambda \right)   \end{align}$$

$$when \space \lambda = 0, \space det\left(A\right) = \lambda_1 \lambda_2 \cdots  \lambda_n $$

#### Diagonalization

why Diagonal matrix useful?
- <span style="color: red">Determiant = diagonal所有数的乘积 </span>
- <span style="color: red">Eigenvalues 就是每个diagonal上(对角线上)的数 </span>
- Eigenvalue 是unit vector
- <span style="color: red">Power of matrix 就是diagonal上数的数=power </span>
  
$$A = \begin{bmatrix} 3&0 \\ 0&7 \end{bmatrix}, det\left( A\right) = 3*7 = 21, eigenvalues = 3,7, A^k = \begin{bmatrix} 3^k &0 \\ 0&7^k \end{bmatrix}$$

__Diagonalization__: Suppose the n by n matrix A has __n linearly independent eigenvectors__ $$x_1, \cdots, x_n$$. Put them into the columns of an __eigenvector matrix X__. Then $$X^{-1}AX$$ is the __eigenvalue matrix__ $$\Lambda$$ (capital lambda)
 
$$X^{-1}AX = \Lambda = \begin{bmatrix} \lambda_1 & & \\ & \ddots & \\ & &\lambda_n \end{bmatrix}, \space A = X \Lambda X^{-1} $$

$$AX= X \Lambda $$,  我们知道$$X \Lambda = \left[ \lambda_1 \vec v_1 , \lambda_2 \vec v_2 , \cdots, \lambda_n \vec v_n  \right] = AX $$ 得到是每个eigenvector 乘以相应的eigenvalue 组成的matrix, 

<span style="color: red">注:</span> 当有n个不同eigenvalues, matrix 一定diagonalizable, 因为n个distinct eigenvalues implies n linearly independent eigenvectors. <span style="background-color: #FFFF00">*Any matrix that has no repeated eigenvalues can be diagonalized, An n by n matrix that has n different eigenvalues must be diagonalizable*</span>.

<span style="color: red">注:</span> 即使eigenvalues 不够 n个, matrix may or may not diagonalized. 比如indentity matrix 的eigenvalues 都是1, 但是eigenvectors matrix 也可以是identity matrix (1 on diagonal), 也是n 个independent eigenvectors. 但是比如matrix = $$\begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$$ not diagonalized, eigenvalues = 2, eigenvectors = $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$ only 1 eigenvector

**prove eigenvectors from different eigenvalues are linearly independent**: suppose $$v_1,v_2$$ are eigenvector correspond to distinct eigenvalues $$\lambda_1$$ and $$\lambda_2$$, to show linearly independent, $$a_1v_1 + a_2v_2 = 0$$, we need to show $$a_1 = a_2 = 0$$

$$T\left(0 \right) =  T\left(a_1v_1 + a_2v_2  \right) = a_1\lambda_1 v_1 + a_2\lambda_2 v_2 \tag{1}\label{eq1}$$

Now instead multiply the original equation by $$\lambda_1$$ 

$$\lambda_1 \left(a_1v_1 + a_2v_2 \right) = a_1 \lambda_1 v_1  + a_2 \lambda_1 v_2  = 0 \tag{2}\label{eq2}$$

Use (1) - (2)

$$ 0  = a_2 \left(\lambda_2 - \lambda_1 \right)v_2 $$

Since $$\lambda_2 - \lambda_1 \neq 0$$ and $$v_2 \neq 0$$, then $$a_2 = 0$$. Then $$a_1v_1 + a_2v_2 = 0 = a_1v_1 = 0$$, Since $$v_1 \neq 0$$, then $$a_1 = 0$$. So $$v_1$$ and $$v_2$$ are linearly independent

Remark: 
1. The eigenvectors in X come in the same order as the eigenvalues in $$\Lambda$$. 
2. Some matrices have too few eigenvectors. Those matrices cannot be diagonalized. Here are two examples. $$A = \begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix}, B = \begin{bmatrix} 0 & -1 \\ 0 & 0 \end{bmatrix}$$. The eigenvectors are 0 and 0.

<span style="background-color: #FFFF00">No connection between invertibility and diagonalizability:</span>
- <span style="color: red">*Invertibility*</span> is concerned with <span style="color: red">*eigenvalues*</span> ($$\lambda =  0 \space or \space \lambda \neq 0$$)
- <span style="color: red">*Diagonalizability*</span> is convered with the <span style="color: red">*eigenvectors* </span>(too few or enough for X, n 个eigenvectors, 即使不是n个eigenvalues, 可能也会有n个eigenvectors)

properties: 
- $$A^2$$has the same eigenvectors in X and squared eigenvalues in $$\Lambda^2$$, $$A^2 = X \Lambda X^{-1} \space X \Lambda X^{-1} = X \Lambda^2 X^{-1} $$
- Power of A: $$A^k = X \Lambda X^{-1} \space X \Lambda X^{-1} \cdots X \Lambda X^{-1}  = X \Lambda^k X^{-1} $$
- Matrix X has inverse, beacuse coumns were assumed to be linearly independent. <span style="color: red">Without n independent eigenvectors, we can't diagonalize</span>
- <span style="background-color: #FFFF00">$$A$$ and $$\Lambda$$ have the same eigenvalues $$\lambda_1, \lambda_2, \cdots, \lambda_n$$</span>, The eigenvectors are different.
    - Suppose the eigenvalues $$\lambda_1, \cdots, \lambda_n$$ are all <span style="color: red">different</span>. Then it is automatic that the eigenvectors $$x_1,\cdots, x_n$$ are <span style="color: red">independent</span>. <span style="color: red">The eigenvector matrix X will be invertile</span>. 
    - All the matrices $$A = B^{-1}CB $$ are *similar*. <span style="background-color: #FFFF00">They all share the eigenvalues of C</span>
- if all $$\mid \lambda \mid < 1$$, then $$A^k $$ -> zero matrix

Let A and C be n by n matrices. We say that A is __similar__ to C if there is an invertible n by n matrix B such that $$A = B^{-1}CB $$, <span style="color: red">C may not be diagonal</span>


prove if A and C are similar, they share the same eigenvalues: suppose $$Cx =\lambda x$$, then $$A = BCB^{-1}$$ has the eigenvalue $$\lambda $$ with the new eigenvector Bx: 

$$A\left( Bx\right) =\left( BCB^{-1} \right) \left( Bx\right) = BCx = B\lambda x = \lambda \left( Bx\right) $$

*Application*: Fibonacci number, $$u_{k+1} = Au_{k}, where \space A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} $$, then $$u_k = A^k u_0  =  X \Lambda^k X^{-1}u_0  =  X \Lambda^k c$$, $$ u_0 = \begin{bmatrix} 1  \\  0 \end{bmatrix} $$, $$u_0 = c X$$, $$u_0$$可以表示为linear combination of eigenvector, Eigenvalue is $$\frac{1}{2}\left(1+\sqrt{5} \right) $$ and $$ \frac{1}{2}\left(1-\sqrt{5} \right)$$, so Fibonacci 数字增加速率是$$\frac{1}{2} \left(1+\sqrt{5} \right)$$

$$u_{k+1} \approx u_k, F_{100} \approx c_1 \left(\frac{1 + \sqrt{5}}{2} \right)^{100}, \space \space \space given \space \left(\frac{1 - \sqrt{5}}{2} \right)^{100} \approx 0 $$

$$u_k = A^{100} u_0 =  c_1 \left(\lambda_1\right)^k x_1 + \cdots + c_n \left(\lambda_n\right)^k x_n \space \space provided  \space \space u_0 = c_1x+1 + \cdots + c_nx_n$$

#### Symmetric Matrices

1. Every real symmetric S can <span style="color: red">always</span> be diagonalized : $$S = Q \Lambda Q^{-1} = Q \Lambda Q^T $$ with $$Q^{-1} = Q^T$$, there are always enough eigenvectors to diagonalize $$S = S^T$$, even with repeated eigenvalues.
2. A symmetric matrix S has n <span style="color: red">**real eigenvalues**</span> $$\lambda_i $$ and n <span style="color: red">orthonormal eigenvectors</span> $$q_1, \cdots, q_n$$ ($$Q^{-1} = Q^T$$)
3.  <span style="color: red">Null space is perpendicular to column space, row space is the same as column space</span>. As we know before, null space should be perpendicular to row space, 因为symmetric, column space = row space,  $$A = A^T$$, so $$N\left( A \right)^{\bot} \space = C\left(A^T\right) = C\left(A \right) $$
4. Every smmetric matrix has $$S = Q \Lambda Q^T = \lambda_1 q_1 q_1^T + \cdots + \lambda_n q_n q_n^T  $$, a combination of perpendicular projection matrix($$\lambda$$ 是combination的系数), $$q_1 q_1^T$$ is projection matrix $$q_1^T q_1 = I$$, $$q_1 \left(q_1^T q_1 \right)^{-1} q_1^T = q_1 q_1^T$$
5. For symmetric matrices the pivots and the eigenvalues have the same sign (有几个正的pivot 就有几个正的eigenvalue).
6. Every square matrix can be "triangularized" by $$A = Q T Q^{-1}$$, if $$A = S$$, then $$T = \Lambda$$

prove(1): S is a symmetric matrix so we have $$S = S^T$$, aslo we know Eigenvalue matrix is symmetric $$\Lambda = \Lambda^T$$

$$S = Q\Lambda Q^{-1} \space  and  \space   S^T =  \left(Q\Lambda Q^{-1}\right)^T = \left( Q^{-1}\right)^T \Lambda^T Q^T \space = Q\Lambda Q^{-1}  => Q^{-1} = Q^T $$

根据orthogonal matrix 定义: Q is orthogonal matrix if $$Q^{-1} = Q^T$$, <span style="color: red">Eigenvector matrix Q becomes an orthogonal matrix Q</span>

对于repeated eigenvalues, 比如10 by 10 indentity matrix, eigenvalues 都是1，但是eigenvector可以是orthonormal vector 比如 1 在diagonal上

$$\begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1   \end{bmatrix}$$

prove (2) Eigenvectors of a real symmetric matrix (when they correspond to different $$\lambda$$'s) are always perpendicular: Suppose $$Sx = \lambda_1x \$$ and $$Sy = \lambda_2 y \$$, we assume $$\lambda _1 \neq \lambda_2$$, we can use the fact $$S^T = S$$, Use dot product with two perpendicular vector is zero.

$$ \left( \lambda_1 x \right)^T y = x^T \lambda_1 y  = \left(Sx \right)^T y = x^T S^T y = x^T S y = x^T \lambda_2 y$$

The left side is $$x^T \lambda_1 y$$ = right side $$x^T \lambda_2 y$$, 因为$$\lambda_1 \neq \lambda_2$$ => so $$x^T y = 0$$

prove(4): <br/>
e.g. 2 by 2 symmetric matrix, we have $$S = \lambda_1 q_1 q_1^T + \lambda_2 q_2 q_2^T$$

$$S = Q \Lambda  Q^T = \left[q_1, \space q_2 \right] \begin{bmatrix} \lambda_1 & \\ & \lambda_2 \end{bmatrix}  \begin{bmatrix} q_1^T \\ q_2^T \end{bmatrix}$$

So if S is symmetric ( q's are orthonormal)  prove properties (4):

$$Sq_i =\left( Q \Lambda Q^T \right) q_i = \left( \lambda_1 q_1 q_1^T + \cdots + \lambda_n q_n q_n^T \right) q_i = \lambda_i q_i$$

(5):  Symmetric matrix: $$\begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}$$ has pivots 1 and -8 and eigenvalues 4 and -2, 一个positive pivot 和 一个positive eigenvalue, 一个negative pivot 和 一个negative eigenvalue

(**Spectral Theorem**): Every symmetric matrix has the factorization $$S = Q \Lambda Q^T$$ with real eigenvalues in $$\Lambda $$ and orthonormal eigenvectors in the columns of Q: 

$$Symmetric \space diagonalization \space \space \space \space \space \bbox[yellow]{ S = Q \Lambda Q^{-1} = Q \Lambda Q^T \space \space Q^{-1} = Q^T } $$

#### Positive Definite Symmetric Matrices
1. The matrix S is  <span style="color: red">**positive definite**</span> if the energy test is $$x^TSx > 0$$ for all vectors $$x \neq 0$$
2. If S and T are symmetric positive definite, so is T+S. 
3. For square matrix $$S = A^T A $$ is square and symmetric. <span style="color: red">If the columns of A are independent then S = $$A^TA$$ is positive definite</span>
4. For symmetric matrix :  <span style="color: red">all eigenvalues > 0 &nbsp; <=> &nbsp; all pivots > 0 &nbsp;  <=> &nbsp; all upper left determinants > 0 (所有eigenvalue 大于0， pivot都大于0，determinant 大于0)</span>
5. Positive semidefinite S allows $$\lambda = 0$$, pivot = 0, determinant = 0, and energy $$x^T S x = 0$$
6. $$A^TA, \space and \space AA^T$$ are both at least Positive semi-definite for any matrix
7. Inverse matrix of positive-definite symmetric matrix is positive-definite.

Prove (2):<br/>
$$x^T \left( S + T \right) x = x^T S x + x^T T x $$ if those two terms are positive (for $$x \neq 0 $$) so $$S+T$$ is aslo positive definite.

Prove (3): <br/>
$$x^T S x = x^T A^T A x = \left(A x \right)^T A x = Ax \cdot Ax = \| Ax \|^2 $$. That vector $$A x \neq 0 $$ when $$x \neq 0 $$ (this is meaning of independent columns). Then $$x^T S x $$ is the positive number $$\| Ax \|^2$$ and matrix S is positive definite

Prove (6): 利用SVD ($$U,V$$是orthonormal basis组成的matrix, $$\Sigma$$ 是diagonal matrix)<br/>
$$A^TA = \left(U \Sigma V^T \right)^T U \Sigma V^T = V \Sigma U^T U \Sigma V^T = V \Sigma^2 V^T$$<br/>
$$ \Sigma^2$$ will be eigenvalue of $$A^TA$$


<span style="background-color: #FFFF00">__验证matrix是positive definite__</span>, when a symmetrix matrix S has one of these five properties, it has them all (任何一个正确，就可以说是positive definite)
1. All **n pivots** of S are positive
2. All **n upper left determinants** are positive (THe upper left determinants 是 左上角 1 by 1, 2 by 2, ... n by n matrix determinants)
3. All **n eigenvalues** of S are positive
4. $$x^T S x$$ is positive except at $$x = 0$$. This is the **energy-based** definition
5. S equals $$A^TA$$ for a matrix A with independent columns

E.g. Test symmetric matrices S for positive definiteness:

$$\begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}$$

The pivots of S are 2 and $$\frac{2}{3}$$ and $$\frac{4}{3}$$ all positive. Its upper left determintes are 2, 3, 4 are all positive. The eigenvalues of S are $$ 2- \sqrt{2} $$ and $$2 + \sqrt{2}$$ all positive. That completes tests 1, 2, 3. Any one test is decisive!<br/> 
We also have $$A_1, A_2, A_3$$ suggest for $$S = A^TA$$

$$S = A_1^T A_1  \space \space \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & -1 & 0 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & 0 & 1 & -1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & -1 & 1 \\ 0 & 0 & -1 \end{bmatrix} $$ 

The three columsn of $$A_1$$ are independent, Therefore S is positive definite. <br/>
$$A_2$$ comes from $$S = LDL^T$$(the symmetric version of S = LU). Elimination gives the pivots $$2, \frac{3}{2}, \frac{4}{3}$$ in D and the multipliers $$-\frac{1}{2}, 0 , -\frac{2}{3}$$ in L. Just put $$A_2 = L \sqrt{D}$$, $$A_2$$ are also called Cholesky factor of S

$$L D L^T = \begin{bmatrix} 1 &  &  \\ -\frac{1}{2} & 1 &  \\ 0 & -\frac{2}{3} & 1 \end{bmatrix}  \begin{bmatrix} 2 &  &  \\  & \frac{3}{2} &  \\  &  & \frac{4}{3} \end{bmatrix} \begin{bmatrix} 1 & -\frac{1}{2} & 0 \\  & 1 & -\frac{2}{3} \\  &  & 1 \end{bmatrix} =  \left(L \sqrt{D} \right) \left(L \sqrt{D} \right)^T = A_2^T A_2 $$

Eigenvalues give the symmetric choice $$A_3 = Q \sqrt{\Lambda} Q^T $$. This is also successful with $$A_3^T A_3 = Q \Lambda Q^T = S$$. All tests show that matrix S is positive definite

To see that the enery $$x^TSx$$ is positive, we can write it as sum of square. 

$$x^T S x = 2x_1^2 - 2x_1x_2 + 2x_2^2 - 2x_2x_3 + 2x_3^2 \space \space \space Rewrite \space with \space squares$$

$$\|A_1 x \|^2 = x_1^2 + \left(x_2 - x_1 \right)^2 + \left(x_3 -x_2 \right)^2 + x_3^2  \space \space \space, Using \space differences \space in \space A_1$$ 

$$ \|A_2 x \|^2 = 2 \left(x_1 - \frac{1}{2} x_2\right)^2 + \frac{3}{2} \left(x_2 - \frac{2}{3} x_3\right)^2 + \frac{4}{3} x_3^2 \space \space \space Using \space S = LDL^T  $$

$$  \|A_3 x \|^2 =\lambda_1\left(q_1^T x \right)^2 + \lambda_2\left(q_2^T x \right)^2 + \lambda_3\left(q_3^T x \right)^2 \space \space \space Using S \space = Q\Lambda Q^T $$


**Positive semidefinite matrices** have all $$\lambda \geq 0$$ and all $$x^T S x \geq 0$$. The upper left determinants are non-negative. $$S = A^TA$$ may have dependent columns in A. e.g.$$\begin{bmatrix} 1 & 2 \\ 2&4 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2&0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 0&0 \end{bmatrix} $$

(a). **Prove**: the eigenvalues of a real symmetric positive-definite matrix A are all positive.Note: positive definite matrix meaning任何的x, $$x^TAx > 0$$, 所以x 也可以是eigenvector

$$Ax = \lambda x, \space multiply \space x^T \space, \space x^T A x = \lambda x^T x = \lambda|x|^{x} > 0$$

(b). **Prove**: if eigenvalues of a real symmetric matrix A are all positive, then A is positive-definite. Note: real symmetric matrix can be diagonalizable.  $$A = Q\Lambda Q^T$$


$$x^TAx = x^T Q \Lambda Q^Tx, \space put \space y = Q^T \space => \space x^TAx = y^TD y $$

$$y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \space \space x^TAx = y^T D y = \left[y_1, y_2, \cdots, y_n  \right] \begin{bmatrix} \lambda_1 & 0 & 0 & 0 \\ 0 & \lambda_2 & 0 & 0 \\ \vdots & \cdots & \ddots& \vdots \\ 0 & 0 & \cdots & \lambda_n   \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

$$x^TAx = y^T D y = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots +\lambda_n y_1^n > 0 \space for \space x \neq 0$$

(C). **Prove**: A is positive definite symmetric n by n matrix then $$A^{-1}$$ is positive definite symmetric matrix

A is invertible, because A is positive definite matrix, eigenvalue all bigger than zero, then $$A^{-1}$$ is symmetric, $$I = I^T = \left(A^{-1} A\right)^T = A^T \left( A^{-1}\right)^T = A \left(A^{-1}\right)$$, so we have $$A^{-1} = \left( A^{-1}\right)^T$$/ Then all eigenvalues of $$A^{-1}$$ are of the form $$1/\lambda$$, where $$\lambda $$ is eigenvalue of A. Since A (positive definite) $$\lambda \geq 0$$, then all eigenvalues of $$A^{-1} \geq 0$$

## Singular Value Decomposition

problem with Diagonalization ($$A = X\Lambda X^{-1}$$): 
1. Eigenvector matrix are usually not orthogonal. 
2. There are not always enough eigenvectors 
3. $$Ax = \lambda x$$ requires A to be a square matrix

$$A = U\Sigma V^T = u_1\sigma_1 v_1^T + \cdots + u_r\sigma_r v_r^T$$

For any Matrix A, can **always** found two orthogonal matrix(orthornomal) U (m by m matrix) and V(n by n matrix) and diagonal matrix $$\Sigma$$ (m by n matrix) where U's is called **left singular vectors** (unit eigenvectors of $$AA^T$$) and V's are called **right singular vectors**(unit eigenvectors of $$A^TA$$), The $$\sigma$$'s are called **singular values** (square roots of the equal eigenvalues of $$AA^T$$ and $$A^TA$$)

#### Geometric meaning  

For every linear map $$A: R^n -> R^m$$ one can find <span style="color: red">orthonormal bases of $$R^n$$</span> (the columns $$V_1, \cdots,V_n$$ yield an orthonormal basis) and <span style="color: red">$$R^m$$</span> ($$U_1, \cdots, U_m$$ yeild an orthonormal basis) such that A maps the  i-th basis vector of $$R^n$$ to a *non-negative* muliple of the i-th basis vector of $$R^m$$, and left-over basis vectors to zero(rank = r, 只有r个diagonal 是正的real number 剩下diagonal 和其余的都是0). The map A is <span style="color: red">a diagonal matrix with non-negative real diagonal entries</span>, where $$\sigma_i$$ is the i-th diagonal entry of $$\Sigma$$, and $$\bbox[yellow]{A\left( V_i\right) = 0 \space for \space i > min\left(m, n\right)}$$ 

$$The \space linear \space transformation: \space \space \space A: R^n -> R^m$$

$$A v_i = \sigma_i u_i, \space \space \space i = 1, \cdots, min \left(m, n \right) $$

以 2 by 2 matrix 为例, SVD 可以在 2 x 2 matrix上的几何意思是主要将两个orthonormal vectors 通过矩阵分解 变成成另一个 二维空间下两个orthornormal vectors

![](\img\post\Linear-Algebra\pic11.png)

We first choose two orthonormal vectors $$\vec v_1, \vec v_2$$, 通过T的linear transformation, 变成another two orthogonal vectors $$A \vec v_1, A \vec v_2$$, then we choose another two orthonormal vectors $$\vec u_1, \vec u_2$$ (unit length) on the direction of $$A \vec v_1, A \vec v_2$$, then we have $$A \vec v_1 = \sigma_1 \vec u_1; \space A \vec v_2 = \sigma_2 \vec u_2 $$

$$In \space left \space graph, \space  \space \space  x = \frac{\left( v_1 \cdot  x \right)}{\| v_1 \|^2}  v_1 + \frac{\left( v_2 \cdot  x \right)}{\| v_2 \|^2} v_2 =  \left( v_1 \cdot  x \right)  v_1 + \left( v_2 \cdot x \right)  v_2  $$

$$Then \space multiply \space A, \space \space  \space A x =  \left( v_1 \cdot x \right) A  v_1 + \left( v_2 \cdot  x \right) A  v_2  $$

$$ A x =  \left( v_1 \cdot x \right) \sigma_1  u_1 + \left( v_2 \cdot x \right) \sigma_2  u_2  $$

$$ A x =  u_1 \sigma_1 v_1^T x + u_2 \sigma_2 v_2^T x $$

$$ get \space rid \space of \space X: \space \space  \space A =  u_1 \sigma_1 v_1^T + u_2 \sigma_2 v_2^T $$


Goal: Orthogonal basis in row space (unit vector v) $$R^n$$ to Orthogonal basis in column space (unit vector u)  $$R^m$$

$$ A v_1 = \sigma_1 u_1,  A v_2 = \sigma_1 u_1$$

#### SVD

U will be m by m matrix, where rank(U) = r <br/> 
- $$\left[u_1, \cdots, u_r \right]$$ is an orthonormal basis for **column space** 
-  $$\left[ u_{r+1}, \cdots, u_m \right ] $$ is an orthonormal basis for the **left null space** $$N\left(A^T\right)$$(用left null space原因是: 与column space orthogonal, $$\Sigma$$ 与left null space对于的数都是0,不会transformation).  

V will be n by n matrix where rank(V) = r
- $$\left[v_1, \cdots, v_r\right]$$ is an orthonormal basis for **row space** 
- $$\left[v_{r+1}, \cdots, v_n\right]$$ is an orthonormal basis for the  **null space** $$N\left(A\right)$$

**Reduced SVD**: $$AV_r = U_r \Sigma_r$$, <span style="background-color: #FFFF00">A is m by n, $$V_r$$ is n by r, $$U_r$$ is m by r and $$\Sigma_r$$ is r by r </span>

$$A\left[v_1, \cdots, v_r \right] = \left[u_1, \cdots, u_r \right] \begin{bmatrix} \sigma_1 & 0 & \cdots & 0 \\  0 & \sigma_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \sigma_r  \end{bmatrix}$$

**Full SVD**:<span style="background-color: #FFFF00">A is m by n, U is m by m  $$\Sigma$$ is m by n, V is n by n </span>, 跟reduced SVD $$\Sigma$$(r by r)一样 with m - r extra zero rows and n - r new zero columns


$$A\left[v_1, \cdots, v_r, \cdots, v_n \right] = \left[u_1, \cdots, u_r, \cdots, u_m \right] \begin{bmatrix} \sigma_1 &  &  &   & \\   & \ddots &  &  &  \\  &  & \sigma_r & \\  &  &  &  \ddots & \\ &  &  &  & 0   \end{bmatrix}$$ 

The <span style="color: red">v's will be orthonormal eigenvectors of $$A^TA$$</span>, V is eigenvector matrix for symmetric positive (semi) definite matrix $$A^TA$$ ($$V \Sigma^T \Sigma V^T \geq 0 $$), $$\Sigma^T \Sigma $$ must be eigenvalue matrix of $$A^TA$$: <span style="color: red">Each $$\sigma^2$$ is $$\lambda$$!</span>

$$A^TA = \left(U \Sigma V^T \right)^T\left(U \Sigma V^T \right) = V \Sigma^T U^T U \Sigma V^T = V \Sigma^T \Sigma V^T$$

$$AA^T =\left(U \Sigma V^T \right) \left(U \Sigma V^T \right)^T =  U \Sigma V^T V \Sigma^T U^T = U \Sigma \Sigma^T U$$

**Singular Value Stability vs Eigenvalue Instability**

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 &0 & 3 \\ 0 & 0 & 0 & 0 \end{bmatrix}  $$, Eigenvalue = 0, 0, 0, 0, singular value = 3, 2, 1

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 &0 & 3 \\ \frac{1}{60000} & 0 & 0 & 0 \end{bmatrix}  $$, Eigenvalue =  $$\frac{1}{10},  \frac{i}{10}, \frac{-1}{10}, \frac{-i}{10} $$, singular value = $$ 3, 2, 1, \frac{1}{60000} $$



**Example 1**: When is $$A = U \Sigma V^T$$ (singular values) the same as $$X \Lambda X^{-1}$$ (eigenvalues)?

A needs orthonormal eigenvectors to allow $$X = U = V$$. A aslo needs eigenvalues $$\lambda  \geq 0 $$ if $$\Lambda = \Sigma$$ (SVD $$\Sigma$$ 需要非负数). So A must be a positive semidefinite (or definite) symmetric matrix (因为eigenvalues都大于等于0). Only then $$A = X \Lambda X^{-1}$$ which is also $$Q \Lambda Q^T$$ coincide with $$A = U \Sigma V^T$$

**Example 2**: If $$A = xy^T$$ (rank = 1) with unit vectors x and y, what is the SVD of A?

The reduced SVD is exactly $$xy^T$$, with rank r = 1. It has $$u_1 = x $$ and $$v_1 = y $$ and $$\sigma_1 =1$$. Full the full SVD, complete $$u_1 = x$$ to an orthonormal basis of u's and complete $$v_1 = y$$ to an orthonormal basis of v's. No new $$\sigma$$'s, only $$\sigma_1 = 1$$

**Example 3**: Find the matrices $$U, \Sigma, V $$ for $$A = \begin{bmatrix}3 & 0 \\ 4 & 5 \end{bmatrix}$$, The rank r = 2

$$A^TA = \begin{bmatrix} 25 & 20 \\ 20 & 25 \end{bmatrix} \space \space \space \space \space AA^T = \begin{bmatrix} 9 & 12 \\ 12 & 41 \end{bmatrix}$$

Those have the same trace 50. The eigenvalues are $$\sigma_1^2 = 45, \sigma_2^2 = 5$$. The eigenvectors of $$A^TA$$ are $$\begin{bmatrix} 1  \\ 1 \end{bmatrix}, \begin{bmatrix} -1  \\ 1 \end{bmatrix} $$ **Right singular vectors** (orthonormal) are $$v_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \space v_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$. 两种方法求 **left singular vectors** 1. 求eigenvector of $$AA^T$$ 2. 用$$u_i = \frac{Av_i}{\sigma_i}$$

$$Av_1 = \frac{3}{\sqrt{2}} \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \sqrt{45} \frac{1}{\sqrt{10}} \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \sigma_1 u_1 $$

$$Av_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -3 \\ 1 \end{bmatrix} = \sqrt{5} \frac{1}{\sqrt{10}} \begin{bmatrix} -3 \\ 1 \end{bmatrix} = \sigma_1 u_1 $$

$$U = \frac{1}{\sqrt{10}} \begin{bmatrix} 1 & -3 \\ 3 & -1 \end{bmatrix} \space \space \Sigma = \begin{bmatrix} \sqrt{45} & \\  & \sqrt{5} \end{bmatrix} \space \space V = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}  $$


## Complex Vectors and Matrices

Complex Vectors

1. *The complex conjugate ($$z = a + bi$$) of*  is $$\bar z = a - bi$$
2. $$\bar{z_1} + \bar{z_2} = \bar{z_1 + z_2}$$
3. $$\bar{z_1} \times \bar{z_2} = \bar{z_1z_2} $$, prove $$z_1 \times z_2 = \left(a+ bi \right)\left(c + di \right) = \left(ac - bd\right) + \left(ad + cd\right)i$$, $$\bar{z_1} \times \bar{z_2} = \left(a - bi \right)\left(c - di \right) =  \left(ac - bd\right) - \left(ad + cd\right)i $$
4. When $$Ax = \lambdax$$ and A is real, taking conjugates, then $$\bar{Ax} = \bar{\lambda x} \space => \space A \bar{x} = \bar{\lambda} \bar{x}  $$
5. $$z = a + bi$$, $$z + \bar{z} = 2a $$, $$z \bar{z} = a^2 + b^2$$ are always **real**, $\left(3 + 2i\right) + \left(3 - 2i\right) = 6 $$, and $$\left(3 + 2i\right) *\left(3 - 2i\right) = 9 + 4 = 13 $$
6. In polar system real part is x and imaginary part is y and $$a^2 + b^2 = r^2$$, $$z = a + b i $$  is aslo  $$z = rcos\theta + irsin\theta$$ which equal to $$re^{i\theta}$$, and $$\bar{z} = re^{-i\theta}$$ (*Euler's Formula*)
7. The nth power of $$z = r \left(r cos\theta + i sin \theta \right)$$ is $$z^n = r^n \left(cosn \theta + i sin n \theta \right)$$

<br/><br/>
```python
import numpy as np
#calculate inverse
A = [[1, 1, 3],
     [1, 2, 4],
     [1, 1, 2]]
Ainv = np.linalg.inv(A)

#solve linear system
A = [[4, 6, 2],
     [3, 4, 1],
     [2, 8, 13]]

s = [9, 7, 2]

r = np.linalg.solve(A, s)

#calculate 长度norm
B = np.array(A, dtype=np.float_)
la.norm(B[:, 1])

#Gram-Schmidt process
def gsBasis(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # Loop over all vectors, starting with zero, label them with i
    for i in range(B.shape[1]) :
        # Inside that loop, loop over all previous vectors, j, to subtract.
        for j in range(i) :
            # Complete the code to subtract the overlap with previous vectors.
            # you'll need the current vector B[:, i] and a previous vector B[:, j]
            B[:, i] = B[:,i] - B[:, i]@B[:,j]*B[:,j]
        # Next insert code to do the normalisation test for B[:, i]
        if la.norm(B[:,i]) > verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])     
            
    # Finally, we return the result:
    return B

#dot product @
B[:, i]@B[:,j]


#矩阵乘法 @
A@B 
"""
[1, 2]   [5,  6]  =    [19, 22]
[3, 4]   [7,  8]  =    [43, 50]

A*B 是elementwise 乘法
[1, 2]   [5,  6]  =    [5,  12]
[3, 4]   [7,  8]  =    [21, 32]

"""

#calculate eigenvector and eigenvalue

M = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
vals, vecs = np.linalg.eig(M)
vals
```

注： not include Basis Transformation, least square, function composition, determinant

## PCA 

####  Assumption

Why PCA? 
1. __Some variables can be explained by one variable__.  e.g X1. # of skidding accidents X2. # snow plow expenditures 3. # patients with heat stoke. 上面几个变量都可以用温度表示. All variables depend on a single quantity which does not observe directly
2. __Curse of Dimensionality__: Data in our dataset have lots of dimension: (machine learning) : 真实的 algorithm 也许用不了这么多，就浪费了 memory
3. As dimensionality grows, fewer observations per region

__Assumption__ about dimensions: 
1. Independence: count along each dimension separately. meaning when counting the frequency of $$x_1$$, ignore $$x_2$$
2. smoothness: propagate class counts to neighboring regions 红的附近是红色的， 蓝色附近是蓝色
3. symmetry: invariance to order of dimensions. The order of variables doesn't matter, $$x_1$$ 与 $$x_2$$ 可以对调

Dimensionality reduction 降维 Goal: represent instances with fewer variables
- try to preserve as much structure in the data as possible 
- discriminative: only structure that affects class separability

__Feature selection__: pick a subset of the original dimensions $$x_1, x_2, \cdots, x_n$$ <br/>
__Feature extraction__: construct a new set of dimensions (e.g. linear combination of original attributes)

#### Principal Components Analysis: 

Define a set of principal compenents:
- 1st: direction of the greatest variability(variance) in the data (in which way the data spread out the most), 可以是any line in the space, 不一定是 X, Y
- 2nd: perpendicular to 1st, greatest variability of what's left 
-  ... and so on until d (original dimensionality)

FIrst m << d components become m new dimensions. Change coodinates of every data point to these dimensions

![](\img\post\Linear-Algebra\pic3.png)

Why greatestt variability: <span style="background-color: #FFFF00">to preserve relative distances</span>

![](\img\post\Linear-Algebra\pic4.png)

如果 project 后距离变 (<span style="color: red">relative distance</span> between points)短了 (比如projection 到绿色的e)，会破坏原来结构， 跟original space比两个红点距离变短. 而我们数据结构的假设是: nearby things should predict the same result. 就像 ML， 两个点很接近，predict 结果很 可能一样


#### Principal Components = eigenvectors

Step: 
1. Center the data at zero. $$X_i = X_i - u$$. subtract mean from each attribute <span style="color: red">origin will be at the center of dataset </span>
2. Compute Covaraince matrix $$\sum$$. Covariance 是 indicator wheter 两个变量一起change, or change together in opposite direction, $$cov\left(x_1, x_2 \right) = \frac{1}{n} \sum_{i=0}^{i=n} x_{i1} * x_{i2}$$
- if we use covariance matrix times a vector (can be any vector) in the plane several time, 最后covariance matrix times vector 不会改变方向，只会让vector 变得longer and longer (turns towards direction of variance), 最后不改变方向的vector is the dimension should be picked as greatest variance. 
3. want vectors e which aren't turned. $$\sum e = \lambda e$$, e is the eigenvectors of $$\sum$$, $$\lambda $$ is corresponding eigenvalues. <span style="background-color: #FFFF00">principle components = eigenvectors with largest eigenvalues </span>

__Projecting to new dimensions__

下面prove <span style="color: red"> d: 原来的维度, m: 新的维度, n: 数据总数(number of data) </span>
  
Project on m dimensions with m eigenvectors (unit length) $$e_1, e_2, \cdots, e_m$$ with m biggest eigenvalues. Have original coordinates $$x = \{x_1, x_2 , \cdots, x_d \}$$, want the new coordinates $$x'= \{x'_1, x'_2,\cdots, x'_m \}$$
- center the instance(subtract the mean): $$x - u$$
- project to each dimension using dot project $$(x-u)^T e_j$$ for j = 1... m ($$scaler = \frac{ \vec x \cdot \vec v  }{ \|\vec v\| } = \vec x \cdot \vec v$$ given $$\|\vec v\| = 1$$):  

$$ \begin{bmatrix} x'_1 \\ x'_2 \\ \vdots \\ x'_m  \end{bmatrix} = \begin{bmatrix} \left(x - \vec u \right)^T \vec e_1 \\ \left(x - \vec u \right)^T \vec e_2 \\ \vdots \\ \left(x - \vec u \right)^T \vec e_m \end{bmatrix} = \begin{bmatrix} \left( x_1 - u_1 \right) e_{1,1} + \left( x_2 - u_2 \right) e_{1,2} + \cdots + \left( x_d - u_d \right) e_{1,d}  \\ \left( x_1 - u_1 \right) e_{2,1} + \left( x_2 - u_2 \right) e_{2,2} + \cdots + \left( x_d - u_d \right) e_{2,d} \\ \vdots \\ \left( x_1 - u_1 \right) e_{m,1} + \left( x_2 - u_2 \right) e_{m,2} + \cdots + \left(x_d - u_d \right) e_{m,d} \end{bmatrix}   $$

把d维projection到m维上, $$x'_i$$ 是对应第i个 eigenvector 的coordiantes, 是number, $$e_{i,j}$$ 是第i个eigenvector 第j维，是number. 所以需要m个eigenvalues 和 m个eigenvectors, 第i个eigenvector是 $$\{e_{i,1}, e_{i,2}, \cdots, e_{i,d}  \}$$

__Prove: projection mean is zero__: 

project 到一个eigenvector上的mean: 利用性质在同一维上原来x平均数为0 $$\frac{1}{n}  \sum_{i=1}^{n} x_{ij} = 0 $$

$$ \begin{align}  u & = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right) \\ & =  \sum_{j=1}^d \left( \frac{1}{n} \sum_{i=1}^{n} x_{ij} \right) e_j \\ & = \sum_{j=1}^d 0 * e_j  = 0 \end{align} $$


__Prove: eigenvector = greatest variance__: 

- eigenvector has unit length (length = 1)
- Select dimension e which maximizes the variance
- Variance of projections (projection mean is zero): 
  
$$\frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j - u \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right)^2$$

we have constraint eigenvector length = 1, then we add Lagrange multiplier

$$ V = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right)^2 - \lambda\left( \sum_{k=1}^d e_j^2  -1 \right)$$

$$ \frac{\partial V}{\partial e_a} = \frac{2}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right) x_{ia} - 2\lambda e_a = 0$$

$$ 2 \sum_{j=1}^{d} e_j \left( \frac{1}{n}  \sum_{j=1}^n x_{ia} x_{ij}  \right)  = 2\lambda e_a $$

$$  \sum_{j=1}^{d} e_j  cov\left( a,j \right)  = \lambda e_a $$

把第 a 行的 covariance matrix 和 ea dot product，得到 eigenvector 的 a 个行, covarince matrix是

$$\begin{Bmatrix} \sum_{i=1}^{d}  cov\left( 1,j \right) e_j = \lambda e_1 \\ \sum_{i=1}^{d}  cov\left( 2,j \right) e_j = \lambda e_2 \\ \vdots \\ \sum_{i=1}^{d}  cov\left( d,j \right) e_j = \lambda e_d  \end{Bmatrix}  =>  \bbox[yellow]{\sum e = \lambda e}$$
 
 
__Prove: eigenvalue = variance along eigenvector__: variane along the projections along eigenvector is eigenvalues

$$ \begin{align}  V = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j - u \right)^2 & = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right)^2 \\ & = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right) \left( \sum_{a=1}^d x_{ia} e_a \right) \\ & = \sum_{a=1}^d  \sum_{j=1}^d \left(  \frac{1}{n}  \sum_{i=1}^{n} x_{ia} x_{ij} \right) e_j e_a  \\ & = \sum_{a=1}^d  \left( \sum_{j=1}^d  cov \left( a, j \right) e_j \right) e_a   \\ & =  \sum_{a=1}^d  \left( \lambda e_a \right) e_a  \\ & = \lambda \|e \|^2 = \lambda \end{align}  $$

__How many dimensions__:

1. 方法一: 
   - Pick $$e_i$$ that explain the msot variance 
   - sort eigenvectors s.t. $$\lambda _1 \ge \lambda _2 \cdots \lambda _d$$
   - pick first m eigenvectors which explain 90% or 95% of total variance (前m个eigenvectors 解释了90% 或 95%的)

2. 方法二：
   -  sort eigenvectors s.t. $$\lambda _1 \ge \lambda _2 \cdots \lambda _d$$
   -  plot eigenvalues as decreasing order 
   -  use a scree plot (pick the stopping point)


![](\img\post\Linear-Algebra\pic5.png)


![](\img\post\Linear-Algebra\pic6.png)

#### PCA Practical Issue 

- <span style="background-color: #FFFF00">Covariance extremly sensitive to large value</span>
   -  multiply some dimension by 1000 (把其中一维的数据都乘以1000), it will affect covariance matrix  e.g 如果人的身高用 micrometers 计算，每个数很大很大，height will become a principal component
   -  <span style="background-color: #FFFF00">solution</span>: normalize each dimension to zero mean and unit variance x' = (x-mean)/st.dev
-	<span style="background-color: #FFFF00">PCA assumes underlying subspace is linear</span>
   -  1 dimension: straight line 
   -  2 dimension: a flat sheet
   -  PCA cannot find manifolds, can only find a single directions, 比如下图，不能发现nonlinear space

![](\img\post\Linear-Algebra\pic7.png)

-  <span style="background-color: #FFFF00">Cannot see the class of classification and only see the dimension</span>. It can be hurt very badly because it doesn't see the class label when decomposition just see the coordinates

![](\img\post\Linear-Algebra\pic8.png)

比如蓝色dimension是PCA给的，但really diseaster, cannot separate them, 因为无法分层，但实际红色dimension更好，但因为PCA，maximize variance，不会选择红色

改进: LDA (linear discriminat analysis)  但LDA not guaranteed to be better for classification.

LDA(linear discriminant analysis): when projected on new direction,  <span style="color: red">LDA maximize the mean difference of class normalized by their variance whereas PCA looks for the dimension of greatest variance</span>. Greater separation between classes

LDA assume:

1. data 是 gaussian 分布，
2. assume have simple boundary between data points,

![](\img\post\Linear-Algebra\pic9.png)

Example where PCA gives a better projection: 

![](\img\post\Linear-Algebra\pic10.png)

#### PCA PROS and CONS

__Pros__:

1.	dramatic reduction in size of data: faster processing, smaller storage
2.	reflects our intuitions about the data

__Cons__:
1.	too expensive for many applications (e.g. bid stream data such as twitter, web)
2.	<span style="color: red">disastrous for tasks with fine-grained classes</span> 
3.	the assuptions behind the methods(<span style="background-color: #FFFF00">linearity of supspace</span>)  



[Detailed LA PDF](https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/linear%20algebra.pdf "file")

[linear Algebra](https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/document/linear%20algebra.pdf "file")


