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

> note from youtube khan Academy

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
 $$ \left( A^T \right)^{-1} =  \left( A^{-1} \right)^T  $$  <br/>
 $$ det\left( A^{-1} \right) =  det\left( A \right)^{-1}  $$  <br/>

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


## Null Space and Column Space

__Dimension__: the number (cardinality) of a basis of V

#### null space

the __kernel(nullspace)__ of a linear map L : V → W between two vector spaces V and W, is the set of all elements v of V for which L(v) = 0, where 0 denotes the zero vector in W: <br/>
$$ ker\left(L\right) = \{ v \in V | L\left( v \right) \} $$ <br/>
or in $$R^N:  N = \{ \vec x \in R^n | A \vec x =  \vec 0  \} $$  

Null space is valid subspace 满足: <br/>
1. $$ \vec 0 $$ in subspace
2. if $$ \vec v_1, \vec v_2 \in N $$, then  $$ A \left( \vec v_1 +  \vec v_2 \right) = A\vec v_1 + A\vec v_2 = \vec 0 \in N $$
3. if $$ \vec v \in N $$, $$ A \left(c \vec v \right) = c \left( A\vec v \right) \in N $$

__Nullity__: Dimension of Null space = number of free variables <span style="color: red">free variables (non-pivot) </span> in reduced echelon form in Matrix A
$$N\left( A \right) =\left( \vec 0 \right) $$ if and only if column vectors of A are linearly independent (only apply when A is n by n matrix)

#### column space

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
5. A is n by n matrix whose columns form an orthonormal set, then $$A^{-1} = A^T $$ (因为是 n by n, 所以是invertible)
6. If orthonormal basis 组成的matrix A,  then $$Proj_v \vec x = A \left( A^T A \right)^{-1} A \vec x = A A^T \vec x $$
7. If orthonormal basis 组成的matrix A, then when do transformation, it will preserve length and angle for transformation. prove: 


   $$ \| C \vec x \|^2 = C\vec x \cdot C \vec x = \left( C\vec x \right)^T C \vec x = \vec x^T C^T C \vec x =\vec x^T \vec x =  \| C \vec x \|   $$ 

   $$cos\theta =  \frac{ C\vec w \cdot C \vec v }{ \| C\vec v \| \| C\vec v \| } =  \frac{ \left(C\vec w \right)^T C \vec v }{ \| \vec v \| \| \vec v \| } =  \frac{ \vec w^T C^T C \vec v }{ \| \vec v \| \| \vec v \| }  = \frac{ \vec w^T \vec v }{ \| \vec v \| \| \vec v \| } $$

#### Gram-Schmidt process

__Gram-Schmidt process__, is a procedure which takes a nonorthogonal set of linearly independent functions to constructs an orthogonal basis

![](\img\post\Linear-Algebra\pic2.png)

## Eigenvector & Eigenvalue

If T is a linear transformation from a vector space V over a field F into itself(V) and $$\vec v$$ is a not a zero vector, then  $$\vec v$$ is an __eigenvector__ of T if  $$T\left(\vec v\right)$$  is a scalar multile of v. It can be written: 

$$ T\left(\vec v \right) = \lambda \vec v $$

where $$\lambda$$ is a scalar in the field F, known as __eigenvalue__ associated with __eigenvector__ $$\vec v$$

$$ A \vec v = \lambda \vec v $$

$$ A \vec v - \lambda  I_n \vec v = \vec 0 $$

$$ \left( A - \lambda  I_n  \right) \vec v = \vec 0 $$

因为 $$\vec v $$ is non-zero vector, $$ \left( A - \lambda  I_n  \right) $$ 必须是linear dependent matrix, 否则只有trivial soluion {0}, so $$ det\left( A - \lambda  I_n  \right) = 0 $$ 


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
  
Project on m dimensions with m eigenvectors (unit length) $$e_1, e_2, \cdots, e_m$$ with m biggest eigenvalues. Have original coordinates $$x = \{x_1, x_2 , \cdots, x_d \}$$, want the new coordinates $$x'= \{x'_1, x'_2,\cdots, x'_m \}$$
- center the instance(subtract the mean): $$x - u$$
- project to each dimension using dot project $$(x-u)^T e_j$$ for j = 1... m ($$scaler = \frac{ \vec x \cdot \vec v  }{ \|\vec v\| } = \vec x \cdot \vec v$$ given $$\|\vec v\| = 1$$):  

$$ \begin{bmatrix} x'_1 \\ x'_2 \\ \vdots \\ x'_m  \end{bmatrix} = \begin{bmatrix} \left(x - \vec u \right)^T \vec e_1 \\ \left(x - \vec u \right)^T \vec e_2 \\ \vdots \\ \left(x - \vec u \right)^T \vec e_m \end{bmatrix} = \begin{bmatrix} \left( x_1 - u_1 \right) e_{1,1} + \left( x_2 - u_2 \right) e_{1,2} + \cdots + \left( x_d - u_d \right) e_{1,d}  \\ \left( x_1 - u_1 \right) e_{2,1} + \left( x_2 - u_2 \right) e_{2,2} + \cdots + \left( x_d - u_d \right) e_{2,d} \\ \vdots \\ \left( x_1 - u_1 \right) e_{m,1} + \left( x_2 - u_2 \right) e_{m,2} + \cdots + \left(x_d - u_d \right) e_{m,d} \end{bmatrix}   $$

把d维projection到m维上, $$x'_i$$ 是对应第i个 eigenvector 的coordiantes, 是number, $$e_{i,j}$$ 是第i个eigenvector 第j维，是number

__Prove: projection mean is zero__: 

$$ \begin{align}  u & = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right) \\ & =  \sum_{j=1}^d \left( \frac{1}{n} \sum_{i=1}^{n} x_{ij} \right) e_j \\ & = \sum_{j=1}^d 0 * e_j  = 0 \end{align} $$

__Prove: eigenvector = greatest variance__: 

- eigenvector has unit length (length = 1)
- Select dimension e which maximizes the variance
- Variance of projections (projection mean is zero): 
  
$$\frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j - u \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right)^2$$

we have constraint eigenvector length = 1, then we add Lagrange multiplier

$$ V = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right)^2 - \lambda\left( \sum_{k=1}^d e_j^2  -1 \right)$$

$$ \frac{\partial V}{\partial e_a} = \frac{2}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^d x_{ij} e_j \right) x_{ia} - 2\lambda e_a = 0$$

$$ 2 \sum_{i=1}^{d} e_j \left( \frac{1}{n}  \sum_{j=1}^n x_{ia} x_{ij}  \right)  = 2\lambda e_a $$

$$  \sum_{i=1}^{d} e_j  cov\left( a,j \right)  = \lambda e_a $$

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


