---
layout:     post
title:      "Linear Algebra - Summary "
subtitle:   "Linear Algebra 线性代数 总结"
date:       2018-09-23 20:00:00
author:     "Becks"
header-img: "img/post/math.jpg"
catalog:    true
tags:
    - Linear Algebra
    - 学习笔记
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
vector projection: &nbsp;&nbsp;&nbsp; $$  \vec proj_{L} \left(\vec x\right) =  c \vec v =  \frac{ \vec x \cdot \vec v  }{ \vec v \cdot \vec v } \vec v $$<br/>
two vector $$  \vec  x $$ and $$  \vec y $$ are orthorgonal: &nbsp;&nbsp;&nbsp; $$  \vec  x \cdot \vec y  = 0 $$   <br/>
(Optional) : $$  \vec a \left( \vec b \times \vec c \right)  =  \vec b \left(\vec a \cdot \vec c) - \vec c \left(\vec a \cdot \vec b)  $$

cross product: <br/>
 $$  \vec a =  \begin{bmatrix} a_1 \\ a_2  \\ a_3 \\ \end{bmatrix}, \vec b =  \begin{bmatrix} b_1 \\ b_2  \\ b_3 \\ \end{bmatrix}, \vec a \times \vec b  = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \\ \end{bmatrix} $$<br/>

<span style="background-color: #FFFF00">Dot product tells: product of lengths of vectors move together at same direction with b.</span> When 𝑎⃗ ∙ 𝑏^⃗ = 0, perpendicular, 𝑎⃗ onto 𝑏^⃗ is zero
<span style="background-color: #FFFF00">Cross product tells: product of lengths of vectors move perpendicular direction with b.</span> When 𝑎⃗ ∙ 𝑏^⃗ = ñ|𝑎⃗|ñ °ñ𝑏^⃗ñ°,
perpendicular, 获得最大值, 当 a 和 b colinear, 𝑎⃗ ∙ 𝑏^⃗ = 0 no perpendicular vector


注: <span style="color: red">difference between perpendicular and orthogonal</span>:  $$  \vec  x \cdot \vec y  = \vec 0 $$   only means orthogonal, zero vector is orthogonal to everything but zero vector not perpendicular to everything; perpendicular is orthogonal, 但是othogonal 不一定是perpendicular, 因为 $$  \vec  x \cdot \vec 0  = 0 $$ 不是perpendicular

求basis的coordinate的时，若basis vector orthogonal to each other, 可以用scaler projection, 看$$  \vec x$$到$$  \vec v_i$$的projection $$ \frac{ \vec x \cdot \vec v  }{ \|\vec v\|^2 } $$ 即是coordinate

#### basis

Basis is a set of n vectors that 
- are not linear combinations of each other (linear independent): <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for all $$ a_1, …, a_n \subseteq F $$, if $$ a_1v_1 + … + a_nv_n = 0 $$, then necessarily  $$ a_1 = … = a_n = 0;  $$
- span the space: <br/> 
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; every (vector) $$\vec x$$ in V it is possible to choose $$ a_1, …, a_n \subseteq  F $$ such that $$ x = a_1 \vec v_1 + … + a_n \vec vn. $$
- The space is then n-dimensional 

## INVERSE

 $$ A =  \begin{bmatrix} a & b  \\ c & d \\ \end{bmatrix}, A^{-1} =  \frac{ 1  }{ det\left(A\right) } \begin{bmatrix} d & -c  \\ -b & a \\ \end{bmatrix} = \frac{ 1  }{ ad - bc } \begin{bmatrix} d & -c  \\ -b & a \\ \end{bmatrix}  $$

多维的inverse 可以row operation, 需要inverse matrix 在左面，indentity matrix在右侧，把左侧的matrix通过row operation变成indentiy matrix，一样的row operation 也apply 在右侧，<span style="color: red">最后得到右侧的matrix</span>就是inverse 

$$ \left[ \begin{array}{ccc \| ccc}  1 & 0 &1 &1 & 0 &0  \\ 0 & 2 & 1 &0 & 1 &0  \\ 1&1&1 &0 & 0 &1 \end{array} \right]  =>  \left[ \begin{array}{ccc\|ccc}  1 & 0 &0 & -1 & -1 &2   \\  0 & 1 &0 & -1 & 0 & 1  \\ 0 & 0 &1 & 2&1&0  \end{array} \right] $$

A square matrix is __not invertible__ is called __singular__ or **degenerate**. A square matrix is singular if and only if its __determinant__ is 0. <span style="background-color: #FFFF00">Non-square matrices</span> do not have inverse.

#### properties
 $$ \left( A^{-1} \right)^{-1} = A  $$ <br/>
 $$ \left( kA \right)^{-1} = k^{-1} A^{-1}  $$ for nonzero scalar K <br/>
 $$ \left( A^T \right)^{-1} =  \left( A^{-1} \right)^T  $$  <br/>
 $$ det\left( A^{-1} \right) =  det\left( A \right)^{-1}  $$  <br/>

 if A be a square n by n matrix is invertible, following must be all true of all false
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

## TRANSPOSE

 $$ \left( A^T \right)^{T} = A  $$ <br/>
 $$ \left( A + B \right)^T = A^T + B^T $$ for nonzero scalar K <br/>
 $$ \left( AB \right)^{T} =  B^T A^T  $$   <br/>
 $$ \left( cA \right)^T = cA^T  $$ for scalar c <br/>
 $$ det\left( A^{T} \right) =  det\left( A \right)  $$  <br/>
 $$ a \cdot b = a^Tb   $$ for column vectors a and b  <br/>

if $$  A^T = A $$, then A is called __symmetric matrix__

## Linear Independence and Span

a set of vectors is __linearly dependent__ if one of the vectors in the st can be defined as a linear combination of the others. 比如$$ \begin{bmatrix} 2 \\ 3 \\ \end{bmatrix}, \begin{bmatrix} 4 \\ 6 \\ \end{bmatrix} $$ is linear dependent  If no vector in the set can be written in this way, then vectors are __linearly independent__

n vectors in $$R^N$$ are __linearly independent__ if and only if the <span style="background-color: #FFFF00">determinant </span>of the matrix formed by the vectors as its column is non-zero. ($$det\left(A \right)\neq 0 $$)

$$span \left( v_1, v_2, \cdots, v_n \right) = \{ c_1v_1 + c_2v_2 +\cdots + c_nv_n \| c_i \in R  \}  $$ The space of all of the combination of vectors $$v_1, v_2, ...,v_n$$

## Subspace 

if V is subspace of $$R^N$$
1. The zero vector 0 is in V
2. if  $$ \vec x $$ in V, any scaler c, then the scalar product $$ c\vec x $$ is in V (closure under scaler multiplication)
3. if  $$ \vec a $$ in V and $$ \vec b $$ in V , then $$ \vec a + \vec b $$ also in V (closure under addition)

![](/img⁩/post⁩/Linear_Algebra/pic1.png)

![](https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Linear_Algebra/pic1.png)


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

[file](https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/Data-Structure-and-Algorithms.pdf "file")

[linear Algebra](https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/document/linear%20algebra.pdf "file")


<object data="https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/img/CSharp.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/img/CSharp.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/img/CSharp.pdf">Download PDF</a>.</p>
    </embed>
</object>




<embed src="" width="500" height="375">
