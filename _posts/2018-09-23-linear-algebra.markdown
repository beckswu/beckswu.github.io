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


## Properties: 
#### vector

<span style="background-color: #FFFF00">property: </span>

Communicative: &nbsp;&nbsp;&nbsp;  $$ \vec v \cdot \vec w  =  \vec w \cdot \vec v $$ <br/>
Distributive: &nbsp;&nbsp;&nbsp;  $$ \left(\vec w + \vec v \right)\cdot \vec x  =  \vec w \cdot \vec x + \vec v \cdot \vec x  $$<br/>
Associative over scaler multiplication: &nbsp;&nbsp;&nbsp; $$ \left( c \vec v \right) \cdot \vec w  =  c \left( \vec v \cdot \vec w \right)  $$<br/>
Dot product self is the length square: &nbsp;&nbsp;&nbsp; $$ c\vec v \cdot \vec v  =  ||v||^2 = v_1^2 + v_2^2 + ... + v_n^2  $$<br/>
Cosine: &nbsp;&nbsp;&nbsp; $$  \vec a \cdot \vec b  =  ||a||^2||b||^2 cos\theta  $$<br/>
scaler projection: &nbsp;&nbsp;&nbsp; $$  \vec proj_{L} \left(\vec x\right) =   \frac{ \vec x \cdot \vec v  }{ ||\vec v|| } $$<br/>
vector projection: &nbsp;&nbsp;&nbsp; $$  \vec proj_{L} \left(\vec x\right) =  c \vec v =  \frac{ \vec x \cdot \vec v  }{ \vec v \cdot \vec v } \vec v $$<br/>
two vector $$  \vec  x $$ and $$  \vec y $$ are orthorgonal: &nbsp;&nbsp;&nbsp; $$  \vec  x \cdot \vec y  = 0 $$ 


注: <span style="color: red">difference between perpendicular and orthogonal</span>:  $$  \vec  x \cdot \vec y  = \vec 0 $$   only means orthogonal, zero vector is orthogonal to everything but zero vector not perpendicular to everything; perpendicular is orthogonal, 但是othogonal 不一定是perpendicular, 因为 $$  \vec  x \cdot \vec 0  = 0 $$ 不是perpendicular

求basis的coordinate的时，若basis vector orthogonal to each other, 可以用scaler projection, 看$$  \vec x$$到$$  \vec v_i$$的projection $$ \frac{ \vec x \cdot \vec v  }{ \|\vec v\|^2 } $$ 即是coordinate

#### basis

Basis is a set of n vectors that 
- are not linear combinations of each other (linear independent): <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for all $$ a_1, …, a_n \subseteq F $$, if $$ a_1v_1 + … + a_nv_n = 0 $$, then necessarily  $$ a_1 = … = a_n = 0;  $$
- span the space: <br/> 
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; every (vector) $$\vec x$$ in V it is possible to choose $$ a_1, …, a_n \subseteq  F $$ such that $$ x = a_1 \vec v_1 + … + a_n \vec vn. $$
- The space is then n-dimensional 

## Inverse:

 $$ A =  \begin{bmatrix} a & b  \\ c & d \\ \end{bmatrix}, A^-1 =  \frac{ 1  }{ det\left(A\right) } \begin{bmatrix} d & -c  \\ -b & a \\ \end{bmatrix} = \frac{ 1  }{ ad - bc } \begin{bmatrix} d & -c  \\ -b & a \\ \end{bmatrix}  $$

多维的inverse 可以row operation, 需要inverse matrix 在左面，indentity matrix在右侧，把左侧的matrix通过row operation变成indentiy matrix，一样的row operation在apply 在右侧，最后得到右侧的matrix就是inverse 

$$ \left[ \begin{array}{ccc|ccc}  1 & 0 &1 &1 & 0 &0  \\ 0 & 2 & 1 &0 & 1 &0  \\ 1&1&1 &0 & 0 &1 \end{array} \right] $$ => $$ \left[ \begin{array}{ccc|ccc}  1 & 0 &0 & -1 & -1 &2   \\  0 & 1 &0 & -1 & 0 & 1  \\ 0 & 0 &1 & 2&1&0  \end{array} \right] $$

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


<object data="https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/linear%20algebra.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/linear%20algebra.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://docs.google.com/viewer?url=http://nbviewer.jupyter.org/github/beckswu/beckswu.github.io/blob/master/document/linear%20algebra.pdf">Download PDF</a>.</p>
    </embed>
</object>

<embed src="" width="500" height="375">
