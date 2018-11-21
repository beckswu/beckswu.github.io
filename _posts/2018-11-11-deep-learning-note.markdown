---
layout:     post
title:      "Deep Learning Summary"
subtitle:   "深度学习 Deep Learning - Note"
date:       2018-11-11 12:00:00
author:     "Becks"
header-img: "img/post-bg-city-night.jpg"
catalog:    true
tags:
    - Coursera
    - Deep Learning
    - Machine Learning
    - 总结
    - 学习笔记
---

> note from Coursera Deep Learning

| Input(x)  | Output(y)   |  application  | Type of Neuron Network  |
| ------------ | ------------ | ------------ | ------------ |
| Home Features  | Price   | Real Estate   | Standard Neural Network   |
| Ad, User Info  | Click on ad(0,1)   | Online Advertising   | Standard Neural Network   |
| Image | Object(1,...,1000)  | Photo Tagging  | Convolution NN  |
| Audio | Text Transcript  | Speech recognition  | Recurrent NN  |
| ENglish | Chinese  | Machine Translation  | Recurrent NN  |
| Image, Radar Info | Position of cars  | Autonomous Driving  | Custom / Hybrid NN  |

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


**Structured Data**: Database data: 比如housing price prediction 输入值是size, #bedroom, 输出值是 price
**Unstructure Data**: 比如audio, image, Text

## Standard Neural Network

#### Notation

**Layer** 分为Input layer, hidden layer(not observed), 和 output layer <br/>
**Activation** 每一层的input也是上一层的output, $$a^{\left[0 \right]} = X$$, $$a_2^{\left[1 \right]} $$, 第一层的第二个input, 也是第0层的第二个output <br/>
当数neural network 时候不算input layer， 所以当 $$y = a^{\left[2 \right]}$$, 一个input layer, 一个hidden layer, 一个output layer时候，被称为2 layer NN

m 是 number of training example, n 是 number of features

**For Logistic Regression**
x 是 n by m 维, <br/>
W 是 n by 1 维 <br/>
b 是 1 by 1 维， <br/>
$$z = W^T x + b $$ 是 1 by m 维

$$z = w^Tx + b = \left[ w_1, w_2, \cdots, w_n  \right] \begin{bmatrix} \mid & \mid & \cdots & \mid \\ x_1 & x_2 & \cdots & x_m \\ \mid & \mid & \cdots & \mid \end{bmatrix} + \left[ b\right] $$

**For Neural Network**

用$$\mathbf{\left[ i \right]}$$ 表示第i层, 用$$\mathbf{\left( i \right)}$$ 表示第i个training example <br/>
$$\mathbf{W^{\left[ i \right]}}$$ 表示 从第i-1 层到第i层的paramter <br/>
X ($$a^{\left[ 0 \right]}$$) 是 n by m 维, <br/>
A ($$a^{\left[ 0 \right]}$$) 是 #$$a_i$$ by m 维, 每列是每一个training example 每一行是different nodes <br/>
W 是 # $$a_{i}$$ by #$$a_{i-1}$$ 维, 表示从第i-1 层 到 第i层的参数。 # $$a_{i}$$是下一层layer的neuron(node) 数, # $$a_{i-1}$$是上一层neuron(node) 数， W的j行代表从 $$a_{i-1 }$$ 到 $$a_{i}$$ 第j 个 nodes 所有参数   <br/>
b 是 # $$a_{i}$$ by 1 维 <br/>

$$z^{\left[ i \right]} = W^{\left[ i \right]}a^{\left[ i-1 \right]} + b^{\left[ i \right]} = \begin{bmatrix} ---W_1^{\left[ i \right]T}--- \\ ---W_2^{\left[ i \right]T}--- \\ \vdots \\ ---W_n^{\left[ i \right]T}--- \end{bmatrix}  \begin{bmatrix} \mid & \mid & \cdots & \mid \\ a_1^{\left[ i-1 \right]} & a_2^{\left[ i-1 \right]} & \cdots & a_m^{\left[ i-1 \right]} \\ \mid & \mid & \cdots & \mid \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}  $$


![](\img\post\Deep-Learning\pic3.png)


#### Cost Function

$$\mathbf{\text{Loss function: }\mathscr{L} \left(\hat y, y \right) = \bbox[yellow]{ - ylog\left( \hat y \right) - \left( 1- y \right) log\left( 1 - \hat y \right)} }$$

$$\mathbf{\text{Cost function: }} J \left(w, b\right) =  -\frac{1}{m} \sum_{i=1}^m \mathscr{L} \left(\hat y^{\left(i\right)}, y^{\left(i\right)} \right) = \frac{1}{m} \sum_{i=1}^m  \ y^{\left(i\right)} log\left( \hat y^{\left(i\right)}  \right) + \left( 1- y^{\left(i\right)} \right) log\left( 1 - \hat y^{\left(i\right)} \right) $$


**Loss function** measures how well your algorithm output $$\hat y^{\left(i \right)} $$ on each of the training examples or compares to the ground true label $$ y^{\left(i \right)}$$ on each of the training examples (loss function是对于一个 training example )

**Cost function measures** how well parameter w and b doing **on the training set** (cost function是对于entire training set )

![](\img\post\Deep-Learning\pic2.png)

Loss (error) function 不用 $$L\left(\hat y, y \right) = \frac{1}{2}\left( \hat y - y\right)^2$$ (<span style="color: red">Optimization is not a convex function having many local optimum</span>, so Grandient Descent may not find global optimum)


#### Gradient Descent

Repeat { <br/>
$$W := w - \alpha \frac{ \partial J\left(w, b\right)}{\partial w} =  w - \frac{1}{m} \frac{\partial \sum_{i=1}^m \mathscr{L} \left(\hat y, y \right)}{\partial w} , \alpha \text{ learning rate} $$<br/>
$$b := b- \alpha \frac{ \partial J\left(w, b\right)}{\partial b} =  b - \frac{1}{m} \frac{\partial \sum_{i=1}^m \mathscr{L} \left(\hat y, y \right)}{\partial b} $$<br/>
}

Gradient Descent $$W := w - \alpha \frac{ \partial J\left(w, b \right)}{\partial w}$$, $$\alpha$$ <span style="color: red">前面是减号的原因</span>

![](\img\post\Deep-Learning\pic1.png)

$$\frac{\partial J \left(w, b \right)}{\partial w} = \frac{ J\left( w + 0.0001, b\right) - J\left(w,b \right)}{\left( w + 0.0001\right) - w}$$

若J(w,b)的值随着w的增加而增加(increasing, slope为正), 找global minimum 就是要w的基础上减小(与slope相反),同理如果J(w,b)的值随着w的增加而减小(decreasing, slope为负), 找global minimum 就是要w的基础上增加

*mathematic proof of gradient*: Using Chain Rule

$$\text{As we know: }z = W^T x + b,  \hat y = a = \sigma\left(z\right) = \frac{1}{1 + e^{-z}}$$

$$\text{As we know: } \mathscr{L} \left(a, y \right) = - ylog\left( a \right) + \left( 1- y \right) log\left( 1 - a \right) $$

$$\frac{\partial \mathscr{L} \left(a, y \right)}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}   $$

$$\frac{\partial a }{\partial z} = \frac{1}{1+e^{-z}} \frac{e^{-z}}{1+e^{-z}} = a * \left( 1-a \right)$$

$$dz = \frac{\mathscr{L} \left(a, y \right)}{\partial a} * \frac{\partial a}{\partial z} = -y*\left(1-a \right) + a\left(1- y \right) = a - y $$

$$\mathbf{dw_1 = \frac{\mathscr{L} \left(a, y \right)}{\partial z} \frac{\partial z}{\partial w1} = x_1 dz = x_1 \left(a - y \right)}$$

$$\mathbf{dw_2 = \frac{\mathscr{L} \left(a, y \right)}{\partial z} \frac{\partial z}{\partial w2} = x_2 dz = x_2 \left(a - y \right)}$$

$$\mathbf{db = \frac{\mathscr{L} \left(a, y \right)}{\partial z} \frac{\partial z}{\partial b} = dz = a - y}$$


#### Logistic Regresion

**Logistic Regression with Gradient Descent on m Examples**

------------
$$i = 0, dw_1 = 0, dw_2 = 0, db = 0, Jt = 0$$<br/>
For i from 1 to m:<br/>
$$\space \space \space \space \space z^{\left( i \right)} = W^T x^{\left( i \right)} + b $$ <br/>
$$\space \space \space \space \space a^{\left( i \right)} = \sigma \left(z^{\left( i \right)} \right)$$<br/>
$$\space \space \space \space \space Jt += -y^{\left( i \right)} log a^{\left( i \right)} - \left(1 - y^{ \left( i \right)} \right) log  \left(1 - a^{ \left( i\right)} \right)$$<br/>
$$\space \space \space \space \space dz^{\left( i \right)} = a^{\left( i \right)} - y^{\left(i \right)}$$<br/>
$$\space \space \space \space \space  dw_1 += x_1^{\left(i\right)} dz^{\left( i \right)}$$<br/>
$$\space \space \space \space \space  dw_2 += x_2^{\left(i\right)} dz^{\left( i \right)}$$<br/>
$$\space \space \space \space \space  db +=  dz^{\left( i \right)}$$<br/>
End Loop<br/>
$$Jt /= m; \space dw_1 /= m; \space dw_2 /= m; \space db /= m$$<br/>
$$w_1 = w_1 - \alpha dw_1; \space w_2 = w_2 - \alpha dw_w; \space b = b - \alpha db;$$

------------

**Vectorizing Logistic Regression with Gradient Descent on m Examples**

------------
n: # attributes, m: #training examples;  W: n by 1 matrix; X : n by m matrix (every columns is one training example, every row is each attributes); b: 1 by m matrix (with the same number)<br/>
$$Z = W^T X + b = np.dot\left(w.T, x\right) + b$$, Z is 1 by m matrix<br/>
$$A = \sigma\left( Z \right)$$ , A is 1 by m matrix<br/>
$$dZ = A - Y $$, dZ is 1 by m matrix<br/>
$$dW = \frac{1}{m} X dZ^T$$ dW is n by 1 matrix<br/>
$$db =  \frac{1}{m} \sum_{i=1}^m dz^{\left( i\right)} = \frac{1}{m} np.sum\left(dZ \right)$$, db is a number<br/>
$$w := w - \alpha dw; \space  b:= b - \alpha db$$ <br/>

------------

<span style="color: red">*Logistic Regression Cost Function 演变*</span>

$$P\left( y \mid x \right) = \hat y ^y \left( 1 - \hat y \right) ^{1-y} \tag{1}\label{eq1}$$

当 y = 1 时候, $$P\left(y \mid x \right) = \hat y$$ <br/>
当 y = 0 时候, $$P\left(y \mid x \right) = 1 - \hat y$$

$$\text{take the log of (1):  } y log \hat y + \left( 1- y \right) log \left(1 - \hat y \right)$$

Loss function 加上负号 因为想要make probability large, we want to minimize loss function, minimize loss function corresponding to maximize the log of the probability

同理: 我们想最大化probability: $$\prod_{i=1}^m p\left( y^{\left(i \right)} \mid x^{\left(i \right)} \right)$$ 也等于最小化 $$- \frac{1}{m} \sum_{i=1}^m log\left( y^{\left(i \right)} \mid x^{\left(i \right)} \right)$$

#### Forward Propagation

**Forward Prop on m Examples**

------------

for i = 1 to m: <br/>
 $$ \space \space \space \space \space   z^{\left[ 1 \right] \left( i \right)} = W^{\left[ 1 \right]} x^{\left( i \right)} + b^{\left[ 1 \right]} $$<br/>
 $$\space \space \space \space \space   a^{\left[ 1 \right] \left( i \right)} = \sigma \left(z^{\left[ 1 \right]} x^{\left( i \right)} \right) $$<br/>
$$ \space \space \space \space \space   z^{\left[ 2 \right] \left( i \right)} = W^{\left[ 2 \right]} a^{\left[1 \right] \left( i \right)} + b^{\left[ 2 \right]} $$<br/>
$$ \space \space \space \space \space   a^{\left[ 2 \right] \left( i \right)} = \sigma \left(z^{\left[ 2 \right]} x^{\left( i \right)} \right) $$<br/>

------------

$$z^{\left[ 1 \right]} = w^{\left[ 1 \right]}x + b^{\left[ 1 \right]}$$

$$z^{\left[ 1 \right]T}  = \begin{bmatrix} ---W_1^{\left[ 1 \right]T}--- \\ ---W_2^{\left[ 1 \right]T}--- \\ ---W_3^{\left[ 1 \right]T}--- \\ ---W_4^{\left[ 1 \right]T}--- \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} + \begin{bmatrix} b_1^{\left[ 1 \right]} \\ b_2^{\left[ 1 \right]} \\ b_3^{\left[ 1 \right]} \\ b_4^{\left[ 1 \right]} \end{bmatrix}   $$ 


**Vectorized Forward Prop on m Examples**

------------

$$Z^{\left[ 1 \right]} = W^{\left[ 1 \right]} X + b^{\left[ 1 \right]} $$<br/>
$$A^{\left[ 1 \right]} = \sigma\left(Z^{\left[ 1 \right]}\right)$$ <br/>
$$Z^{\left[ 2 \right]} = W^{\left[ 2 \right]} A^{\left[ 1 \right]}  + b^{\left[ 2 \right]} $$<br/>
$$A^{\left[ 2 \right]} = \sigma\left(Z^{\left[ 2 \right]}\right)$$

------------

$$z^{\left[ 1 \right]} = w^{\left[ 1 \right]}x + b^{\left[ 1 \right]}$$

$$z^{\left[ 1 \right]}  = \begin{bmatrix} ---W_1^{\left[ 1 \right]T}--- \\ ---W_2^{\left[ 1 \right]T}--- \\ ---W_3^{\left[ 1 \right]T}--- \\ ---W_4^{\left[ 1 \right]T}--- \end{bmatrix} \begin{bmatrix} \mid & \mid & \cdots & \mid \\ a_1^{\left[ i-1 \right]} & a_2^{\left[ i-1 \right]} & \cdots & a_m^{\left[ i-1 \right]} \\ \mid & \mid & \cdots & \mid \end{bmatrix} + \begin{bmatrix} b_1^{\left[ 1 \right]} \\ b_2^{\left[ 1 \right]} \\ b_3^{\left[ 1 \right]} \\ b_4^{\left[ 1 \right]} \end{bmatrix}   $$ 


#### Activation Function

**1. Sigmoid Function**

$$a = \frac{1}{1 + e^{-z}}$$

$$\text{derivatives: } dz = g\left(z\right) * \left(1 - g\left(z\right) \right) = a * \left( 1- a\right)$$


当z趋近于10, $$g\left(z \right) = 1, dz = 1 * \left( 1- 1\right) \approx 0$$ <br/>
当z趋近于-10, $$g\left(z \right) \approx 0, dz = 0 * \left( 1- 0\right) \approx 0$$<br/>
当z趋近于0, $$g\left(z \right) = \frac{1}{2}, dz = \frac{1}{2} * \left( 1- \frac{1}{2} \right) = \frac{1}{4}$$

**2. Tanh Function**

$$g\left(z \right) = \frac{ e^z - e^{-z}}{e^z + e^{-z}}$$

$$\text{derivatives: } dz = 1 - \left( tanh \left(z \right) \right)^2$$

$$\text{derivatives of ReLu: } dz = \begin{cases}  0 & \text{if z < 0}    \\ 1 & \text{ if z } \ge \text{0} \end{cases}$$

$$\text{derivatives of Leaky ReLu: } dz = \begin{cases}  0.01 & \text{if z < 0}    \\ 1 & \text{ if z } \ge \text{0} \end{cases}$$

当z趋近于10, $$tanh\left( z \right) \approx 1,  dz \approx 0$$ <br/>
当z趋近于-10, $$tanh\left( z \right) \approx -1,  dz \approx 0$$ <br/>
当z趋近于0, $$tanh\left( z \right) = 0,  dz = 1$$ 

tanh performance always <span style="color: red"> better than </span> sigmoid function. <span style="background-color: #FFFF00">Tanh function the mean of activation close to 0. And when normalized data, 也有zero mean.  </span>.

Ng Suggestion: 不再使用sigmoid function， 都在使用tanh function.<span style="background-color: #FFFF00"> 一个exception 用sigmoid function 是在output layer，因为output layer 想要0 或者 1, 而不是1 or - 1 </span>

<span style="color: red">**Downside**</span>: 当z非常大或者小的时候, gradient 会变得非常小, 会slow down gradient descent(因为slope every small)

**3. ReLu Function** (default choice for hidden unit)

$$\text{ReLu: }a = max\left(0,z \right) \space \space \text{Leaky ReLu: } a= max\left(0.01 z, z \right)$$

Technically derivative is not well defined  when z is 0. But when implement in computer, you get z is smaller number( 0.000000001). 自己用的时候，可以pretend derivatives either 1 or 0

<span style="color: red">**Advantage**</span>: the slope of the activation(gradient) function different from 0. Using ReLu or Leaky ReLu, <span style="background-color: #FFFF00">neural network will learn **much faster** than when using the hanh or sigmoid activation function</span> (不像tanh or sigmoid 当z很大很小时候,slope 很小, slow down learning)


<span style="color: red">**Downside**</span>: when z is negative, derivatives is zero. But in practive, enough of your hidden units will have z greater than 0. So learning can still be quite fast for most training examples


![](\img\post\Deep-Learning\pic4.png)

<span style="color: red">**Why we need activation function**</span>: 如果不用的话, no matter how many layer you use, output is linear function of input, always computing linear activation functions, hidden layer 就没有用了. **Linear hidden layer is useless**. 

**One Exception using linear activation function**: <span style="color: red">regression problem</span>. (e.g. predicting housing price), output layer is linear activation function but hidden layer 用ReLu function, output 也可以用relu function (因为价格都大于0）

#### Back-prop Proof

Define Cost is C and  gradient as $$\nabla_{ij}^{\left[ l\right ]} = \frac{\partial C}{\partial w_{ij}^{\left[ l\right ]}}$$ <br/>
Define $$\delta_{ij}^{\left[ l\right ]}  = \frac{\partial C}{\partial z_{ij}^{\left[ l \right]}} $$<br/>
use $$w_{ij}^{\left(l\right)}$$ means the parameters from layer l-1's i-th node to layer l's j-th node

First show $$\nabla_{ij}^{\left[ l\right ] } = \delta_{i}^{\left[ l\right ]} * a_{j}^{\left[ l-1\right ]} $$ with $$\delta_i^{\left[ l\right ]} = \frac{\partial C}{\partial z_{i}^{\left[ l\right ]}} $$

$$\nabla_{ij}^{\left[ l\right ]} = \frac{\partial C}{\partial z_{i}^{\left[ l\right] }} = \sum_k \frac{\partial C}{\partial z_{i}^{\left[ l\right ]}} \frac{\partial z_{i}^{\left[ l\right ]}}{ \partial w_{ij}^{\left[ l \right] } }$$

$$\text{we know: } z_{k}^{\left[ l\right] } = \sum_m w_{km}^{\left[ l\right ]} * a_{m}^{\left[ l-1\right] } $$

$$\frac{\partial  z_{k}^{\left[ l\right] } }{ \partial  w_{ij}^{\left[ l\right] } } = \frac{\partial}{\partial  w_{ij}^{\left[ l\right] } } \sum_m w_{km}^{\left[ l\right] } *  a_{m}^{\left[ l-1\right]} = \sum_m \frac{\partial  w_{km}^{\left[ l\right]} }{ \partial  w_{ij}^{\left[ l \right]}  } *  a_{m}^{\left[ l-1\right]}  $$

$$\text{if k}  \neq \text{ i and  m } \neq \text{ j: } \frac{\partial  w_{km}^{\left[ l\right]} }{ \partial  w_{ij}^{\left[ l \right]}  } *  a_{m}^{\left[ l-1\right]} = 0 $$

$$\text{if k = i and m = j: } \frac{\partial  w_{km}^{\left[ l\right]} }{ \partial  w_{ij}^{\left[ l \right]}  } *  a_{m}^{\left[ l-1\right]} = \frac{\partial  w_{ij}^{\left[ l\right]} }{ \partial  w_{ij}^{\left[ l \right]}  } *  a_{j}^{\left[ l-1\right]} $$

$$\frac{\partial z_i^{\left[ l \right]}} {\partial w_{ij}^{\left[ l \right] }  } = \frac{\partial w_{ij}^{ \left[ l \right]}  } {\partial w_{ij}^{\left[ l \right]}  } *a_{j}^{\left[ l-1 \right]} + \sum_{m \neq j} \frac{\partial w_{im}^{\left[ l \right]}  } { \partial w_{ij}^{\left[ l \right]}  } *a_{m}^{\left[ l-1 \right]} = a_{j}^{\left[ l-1 \right]} + 0 = a_{j}^{\left[ l -1 \right]}   $$


$$\nabla_{ij}^{\left[ l\right ]} = \sum_k \frac{\partial C} {\partial z_k^{\left[ l \right]}}  \frac {\partial z_k^{\left[ l \right]}} {\partial w_{ij}^{\left[ l \right]} } = \frac{\partial C} {\partial z_i^{\left[ l \right]}}  \frac {\partial z_i^{\left[ l \right]}} {\partial w_{ij}^{\left[ l \right]} } = \frac{ \partial C} {\partial z_i^{\left[ l \right]}} * a_j^{\left[ l - 1 \right]}$$


#### Back Propagation


**Vectorized Back Prop on m Examples**

------------

$$dZ^{\left[ 2 \right]} = A^{\left[ 2 \right]} -  Y; \space \space \text{  Y }, dZ^{\left[ 2 \right]}  \text{ is 1 by m matrix} $$<br/>
$$dW^{\left[ 2 \right]} = \frac{1}{m} dZ^{\left[ 2 \right]}  A^{\left[ 1 \right] T} $$ <br/>
$$db^{\left[ 2 \right]} = \frac{1}{m} np.sum\left(  dZ^{\left[ 2 \right]}, \text{ axis = 1, keepdims = true} \right)$$ <br/>
如果不加keepdims 可能产生 np.array funny (n, )array, 加上keepdims = true 给出shape = $$\left( n^{\left[ 2 \right]} , 1\right)$$ <br/>
$$dZ^{\left[ 1 \right]} = W^{\left[ 2 \right]T}   dZ^{\left[ 2 \right]} \cdot  g^{\left[ 1 \right} \left( Z^{\left[ 1 \right] \right) $$<br/>
$$W^{\left[ 2 \right]T}   dZ^{\left[ 2 \right]} $$ is $$\left( n^{\left[ 1 \right]}, m \right) matrix, g^{\left[ 1 \right]} \left( Z^{\left[ 1 \right] \right) $$ is also a $$\left( n^{\left[ 1 \right]}, m \right) matrix $$<br/>
$$dW^{\left[ 1 \right]} = \frac{1}{m} dZ^{\left[ 1 \right]}  X^{T} $$<br/>
$$db^{\left[ 1 \right]} = \frac{1}{m} np.sum\left(  dZ^{\left[ 1 \right]}, \text{ axis = 1, keepdims = true} \right)$$

------------
