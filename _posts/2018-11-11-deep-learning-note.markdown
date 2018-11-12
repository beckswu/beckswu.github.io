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

#### Cost Function

$$\mathbf{\text{Loss function: }\mathscr{L} \left(\hat y, y \right) = \bbox[yellow]{ - ylog\left( \hat y \right) + \left( 1- y \right) log\left( 1 - \hat y \right)} }$$

$$\mathbf{\text{Cost function: }} J \left(w, b\right) =  \frac{1}{m} \sum_{i=1}^m \mathscr{L} \left(\hat y^{\left(i\right)}, y^{\left(i\right)} \right) = \frac{1}{m} \sum_{i=1}^m  \ y^{\left(i\right)} log\left( \hat y^{\left(i\right)}  \right) + \left( 1- y^{\left(i\right)} \right) log\left( 1 - \hat y^{\left(i\right)} \right) $$


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

$$\frac{\partial a }{\partial z} = \frac{1}{1+e^{-z}} \frac{-e^{-z}}{1+e^{-z}} = a * \left( 1-a \right)$$

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
$$\space \space \space \space \space Jt = -y^{\left( i \right)} log a^{\left( i \right)} + \left(1 - y^{ \left( i \right)} \right) log  \left(1 - a^{ \left( i\right)} \right)$$<br/>
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
n: # attributes, m: #training examples;  W: n by 1 matrix; X : n by m matrix (every columns is one training example); b: 1 by m matrix (with the same number)<br/>
$$Z = W^T X + b = np.dot\left(w.T, x\right) + b$$<br/>
$$A = \sigma\left( Z \right)$$<br/>
$$dZ = A - Y $$<br/>
$$dW = \frac{1}{m} X dZ^T$$ <br/>
$$db =  \frac{1}{m} \sum_{i=1}^m dz^{\left( i\right)} = \frac{1}{m} np.sum\left(dZ \right)$$<br/>
$$w := w - \alpha dw; \space  b:= b - \alpha db$$ <br/>
------------