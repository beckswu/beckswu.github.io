---
layout:     post
title:      "Convolutional Neural Networks"
subtitle:   "coursera note"
date:       2019-07-12 20:00:00
author:     "Becks"
header-img: "img/post-bg-city-night.jpg"
catalog:    true
tags:
    - Deep Learning
    - Machine Learning
    - 学习笔记
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


## 1.1 Convolution Neural Networks




Computer Vision **Challenge**: the input can be really big, e.g if a image `1000*1000` pixel，Each pixel is controlled by three **rgb** channel，image input is `1000*1000*3`. Input for Neural Network model is 3 million, and the first hidden layer(e.g. fully-connected network)  is 1000 hidden units, the weight will be `(1000, 3 Million)`, 3 billion parameters. <span style="background-color:#FFFF00">It is difficult to get enough data to avoid overfitting and the computational requirement and memory requirement to train is infeasible</span>


#### Edge Detection

**Vertical Edge Detection**

convolve image by filter (Kernel) 

e.g. Vertical edge detection

![](/img/post/cnn/week1pic1.gif)

Why above filter works for vertical detection?  e.g. the image(left matrix), left half give brigther pixel intensive and right half give darker pixel intensive values.

![](/img/post/cnn/week1pic2.png)


Above detected edge (value 30, 2 columns) seems very thick, because we use only small iamges. If you are using 1000 by 1000 image rather than 6 and 6, it does pretty good job

**Inituition**: vertical edge detection, since example use 3 by 3 region where <span style="color:red">bright pixels on the left and dark pixels on the right (don't care about what's in middle)</span>

![](/img/post/cnn/week1pic3.png)

-30 example, could take absolute values of output matrix. But this filter does make the difference between light to dark vs dark to light


**Horizontal Edge Detection**

<span style="background-color:#FFFF00">light on top and dark on the bottom row</span>

![](/img/post/cnn/week1pic4.png)

30 is the edge that light on top and dark on bottom and -30 is the edge that dark on top and light on bottom. -10 reflect that parts of positive edge on the left and parts of negative edge on the right, so blending those together gives some *intermediate value*. <span style="color:red">But if a image is 1000x1000, won't see these transitions regions of 10s. The intermediate values would be quite small relative to the size of iamge.</span>


```Python
# Conv-forward
tf.nn.conv2d #TensorFlow
Conv2D       #Keras
```

#### Filter

**Sobel filter**: <span style="background-color:#FFFF00">advantage: put more weight in the middle(2, -2), make it a little bit more robust
</span>
$$    \begin{bmatrix}
    1 & 0 & -1 \\
    2 & 0 & -2 \\
    1 & 0 & -1 \\
    \end{bmatrix}
$$

**Scharr filter**:

$$    \begin{bmatrix}
    3 & 0 & -3 \\
    10 & 0 & -10 \\
    3 & 0 & -3 \\
    \end{bmatrix}
$$

<span style="color:red">can filp above 90 degree to get horizontal edge detection</span>


也许不用hand pick those number，让computer 自己学

$$    \begin{bmatrix}
    w_1 & w_2 & w_3 \\
    w_4 & w_5 & w_6 \\
    w_7 & w_8 & w_9 \\
    \end{bmatrix}
$$
<span style="color:red">The goal: give a image, convolve it with 3x3 filter, that gives a good edge detector. It may learn something even better than hand coded filter.</span> 通过neural network backprop，也许不是vertical的，也许是 45度的，70度的，不是完全vertical, horizontal



#### Padding

if image is `n x n` the filter is `f x f` and  the output is ` n-f+1 x n-f+1`

<span style="background-color:#FFFF00">Downside of filter</span>: 

1. Image will shrink if performing convolution neural networks (to (n-f+1)*(n-f+1) ) 
2. Corner pixel only be used once(e.g. upper-left, upper-right), but middle pixel used by multiple times, <span style="color:red">throw away a lot of information near the edge of the image</span>
  
**Solution**: <span style="background-color:#FFFF00">pad the image by additional border</span>. e.g. `6 x 6` pad to `8 x 8`, filter is `3 x 3`, then the output is `6 x 6`

Denote `p = padding amount`, above example `6 x 6` to `8 x 8`, `p = 1`, padding on top, left, bottom, right by 1


- <span style="background-color:#FFFF00">**Valid convolutions**</span>:  no padding:   `n x n * f x f = (n-f+1) x (n-f+1)`    
- <span style="background-color:#FFFF00">**Same convolutions**</span>:  Pad so that output size is the same as the input size. `(n+2p - f+1) x (n+2p - f+1) = n x n` =>  `p = (f-1)/2`. so when filter is `3x3`, `p=1`, when fitler is `5x5`, `p=2`



<span style="color:red">Filter size `f` usually be odd
nu convention in computer vision</span>. If `f` is even, you will come up asymmetrix padding, Besides, when have odd number of padding, you will have a central position in the middle for the filter.  


#### Strided Convolution

![](/img/post/cnn/week1pic5.gif)


$$\text{n x n } \times  \text{ f x f}   =   \lfloor \frac{n + 2p -f}{s} + 1 \rfloor  \times  \lfloor  \frac{n + 2p -f }{s} + 1 \rfloor $$  

round down to the nearest integer if fraction is not integer (Floor). For above example, `(7 + 0 - 3)/2 + 1  = 3`

If after padding, the the filter box hangs outside of image, don't do that computation

![](/img/post/cnn/week1pic6.png)


#### Convolutional Operation

The operation done before is called <span style="color:red">**cross-correlation**</span> instead of convolution operation. 

By convention in machine learning, 通常忽略 flipping operation. The operation done before btter called cross-correlation, but most of deep learning literature called it convolutional operater(without flip)

<span style="color:red">The convolution before product and summing is to flip filter horizontal and vertically. Then use flipped filter to compute element wise product and summation </span>e.g.

$$    \begin{bmatrix}
    3 & 4 & 5 \\
    1 & 0 & 2 \\
    -1 & 9 & 7 \\
    \end{bmatrix}
$$

to 

$$    \begin{bmatrix}
    7 & 9 & -1 \\
    2 & 0 & 1 \\
    5 & 4 & 3 \\
    \end{bmatrix}
$$

Convolution satisfy **associativity**: A convolve B convolve C equal to A convolve (B convolve C)

$$\left( A * B \right) * C = A * \left( B * C \right) $$

#### Convolutions Over Volume

input is `6 x 6 x 3` where 3 is the color channel(red, green, blue), the filter is `3 x 3 x 3`. Denote first demension is height, second demension is width, third dimension is channel(*in literature, some people called it depth*). <span style="color:red">The number of channel in image must match the channel of the filter</span>


above example, filter size is `3 x 3 x 3`, element wise product for image and filter, for the first channel, second channel, third channel one by one. then add those 27 number together as output

![](/img/post/cnn/week1pic7.png)

e,g1, if you want to detect vertical edges in the red channel,

$$    \underbrace{\begin{bmatrix}
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    \end{bmatrix}}_{red}, 
   \underbrace{\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \end{bmatrix}}_{green},
    \underbrace{\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \end{bmatrix}}_{blue},
$$

e.g2 If want to detect edges, then could use

$$    \underbrace{\begin{bmatrix}
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    \end{bmatrix}}_{red}, 
   \underbrace{\begin{bmatrix}
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    \end{bmatrix}}_{green}, 
    \underbrace{\begin{bmatrix}
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    \end{bmatrix}}_{blue}, 
$$

If want to detect both horizontal and vertical edges, use multiple filters at the same time to convolve the image. Then the first output at the front, and second output at back


![](/img/post/cnn/week1pic8.png)

Summary:    `n*n*n_c(the number of channel)    *    n*n*n_c    =  (n – f + 1) * (n – f + 1) * n_c’  (n_c’ is the number of filter we used)`, assumed stride one and no padding

#### Convolutional Network

After convolve using filter, add a real number bias to every number in the output then apply non-linearity (e.g. Relu)

![](/img/post/cnn/week1pic9.png)

Notice: filter is the $$W$$

Q: if have 10 filters that are `3x3x31 in one leayr of a neural network, how many parameters that layer have

A: each filter has `3x3x3 = 27` filter plus one bias, so 10 filter has `28x10 = 280` parameters

#### Notation

if layer l is a convolution layer:

$$ f^{\left[ l \right]} = \text{filter size} $$  

$$ p^{\left[ l \right]} = \text{padding} $$ 

$$ s^{\left[ l \right]} = \text{stride} $$  

$$ n_c^{\left[ l \right]} = \text{number of filters} $$  


Input: $$n_c$$ is the number of, use superscript l-1 because that's the activation from previous layer, H and W denotes height and width

$$ n_H^{\left[ l-1 \right]} \times  n_W^{\left[ l-1 \right]} \times  n_c^{\left[ l - 1 \right]}$$  

$$ n_H^{\left[ l \right]} \times  n_W^{\left[ l \right]} \times  n_c^{\left[ l  \right]}$$

$$ \text{where the height and width: } n^{\left[ l \right]}=   \lfloor \frac{n^{\left[ l-1 \right]} + 2p^{\left[ l \right]} -f^{\left[ l \right]}}{s^{\left[ l \right]}} + 1 \rfloor $$  

$$ \text{Each filter is } f^{\left[ l \right]} \times  f^{\left[ l \right]} \times  n_c^{\left[ l-1  \right]}$$  

$$\text{where }n_c^{\left[ l-1  \right]} \text{ last layer's number of channel}$$ 

$$\text{Activations: } a^{\left[ l \right]} -> n_H^{\left[ l \right]} \times  n_W^{\left[ l \right]} \times  n_c^{\left[ l  \right]} \text{or some write} a^{\left[ l \right]} -> n_c^{\left[ l \right]} \times  n_H^{\left[ l \right]} \times  n_W^{\left[ l  \right]}   $$ 

$$\text{Weights}: f^{\left[ l \right]} \times  f^{\left[ l \right]} \times  n_c^{\left[ l-1  \right]} \times  n_c^{\left[ l  \right]}  $$ 

$$\text{bias: }  n_c^{\left[ l-1  \right]}, \text{in program, deminsion write: } \left(1,1,1,n_c^{\left[ l  \right]} \right) $$

If using batch gradient descent or mini batch gradient descent:

$$\text{Activations: } A^{\left[ l \right]} -> m \times n_H^{\left[ l \right]} \times  n_W^{\left[ l \right]} \times  n_c^{\left[ l  \right]}, \text{m examples}  $$


An example of ConvNet

![](/img/post/cnn/week1pic10.png)

Input : `39 x 39 x 3` image 

$$
$\require{AMScd}$
\begin{CD}
    \text{Image 39 x 39 x 3} @>{f^{\left[ 1 \right]} = 3, s^{\left[ 1 \right]} = 1 }> {p^{\left[ l \right]} = 0}, \text{10 filters}> \text{37 x 37 x 10}  \\
    @. @V{f^{\left[ 2 \right]} = 5, s^{\left[ 2 \right]} = 2 }V {p^{\left[ 2 \right]} = 0}, \text{20 filters}V \\
    \text{7 x 7 x 40}@<{f^{\left[ 3 \right]} = 5, s^{\left[ 3 \right]} = 2 }< {p^{\left[ 3 \right]} = 0}, \text{40 filters}< \text{17 x 17 x 20}  \\
@VV{\text{unroll it}}V  @. \\
\text{1960 vectors} @>{\text{logistic regression}}>{\text{or softmax unit}}> output 
\end{CD}
$$

role out ` 7 x7 x 40 = 1960 ` to unroll 1960 units into a long vector and feed into a logistic regression unit or a softmax unit 

Lots of work in designing convolutional neural net <span style="background-color:#FFFF00">is selecting hyparameters like deciding what's total size? filter size? padding? how many filter are use? </span>

<span style="background-color:#FFFF00">Typically, start with a large image, **then height and width will stay the same for a while and gradually trend down as go deeper in the neural network whereas the number of channel general increase**</span>


#### Pooling Layer

**Pooling**:  reduce the size of the representation to speed computation as well as make some of the features that detects a bit more robust. 

**Max Pooling**: Intuition: If these features detected anywhere in this filter, then keep a high number. If feature not detected, then maybe this feature doesn't exist in this quadrant. People Found a lot of experiences to work well.

![](/img/post/cnn/week1pic11.png)

**Property**: Above example, hyparameter filter size = 2(because take 2 by 2 region) and stride = 2, <span style="background-color:#FFFF00">but has no parameter to learn</span>. Once fix f and s, just a fixed computation and gradient descent doesn't change anything

If have 3D input, max pooling is done independently on each of channel one by one and stack output together

**Average Pooling**: Instead of taking max of each filter, take the average of each filter. <span style="color:red">not used often compared Max Pooling</span>. <span style="background-color:#FFFF00">One exception: Very deep neural network, you might use average pooling to collapse representation</span>. e.g. (`7 x 7 x 1000` to `1 x 1 x 1000`)

![](/img/post/cnn/week1pic12.png)



Max or average pooling.   或者有时候 (very very rare ) add padding , most of time max pooling p = 0,   no parameter to learn! 

Common choice of f = 2, s = 2, has the effect of shrinking the height and with of representation by factor of two. Some use f = 3, s = 2

The formula still works. Since pooling applies each of channel independently, the number of input channel match the number of output channel

$$n_H \times n_W \times n_c  =   \lfloor \frac{n + 2p -f}{s} + 1 \rfloor  \times \lfloor  \frac{n + 2p -f }{s} + 1\rfloor \times n_c  $$  

#### CNN Example

example inspired by LeNet-5

![](/img/post/cnn/week1pic13.png)



|         | Activation shape           | Activation Size  | num of parameters |
| :-------------: |:-------------:| :-----:|:-----:|
| Input     | (32, 32,3) | 3072  | 0 |
| CONV1(f=5, s = 1)  | (28,28,8)      |  6272  | 608 `((5*5*3+1)*8)` |
| POOL1 |   (14,14,8)   | 1568  | 0 |
| CONV2(f=5, s=1) | (10,10,16) | 1600 | 3216 `(5*5*8+1)*16` |
| POOL2 | (5,5,16) | 400 | 0 |
| FC3 | (120, 1) | 120 | 48120 `400*120+120`(120 bias) | 
| FC4 | (84, 1) | 84 | 10164 `120*84 + 84` | 
| Softmax | (10, 1) | 10 | 850 |


When report number of layer, people usually report the number of layers that have weight, that have parameters. Beacuse pooling layer has no weights and no parameters only has hyperparameters, so put ConV and pooling layer together in convention. 

**Guideline**: not to invent your own hyperparameter, <span style="color:red">but look into literature to see what hyper parameters you work for others</span>, just choose a architecture that works well for others and there is a chance that works well for you as well. 

**Common pattern** <span style="background-color:#FFFF00">one or more conv layers followed by a pooling layer and in the end have a few fully connected layers then followed by a softmax </span>

Notice:

1. pooling layer has no parameters
2. <span style="color:red">Conv layer tend to have a few parameters and lots of parameters tend to be in the fully connected layer</span>
3. <span style="color:red">Activation size tend of go down gradually as go deeper in the neural network. If drop too quickly, not great for performance as well
</span>

#### Why Convolutions


Two main advantage of using Conv layer instead of fully connected layer

- **parameter sharing**: A feature detector (such as vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image (因为parameter 少了，<span style="color:red">allowed to train a smaller training set and less proned to overfitting)</span>. 
  - e.g. apply 3 by 3 filter on the top-left of the image and apply the same filter on top-right of the image. 
  - True for low-level features like edges as well as high-level features like detecting the eye that indicates a face or a cat
- **sparsity of connections** : In each layer, each output value depends only on a small number of inputs.  比如filter 是 3*3， output 第1行1个只取决于 input 的top left 3*3 parameter，不取决于 第一行第四个或者第五个值
- Convolution neural network aslo very good at capturing **translation invariance**（即使原来图片发生一点点位移，还是原来feature) e.g 比如一只猫shift couple of pixels to right 仍是猫. And convolutional structure helps that shifted a few pixels should result pretty similar feature. Apply the same filter on the image helps to be more robust to caputre the desirable property of translation invariance

e.g. 

$$
\require{AMScd}
\begin{CD}
    \underbrace{\text{Image 32 x 32 x 3}}_{3072} @>{f = 5}>{\text{6 filters}}> \underbrace{\text{28 x 28 x 6} }_{4074} 
\end{CD}
$$

- A fully-connected layer: the number of parameters is `3072 x 4074 = 14 million`
- Convolutional layer: `(5 x 5 + 1 ) x 6  = 156 ` parameters


Cost function: Use gradient descent or gradient descent momentum, RMSProp or Adams to optimize parameters to reduce J

$$J = \frac{1}{m} \sum_{i=1}^m L\left(\hat y^{\left(i \right)},  y^{\left(i \right)}\right),\text{where }, \hat y \text{ is the predicted label and y is true label} $$