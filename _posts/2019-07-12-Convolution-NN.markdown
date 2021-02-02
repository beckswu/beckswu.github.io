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


## 1 Convolution Neural Networks




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
\require{AMScd}
\begin{CD}
    \text{Image 39 x 39 x 3} @>{f^{\left[ 1 \right]} = 3, s^{\left[ 1 \right]} = 1,p^{\left[ l \right]} = 0, \text{10 filters} }>> \text{37 x 37 x 10}  \\
    @. @V{f^{\left[ 2 \right]} = 5, s^{\left[ 2 \right]} = 2,p^{\left[ 2 \right]} = 0, \text{20 filters}}VV \\
    \text{7 x 7 x 40}@<{f^{\left[ 3 \right]} = 5, s^{\left[ 3 \right]} = 2, p^{\left[ 3 \right]} = 0, \text{40 filters}}<< \text{17 x 17 x 20}  \\
@VV{\text{unroll it}}V  @. \\
\text{1960 vectors} @>{\text{logistic regression or softmax unit}}>> output 
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

<span style="background-color:#FFFF00">If have 3D input, max pooling is done independently on each of channel one by one and stack output together(每个channel分开max pooling, 然后stack together)</span>

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
    \underbrace{\text{Image 32 x 32 x 3}}_{3072} @>{\text{f = 5, 6 filters}}>> \underbrace{\text{28 x 28 x 6} }_{4074} 
\end{CD}
$$

- A fully-connected layer: the number of parameters is `3072 x 4074 = 14 million`
- Convolutional layer: `(5 x 5 + 1 ) x 6  = 156 ` parameters


Cost function: Use gradient descent or gradient descent momentum, RMSProp or Adams to optimize parameters to reduce J

$$J = \frac{1}{m} \sum_{i=1}^m L\left(\hat y^{\left(i \right)},  y^{\left(i \right)}\right),\text{where }, \hat y \text{ is the predicted label and y is true label} $$

<br/><br/><br/>

## 2. Classic Networks

#### LeNet-5 

Last year is softmax layer although back then, LeNet-5 used a different classifier at the output which is useless today.

![](/img/post/cnn/week2pic1.png)

Above model has 60k parameters. Today, often see a network with 10 million to 100 million parameters. When you go deeper with your network, the height and weight tend to go down and the number of channel tend to increase. 

Pattern often used today: 

$$
\require{AMScd}
\begin{CD}
    \text{one or more Conv Layer} @>>> \text{Pooling Layer} @>>> \text{one or more Conv Layer} @>>> \text{Pooling Layer} \\
@. @. @. @VVV \\
@. \text{output} @<<<  \text{fully-connected layer} @<<< \text{fully-connected layer}
\end{CD}
$$


(Optional): In paper, use sigmoid and tanh in the paper (no ReLu used at that time). Besides, in paper, use sigmoid non-linearity after the pooling layer(not used today)


#### Alexnet


Alexnet has a **lot of similarity to Lenet but it was much bigger**. Lenet has around 60 thousand parameters whereas Alexnet has 60 million parameters. 

![](/img/post/cnn/week2pic2.png)

- Compared to LeNet, the fact that they could take similar building block, but a lot more hidden units and training on a lot of more data to  allow it to have a remarkable performance
- Another aspects make it much better than letnet is <span style="color:red">using Relu activation function </span>
- had a complicated way of training on two GPUs. A lot of ConV and pooling layer split across two different GPUs and  a thoughtful way for when two GPUs communicate with each others.
- **Local Response Normalization**(not use too much): 方法: 一个block 是 `13*13*256`，比如把第一维是5， 第二维是6的上所有的256 的数，normalize 这256个数据
  - Motivation: maybe you don’t want for each position in 13 by 13 image，you don’t want too many neurons with a very high activation. But subsequently, some researcher 发现local response normalization less important(Andrew Ng doesn't use it)

#### VGG-16 

Motivation: instead of having so many hyperparameters, use a much simpler network where  focus on just having <span style="color:red">ConV layers</span> that are just <span style="color:red">3 by 3 filters with stride one</span>, and <span style="color:red">always use the same padding</span>. And make all your <span style="color:red">max pulling layers 2 by 2 with stride of two</span>; it simplify this neural network architectures.  

<span style="background-color:#FFFF00">**Downside**: large netowrk in terms of the number of parameters you train</span>

![](/img/post/cnn/week2pic3.png)

- VGG-16: 16 layers have weights
- This network has 138 million parameters, large network
- The architecture is really quite uniform, a few ConV layers followed by a polling layer whereas the pooling layers reduce the height and weight. Number of channel from 64 到128， 到256，到512(512 is big enough, not need to double)， <span style="background-color:#FFFF00">roughly doubling on every step or doubling through every stack of ConV layers is another principle used to desgin the architecture of this network</span>

Some literature mention VGG-19. It's even bigger version of this network.

<br/><br/><br/>

## 3. ResNets

**ResNets**:  Very deep neural Networks are difficult to train because <span style="color:red">**gradient vanishing/exploding**</span> problem. ResNets enables you to train very very deep networks (sometimes even over 100 hundred layers)

Residual Blocks : take $$a^{\left[l \right]}$$ fast foward it, copy it to much further into the neural networks,<span style="background-color:#FFFF00">inject after the linear part but before perform nonlinear function (ReLu)</span>: called it  **Shortcut(skip connection)** using 

<span style="background-color:#FFFF00">Residual blocks allow you train much deeper neural network</span>. The way to build ResNet is by taking many of these **residual blocks** and stacking them together to form a deep network

![](/img/post/cnn/week2pic4.png)

$$
\require{AMScd}
\begin{CD}
    @>{\text{short cut}}>> --> @>{\text{short cut}}>> --> @>{\text{short cut}}>>  a^{\left[l \right]}  \\
    
    @AAA @. @. @VVV \\
    a^{\left[l \right]} @>{ w^{\left[l+1 \right]}a^{\left[l \right]} + b^{\left[l+1 \right]}}>> Linear: z^{\left[l+1 \right]} @>{g\left(z^{\left[l+1 \right]} \right)}>> ReLu: a^{\left[l+1 \right]} @>{ w^{\left[l+2 \right]}a^{\left[l+1 \right]} + b^{\left[l+2 \right]}}>> Linear: z^{\left[l+2 \right]} @>{g\left(z^{\left[l+2 \right]} + a^{\left[l \right]} \right)}>> ReLu: a^{\left[l+2 \right]}
\end{CD}
$$

if use optimization algorithm such as gradient descent to train a plain neural network, without extra shortcuts or skip connections, training error like below graph. <span style="color:red">But in reality your training error get worse if you pick a network that’s too deep</span>. 

<span style="background-color:#FFFF00">For ResNet is that even as the number of layers gets deeper, having the performance of training error keeping going down, even train over a hundred of layers. It **helps with the vanishing and exploding problems**. It allows  train much deeper neural networks without appreciable loss in performance </span>

![](/img/post/cnn/week2pic5.png)

#### Why work?

For example, if have a network below, we have Relu and assume a is bigger than or equal to 0.

$$
\require{AMScd}
\begin{CD}
    @. @. @>{\text{short cut}}>> --> @>{\text{short cut}}>>  a^{\left[l \right]}  \\
    
    @. @. @AAA @.  @VVV \\
    x @>>> BigNN @>>> a^{\left[l \right]} @>>> layer1 @>>> layer2 @>>> a^{\left[l+2 \right]}
\end{CD}
$$

then 

$$ \begin{align}
  a^{\left[l + 2 \right]} &= g\left(  z^{\left[l + 2 \right]} + a^{\left[l \right]} \right) \\
&=  g\left(  w^{\left[l + 2 \right]}a^{\left[l + 1 \right]} + b^{\left[l + 2 \right]} + a^{\left[l \right]} \right) 
\end{align}
$$


If you use L2 regularization to shrink $$w^{\left[l + 2 \right]}$$ , then if you apply weight to b it will also shrink weight to b) although in practice don’t apply regularization weight to b, 

$$
  a^{\left[l + 2 \right]} = g\left( a^{\left[l \right]}  \right) = a^{\left[l \right]} 
 
$$ 


<span style="background-color:#FFFF00">Note: we assume $$z^{\left[l + 2 \right]}  $$ and $$a^{\left[l \right]} $$ have the same dimension. **Use the same convolution**</span>

But if they don't have the same dimension, add $$W_s$$, e.g. $$z^{\left[l + 2 \right]} $$ 256 维, $$a^{\left[l \right]} $$ 128 维, then $$W_S$$ is 256 by 128. <span style="color:red">$$W_S$$ could a parameter to learn or a fixed matrix that just implement zero paddings (padding 128 to 256</span>)

$$a^{\left[l + 2 \right]} = g\left(  z^{\left[l + 2 \right]} + W_Sa^{\left[l \right]} \right)$$


 It shows the <span style="background-color:#FFFF00">because of skip connection, it's so easy for residual networks to learn identity function, guarantee adding residual blocks doesn't hurt neural network performance</span>, it's easy to get $$a^{\left[l + 2\right]} $$ equal to $$a^{\left[l \right]} $$. Hidden units if actually learn something useful then can do even better than learning the identity function

 In plain network without Residual Network, when you make the network deeper and deeper, it is very difficult for it to choose parameters that learn even the identity function, which a lot of layer end-up  worse.

#### Resnet on Image


 ![](/img/post/cnn/week2pic6.png)

 - lots 3 by 3 convolutions. And most of them are 3 by 3 same convolution instead of fully-connected layer . That's why $$z^{\left[l + 2 \right]} + a^{\left[l \right]} $$ make senses
 - After pooling layer, need to adjust dimension, liked discussed above by $$W_S $$
 - In the end, have a fully connected layer that makes a prediction using a softmax

<br/><br/><br/>


## 3. 1x1 Convolution

sometimes 1x1 Convolution called **Network in Network**

What does it do in the below example: look at each of the 36 different positions, and take the <span style="background-color:#FFFF00">element-wise product between 32 numbers on the input</span> (same height and same width)<span style="background-color:#FFFF00"> and 32 numbers in the filter, add together(summation) then apply Relu non-linearity</span>(32 -> 1 number).

One way to think of 1 by 1 convolution is bsically having a fully connected neural network that applies to each of 36 different position

 ![](/img/post/cnn/week2pic7.png)

#### Use of 1x1 Convolution

-	Shrink the number of  channel（save computation）: usually  pooling is used to shrink height and weight.  而1*1 convolution 是 比如input 是`28*28*192`，`1*1` dimension convolution 是 `1*1*192`  32 filters，output is `28*28*32` <span style="background-color:#FFFF00">which allow shrink $$n_c$$ as well whereas Pooling layer only shrink $$n_H$$ and $$n_W$$</span>
- If want to keep the number of Channel, <span style="background-color:#FFFF00">1 by 1 convolution add linearity to allow to learn more complex function of your network by adding another layer that inputs </span>, 接着上面例子, filter size is `1x1x192` and have 192 filters, output will be `28*28*192`

<br/><br/><br/>


## 4. Inception Network

Instead of choosing to filter size, or choose convolutional layer or pooling layer, let use them all together, stack(concatenate) all output \ together 


e.g. image input is `28 x 28 x 192`,

  1. use 64 filters with size `1 x 1 x 192`, get `28 x 28 x 64`
  2. use 128 filters with size `3 x 3 x 192`, same convolution, get `28 x 28 x 128`
  3. use 32 filters with size `5 x 5 x 192`, same convolution, get `28 x 28 x 32`
  4. max pooling and 1 by 1 convolution, get `28 x 28 x 32`, need padding to get the same height and width
  5. Stack previous output together

 ![](/img/post/cnn/week2pic8.png)

 A problem: <span style="color:red">Computational Cost</span>

e.g. 32 filters with each size `5 x 5 x 192`, output size is `28 x 28 x 32`, each output is needed to calculate `5 x 5 x 192` multiplications. Total number of multiplications is `28 x 28 x 32 x 5 x 5 x 192 = 120 million`, expensive operation. <span style="color:red">Could use 1 by 1 convolution to reduce the computational cost</span>s by a factor of 10

  ![](/img/post/cnn/week2pic9.png)

Use 1 by 1 convolution, can see output dimension is the same. First shrink to much smaller intermediate volume `28 x 28 x 16` called **bottleneck layer**


  ![](/img/post/cnn/week2pic10.png)

  Computation Cost: Cost of first convolutional layer `28 x 28 x 16 x 1 x 1 x 192 = 2.4 million`. `28 x 28 x 32 x 5 x 5 x 16 = 10 million`. total is `12.4 million << 120 million`


1. Apply 64 filters of 1 by 1 convolution, get `28 x 28 x 64` output
2. Apply 96 filters of 1 by 1 convolution, then apply 128 `5 x 5` filters to get `28 x 28 x 128` output
3. Apply 96 filters of 1 by 1 convolution, 
4. For above example pooling,<span style="color:red">in order to concatenate all of outputs at the end, use the same type of padding for pooling.</span>. the output will be `28 x 28 x 192`. Then apply one more 1 by 1 convolutional layer to shrink the number of channel to `28 x 28 x 32`
5.  In the end, channel concatenation

 ![](/img/post/cnn/week2pic11.png)

What the inception network does is to put all inception block together. 下面的inception network 是多个上面的repeated inception block stack在一起. 红色小箭头是max pooling to change the height and width

Inception network 有**side branch**，takes a hidden layer to pass through a few fully-connected layer, then has a softmax to make prediction， It is said even in intermediate(hidden) layer 加上几层用softmax, not so bad to predict outcomes. It also <span style="background-color:#FFFF00">has regularization effect, prevent network overfitting (避免训练network too deep) </span>

 ![](/img/post/cnn/week2pic12.png)


<br/><br/><br/>

## 5. Advices for ConvNets

#### Transfer Learning


Sometimes, training takes several weeks and might take many GPU. Can download open source weight that took someone else many weeks or months as a good initialization for your own neural network.

**When don't have enough training data**:

download Github open-source implementation and weight, <span style="color:red">then get rid of softmax layer and create your own softmax unit. Take early stage layers and parameters as frozon. And train the parameters associated with your softmax layer</span>. Then might get good performance even with a small dataset

Some deep learning framework can let you specify whether to train parameter or freeze parameters.

**Trick**: because of all of early layers are frozon, Use some fixed function to take a image to the activation in the layer which you begin to train. <span style="color:red">Can pre-compute that layer(image 和所有froozen layer 到开始训练的layer) and save to disk</span>. Then train a shallow softmax model to make a prediction <br/> 
<span style="background-color:#FFFF00">**Advantage**</span>: don't need to recompute the activations everytimes

**If have a large training set**:

- freeze a fewer layer, then train latter layers and create the your own output unit.
-  Or delete these latter layers and just use your own new hidden units and your own softmax output. 
-  Or train some other size neural networks(different size of hidden layers)  that comprise last layer of your own softmax output

<span style="background-color:#FFFF00">When having more data, the number of layers freeze could be smaller and the number of layers trained could be greater.</span>

If have lots of data, take whole things as initialization(to replace random initialization) and train the whole network and update all the weights of all the layers of the network by gradient descent

 ![](/img/post/cnn/week2pic13.png)

 Transfer learning is something should always do unless have exceptionally large data and very large computation budge

#### Data Augmentation

- **Mirroring**: 比如一个猫的图片左右颠倒下 and Mirroring preserves what is in the picture 
- **Random Cropping**:  随机选取图片的一部分剪切下来，当做新的，random cropping <span style="color:red">isn’t the perfect way</span> for data augmentation. In practice, it works well.
- Some other method: **rotation**, **shearing**(正方形变平行四边形), local warping: no harm with trying of all these things. But in practice, they seem to be used a bit less. perhaps because of complexity
- **Color shifting**: 比如red 增加 20， green 减小20， 蓝色增加20. <span style="color:red">In practice, R,G, and B are drawn from some probability distribution</span>
  - Motivation: maybe sunlight might a little bit yellow that could easily change the color of the image but the indentity of the content (y) should stay the same
  - Introduce color shifting, make learning algorithm more robust to change in colors of image

 ![](/img/post/cnn/week2pic14.png)


One of the ways to implement color distortion algorithm: **PCA color Augmentation**

What PCA do: 比如 image mainly has red and blue tints, and very little green. PCA color Augmentation will add / subtract a lot for red and blue and very little green so that keeps the overall color the tint the same. 

**Implementing distortions during training**:

if you have large training set:

- have a CPU thread that is <span style="color:red">constantly loading images</span> of your hard disk, then <span style="color:red">implement distortion</span> (random cropping, color shifting, or mirroring) to form a batch or mini-batches of data
- Then batch / mini-batches are constantly are constantly <span style="color:red">passed to</span>  some other thread or process for <span style="color:red">training</span>(CPU or increasingly GPU if have a large neural network to train)
- Above two thread can run in parallel

Data augmentation 有时候还可以 有hyperparameter 去tune，比如 how much color shifting do you implement and exactly what parameters you use for random cropping.(If you think someone else doesn't not captured more in variances in their implementation, it might be reasonable to use hyperparameters yourself)

 ![](/img/post/cnn/week2pic15.png)

#### State of Computer Vision


Image recognition 是给图片告诉是猫还是狗，object detection 是给你图片where in picture 有障碍物（autonomous driving）we tend to have less data for objection detection than for image recognition (because of the cost of getting bounding box to label is more expensive than label the objects)

![](/img/post/cnn/week2pic16.png)

- When have lots of data, Can have a giant neural networks, use simple algorithms as well as less hand-engineering. Less needing to carefully desgin features for the problem to learn whatever to learn if having lots of data
- When don't have too much data, more hand-engineering. Hand-engineering is the best way to get good performance when has less data.

Two sources of knowledge:

- Label data (x, y)
- Hand engineered features/network architecture/other components


Computer Vision is trying to learn really a complex function. We don't have enough data for computer vision. Even datasets are getting bigger and bigger, often we don't have as much data as we need. That's why computer vision relied more on hand-engineering.  That's way computer vision has developed complex architectures becaue of the absence of more data. 即使最近data increase dramatically, results in a significant reduction in amount f hand-engineering. 但是still lots of hand-engineering of network architectures in computer vision compared to other disciplines.

When don't have enough data, hand-engineering is very diffcult, skillful task taht requires a lot of insight. If have lots of data, wouldn't spend time hand-engineering, would spend time to build up the learning system. 


**Tips for doing well on benchmarks**:
	
- **Ensembling**: Run several neural network independently(3-15 network) and average their output (average $$\hat y$$, not average weight which won’t work) 因为速度会降很多倍，所以用它来win in competition not use in actual production to serve customer 同时ensembling 因为是run 了很多network，会用掉很多memory to keep all these network around (Computational expensive)
- **Multi-crop at test time**: Run classifier on multiple versions of test image and average results. Multi-crop is applying data agumentation to your test image. It might get a little bit better performance in a production system.
  - 一张图片随机选取其中好几个部分作为test sample，每一个sample运行network through classifier，average results

Andrew Ng: 以上方法是research 中提到的，不建议用in production or a system that deploy in an actual application

建议： 

- Use open-source code implementations if possible 
- Use architectures of networks published in the literature 
- use pretrained models and fine-tune on your dataset to get faster  on an application(transfer learning: someone else may train a model on half dozen GPU and over a million images)
