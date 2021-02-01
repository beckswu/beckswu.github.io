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


$$\text{n x n  *   f x f}   =   \lfloor \frac{n + 2p -f}{s} + 1 \rfloor  \text{ x } \lfloor  \frac{n + 2p -f }{s} + 1 \rfloor $$  

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