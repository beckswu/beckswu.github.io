---
layout:     post
title:      "Deep Learning —— Sequence Model 笔记"
subtitle:   ""
date:       2018-09-12 19:00:00
author:     "Becks"
header-img: "img/about-bg.jpg"
catalog:    true
tags:
    - Coursera
    - Deep Learning
    - RNN
---

## Week1 Recurrent Neural Networks
Examples of sequence data:
                
1. Speech Recognition: given audio clip X ----> text Y (both input and output sequence data)
2.  music generation:  output is sequence(音乐), input maybe music 的类型 or nothing
3. sentiment classfication: 像影: "there is nothing to like in this move" ---> 这样的review是几分
4. DNA sequence Analysis:  given DNA AGCCTGA... ---> label which part of DNA sequence corresponds to a protein
5. machine translation
6. video activty recogntion : 给一段录像 -> 在running
7. Name entity recognation：给一段话 --> 挑出人名

sometimes 输入X 和 输出Y 可以是不同的长度，sometimes X和Y(example 4,7)是同样长度的, sometimes 只有X或者只有Y是sequence的 (example 2)
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
#### notation: 
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic1.png)
example: given a sentence 判断哪个是人名<br/> 
$$x^{({i})<{t}>}$$:  表示第i个training example 中第t个word, t 表示temporal sequences althought whether sequences are temporal one or not<br/> 
$$y^{({i})<{t}>}$$:  表示第i个training example 中第t个word的输出label<br/> 
$$T_x^{i}$$:  表示第i个training example的长度<br/> 
$$T_y^{i}$$:  表示第i个training example的ouput长度<br/> 


**representing words:** <br/>

use dictionary and give each word an index, </br>
$$x^{<{t}>}$$:  是one hot vector, 比如字典的长度是10000, x = apple, apple出现在字典的100位, $$x^{<{t}>} = \begin{bmatrix}
    0 \\
    \vdots \\
    1  \\
	\vdots\\
    \end{bmatrix}
$$ vector长度是10000， 只有第100位是1，剩下都是0. if 遇见了word不在字典中，create a new token or a new fake word called unknown word

比如下面看是不是name的，output是长度为9，0代表不是name, 1代表是name
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic2.png)

#### Recurrent Neural Network Model:
<span style="background-color: #FFFF00">Why not a standard network?</span> <br/>
problems:
1. Input, output can be different lengths in different example (不是所有的input的都是一样长度)
2. Doesn't share features learned across different positions of text(也许word Harry在位置1，但是也许Harry也许出现在位置7)

在time 0, have some eith made-up activation or 全部是0的vector. <br/>
step 1: Take a word(first word) to a neural network layer, then try to predict if this word is name or not. <br/>
step 2: 到了第二个位置, instead of predicting y2 using only x2, it aslo gets some input 从step 1. Deactivation value from step 1 被pass 到了step 2. <br/>The activation parameters (vertical的, $$W_{ax}$$, 用x得到a like quantity) used in each step are shared. Activation (horizontal的,$$W_{aa}$$) is the same. $$W_{ya}$$ (用x得到y like quantity) 控制governs the output prediction

![][pic3]

<span style="background-color: #FFFF00">
One weakness: only use information that is earlier in the sequence to make a prediction （Bidirection RNN (BRNN) 可以解决这个问题）
</span>

Forward Propagation:

$$\begin{align} a^{<{0}>} &= \vec0  \\
a^{<{1}>} &= g_1\left(W_{aa}\cdot a^{<{0}>}+ W_{ax}\cdot X^{<{a}>} + b_aa \right) \\
y^{<{1}>} &= g_2\left(W_{ya}\cdot a^{<{1}>} + b_y \right)
\end{align}$$ 

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic4.png)

简化符号
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic5.png)




<br/> activations function often use tanh or Relu. if it is binary classification, can use sigmoid function. 
<span style="background-color: #FFFF00">The choice of activation 取决于what type of output y you have </span>

[pic3]: https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep_Learning-Sequence_Model_note/week1pic3.png