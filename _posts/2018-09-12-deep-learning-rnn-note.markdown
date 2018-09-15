---
layout:     post
title:      "Deep Learning —— Sequence Model 笔记"
subtitle:   "深度学习 Deep Learning —— Sequence Model note"
date:       2018-09-12 19:00:00
author:     "Becks"
header-img: "img/post/Deep_Learning-Sequence_Model_note/bg.jpg"
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

[//]: <> (![](/img/post/Deep_Learning-Sequence_Model_note/week1pic1.png))


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
\end{align}$$ </br>
从$$a^{<{t-1}>} $$和 $$x^{<{t}>}$$ 生成$$a^{<{t}>}$$ 的可以是<span style="color: red">tanh</span>, 从$$a^{<{t}>}$$ 到$$y^{<{t}>}$$的是<span style="color: red">softmax</span>

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic4.png)

简化符号
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic5.png)

<br/> activations function often use tanh or Relu. if it is binary classification, can use sigmoid function. 
<span style="background-color: #FFFF00">The choice of activation 取决于what type of output y you have </span>


#### Backpropagation through time

Single Loss Function: $$ L^{<{t}>}\left( \hat y^{<{t}>},  y^{<{t}>} \right) = - y^{<{t}>}log \left( \hat y^{<{t}>} \right) - 
    \left( 1- y^{<{t}>} \right) log\left(1- \hat y^{<{t}>} \right) $$<br/>

Overall Loss Function:  $$ L \left(  \hat y , y \right) =  \displaystyle \sum_{t=1}^{T_x} {L^{<{t}>} \left( \hat y^{<{t}>}, y^{<{t}>} \right) } $$



foward propation goes from left to right. back propagation go from right to left 
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic6.png)

<span style="color: red">Many to Many Architectures</span>: 比如word识别名字，输入的每word，都有输出0，1; 注：many-to-many, input length 和 Output length可以相同，也可以不同，比如翻译先把法语(encoder)句子读完，然后一个一个generate 英语(decoder)的 <br/>
<span style="color: red">Many to One Architectures</span>:  Sentiment Classification: 给一个word，只最后输出0-5代表几个星<br/>
<span style="color: red">One to One Architectures</span>: standard neural network<br/>
<span style="color: red">One to Many Architectures</span>: output set of notes 代表a piece of music (x 可以是null)

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic7.png)

#### Sequence generation
Language Model:

analyze the probbility of sequence of words, 比如<br/>
P(The apple nd pair salad) = $$3.2\times10^{-13} $$<br/>
P(The apple nd pair salad) = $$5.7\times10^{-10} $$

Training Set: large corpus (set) of English text; >首先<span style="background-color: #FFFF00"> tokenize </span>把word map到字典上，生成vector. 有时add extra token EOS 表示句子的结尾。 也可以决定是否把标点符号也tokenize. 如果word 不在字典中，用<span style="color: red">UNK</span> stands for unknown word
 
RNN Model:

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic8.png)

speech generation. 从$$a^{<{1}>} $$到 $$\hat y^{<{1}>}$$ 是softmax matrix，得到字典中每个字的概率， $$y^{<{1}>}$$是一个10002(10000 + unknown + EOS) vector，到了$$a^{<{2}>}$$, given the first correct answer, what is the distribution of P(__ \| cats); 到了$$a^{<{3}>}$$, given the first correct answer, P(__ \| cats, average);到最后一个predict P(_ \|....前面所有的), cost function is softmax cost function; given the first word, $$P\left( y^{<{1}>}, y^{<{2}>}, y^{<{3}>} \right) = P\left( y^{<{1}>}\right) \cdot P\left(y^{<{2}>} \| y^{<{1}>} \right)\cdot  P\left(y^{<{3}>} \| y^{<{1}>}, y^{<{2}>} \right) $$


#### Sampling novel Sequence:

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic9.png)
 
 得到$$\hat y^{<{1}>}$$后，random sample 选取y 根据softmax的distribution(a的概率多大，aaron的概率多大)， 然后得到的sample在next time step作为input,再得到$$\hat y^{<{2}>}$$; 比如$$\hat y^{<{1}>}$$sample = The, 把the 作为input，得到另一个softmax distribution P( _ \| the), 再sample $$\hat y^{<{2}>}$$,  把sample的 pass 到next time step. <span style="color: red">When to stop:</span>, keep sampling until generate EOS token. 如果没有设置EOS. then decide to sample 20 个或者100个words 知道到达这个次数(20 or 100). 有时可能生成unknown word token, 可以确保algorithm 生成sample 不是unknown token，遇到unknown token就继续keep sampling until get non-unknown word

字典除了是vocabulary，也可以是character base， 如果想build character level 而不是word level 的，$$y^{<{1}>}, y^{<{2}>}, y^{<{3}>}$$是individual characters， <span style="background-color: #FFFF00">character 就不会遇见unknown word的情况. Disadvantage: 1. end up much longer sequence. </span> 一句话可能有10个词，但会有很多的char，<span style="background-color: #FFFF00"> 2. character level 不如word level 能capture long range dependencies between how the earlier parts of sentence aslo affect the later part of the sentence. </span> 3. character level more computationally expensive to train. 当计算机变得越来越快，more people look at character level models (not widespread today for character level)


#### Vanishing gradients

languages that comes earlier 可以影响 later的，比如前面提到cats, 十个单词后可能需要用were 而不是was， 除了vanishing gradient的问题，也有explode gradient的问题（expoentially large gradients can cause parameters become so large 导致 parameters blow up, often see NaNs, have overflow in neural network computation),  <span style="background-color: #FFFF00"> exploding gradient 可以用gradient clipping</span>，<span style="color: red">当超过某个threshold得时候，rescale避免too large. thare are clips according to some 最大值</span>




[pic3]: https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep_Learning-Sequence_Model_note/week1pic3.png


#### GRU && LSTM

**GRU**:

 $$\begin{align} \tilde c^{<{t}>} &= tanh \left( W_c \left[ \Gamma_r \times c^{<{t-1}>}, x^{<{t}>}  \right] + b_c \right) \\ \Gamma_r &= \sigma \left( W_r \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_r \right) \\  \Gamma_u &= \sigma \left( W_u \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_u \right) \\ c^{<{t}>} &= \Gamma_u \cdot \tilde c^{<{t}>}  + \left( 1 - \Gamma_u \right) \cdot \tilde c^{<{t-1}>}   \end{align}$$  