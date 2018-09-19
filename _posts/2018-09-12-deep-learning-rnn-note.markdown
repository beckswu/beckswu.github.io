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
    - 学习笔记
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

use dictionary and give each word an index, <br/>
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
\end{align}$$ <br/>
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

languages that comes earlier 可以影响 later的，比如前面提到cats, 十个单词后可能需要用were 而不是was， 除了vanishing gradient的问题，也有explode gradient的问题（expoentially large gradients can cause parameters become so large 导致 parameters blow up, often see NaNs, have overflow in neural network computation),  <span style="background-color: #FFFF00"> exploding gradient 可以用gradient clipping</span>，<span style="color: red">当超过某个threshold得时候，rescale避免too large. thare are clips according to some 最大值</span>, 比如gradient超过[-10,10], 就让gradient 保持10 or -10



#### GRU && LSTM

**GRU**:

 $$\begin{align} \tilde c^{<{t}>} &= tanh \left( W_c \left[ \Gamma_r \times c^{<{t-1}>}, x^{<{t}>}  \right] + b_c \right) \\ \Gamma_r &= \sigma \left( W_r \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_r \right) \\  \Gamma_u &= \sigma \left( W_u \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_u \right) \\ c^{<{t}>} &= \Gamma_u \cdot \tilde c^{<{t}>}  + \left( 1 - \Gamma_u \right) \cdot  c^{<{t-1}>}  \\ a^{<{t}>} &= c^{<{t}>}  \end{align}$$  


1. c是memory cell, a 是output cell, c = memory cell 比如记录cat 是单数还是复数, 用于后面记录是was or were 
2. $$\tilde c^{<{t}>}$$是candidate value 代替$$c^{<{t}>}$$， 
3. $$\Gamma_u$$是表示gate, 如果gate = 1, $$c^{<{t}>}$$ 更新值为 candidate 值 $$\tilde c^{<{t}>}$$, 比如遇到cat gate = 1更新 $$c^{<{t}>}$$为1表示单数, the cat, which already ate.... was full, 从cat 到was, gate =0, means don't update, 直到was, $$c^{<{t}>}$$还为1 (without vanishing)
4. sigmoid function for $$\Gamma_u$$ easy to set zero, 只要 $$ W_u \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_u $$ 是非常大的负数
5. $$c^{<{t}>}$$可以是vector (比如100维，100维都是bits), then$$\Gamma_u$$,$$\tilde c^{<{t}>}$$都是same dimension,  $$ \Gamma_u \cdot \tilde c^{<{t}>}  + \left( 1 - \Gamma_u \right) \cdot  c^{<{t-1}>} $$ 是element wise operation, 点乘告诉哪个bit需要update，哪个保持上一个value，比如用第一个维度代表单数复数，第二维度代表是不是food
6. $$\Gamma_r $$: relevance, how relevant $$c^{<{t-1}>}$$ to update $$c^{<{t}>}$$


**LSTM**:

 $$\begin{align} \tilde c^{<{t}>} &= tanh \left( W_c \left[ \Gamma_r \times a^{<{t-1}>}, x^{<{t}>}  \right] + b_c \right) \\ \Gamma_u &= \sigma \left( W_u \left[ a^{<{t-1}>}, x^{<{t}>}  \right] + b_u \right) \\  c\\  \Gamma_o &= \sigma \left( W_o \left[ a^{<{t-1}>}, x^{<{t}>}  \right] + b_o \right) \\ c^{<{t}>} &= \Gamma_u \cdot \tilde c^{<{t}>}  + \Gamma_f  \cdot  c^{<{t-1}>}  \\ a^{<{t}>} &= \Gamma_o \cdot tanh\left(c^{<{t}>} \right) \end{align}$$  



1. $$\Gamma_u$$是表示update gate,  $$\Gamma_o$$是表示forget gate, $$\Gamma_o$$是表示output gate
2. peephole connection($$c^{<{t-1}>}$$): gate value may not only depend on $$a^{<{t-1}>}$$ & $$x^{<{t}>}$$, 也可能depend on $$c^{<{t-1}>}$$, $$\Gamma_o = \sigma \left( W_o \left[ a^{<{t-1}>}, x^{<{t}>}, c^{<{t-1}>}  \right] + b_o \right)$$

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic10.png)
 

| GRU | LSTM |
| ------:| -----------:|
| $$c^{<{t}>} $$ 等于 $$a^{<{t}>} $$ | $$c^{<{t}>} $$ 不等于 $$a^{<{t}>} $$ |
| update $$c^{<{t}>} $$是由gate $$\Gamma_u$$控制，如果不update, gate = 0, $$c^{<{t}>} $$ = $$c^{<{t-1}>} $$   | 有三个gate  $$\Gamma_u$$,$$\Gamma_f$$,$$\Gamma_o$$ 分别控制update, forget, 和output |

when use GRU or LSTM: isn't widespread consensus in this; Andrew: GRU is simpler model than LSTM, <span style="background-color: #FFFF00">easy to build much bigger network</span> than LSFT, LSTM is <span style="background-color: #FFFF00">more powerful and effective</span> since it has three gates instead of two. LSTM is move historical proven


#### Bidirection RNN && Deep RNNS:

单向的RNN的问题，比如 He said "Teddy bears are on sale"; He said “Teddy Roosevelt was a great President". Teddy都是第三个单词且前两个都一样，而只有第二句话的Teddy表示名字<br/>
Bidirection RNN: part forward prop从左向右，part forward prop从右向左, 每个Bidirection RNN block还可以是GRU or LSTM的block

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic11.png)

 $$ \tilde y^{<{t}>} = g\left( W_y\left[ \overrightarrow a^{<{t}>}, \overleftarrow a^{<{t}>}   \right] + b_y \right)$$  

<span style="background-color: #FFFF00">disadvantage</span>: 需要entire sequence of data before you can make prediction; 比如speech recognition: 需要person 停止讲话 to get entire utterance before process and make prediction

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic12.png)

For RRN, 三层已经是very deep, $$a^{\left[{1}\right]<{0}>}$$表示第1层第0个input，在output layer也可以有stack recurrent layer，但这些layer没有horizon connection， 每个block 也可以是GRU, 也可以是LSTM, 也可以build deep version of bidirectional RNN, <span style="background-color: #FFFF00">Disadvantage: computational expensive to train</span>

比如计算$$a^{\left[{2}\right]<{3}>}$$:   $$a^{\left[{2}\right]<{3}>} = g\left( W_a^2 \left[a^{\left[{2}\right] <{2}>}, a^{\left[ {1}  \right] <{3}>}  \right] \right)$$

<br/><br/><br/>


## Week2 NLP & Word Embedding


#### Word Embedding:
<span style="background-color: #FFFF00">Word Embedding: </span> 让algorithm学会同义词：比如男人vs女人，king vs queen<br/> 
<span style="background-color: #FFFF00">One hot vector的缺点</span>: 10000中(10000是字典)，只有1个为1表示这个词，不能表示gender. age, fruit..., 因为任何两个one-hot vector的inner product是zero

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic1.png)

可能apple 和orange有的feature不一样比如color，但是a lot feature是一样的， <span style="background-color: #FFFF00">T-SNE</span> 把3000vector visualize 到2-D, analogy tends to close

Embedding training dataset 需要很大的，会发现比如durian 和orange， farmer 和cultivator是同义词, 
1. 所以当training set有限的时候，可以先train 从网上的文本（10billion 个）or use pre-training embedding online，
2. 然后再apply <span style="background-color: #FFFF00">transfer learning</span> 到你的task上(size = 100K), then use 300 dimension vector（位置一表示性别，位置二表示color...） to represent word instead of one hot vector(dimension: 10000),<span style="background-color: #FFFF00">**advantage**</span>: use low dimension feature vector.  
3. continue to fine-tune word embeddings with new data(only 你的task dataset is large)


**Cosine Similarity**: 

比如 $$e_{man} - e_{woman} \approx e_{king} - e_{?} 用similarity function $$ $$sim\left( e_{w}, e_{king} - e_{man} + e_{woman} \right)$$, <br/>
$$sim\left( u, v \right)  = \frac{u^Tv}{||u||_2 ||v||_2 } $$ <br/>
如果u,v similar, similarity will be large, 因为$$u^Tv$$表示他们的夹角(cos), 
 <br/>or measure dissimilarity Euclidian distance:
$${||u-v||}^2$$ 通常measure dissimilarity than similarity

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic2.png)


**Embedding Matrix**:

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic3.png)
可以用embedding matrix 乘以one hot vector得到属于现在词的embedding vector,但是通常不用，因为不efficient, in practice用just lookup 那个word的emdding matrix column e

#### Word2vec & Negative Sampling & GloVe

**Word2Vec**:

fixed history: 比如I want a glass of orange ___ , 预测填入的是juice，把前四个word, a glass of orange 代入network, 每个词都是300维的embedded vector(来自same embedded matrix),把4个300 stack together, 带入hidden layer, 再用softmax predict;  <span style="background-color: #FFFF00">Advantage</span>: can deal with arbitrary long 句子，因为input size is fixed 

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic4.png)

Context/target pairs:   Context可以是 last 4 words; Context也可以是4 word on left & right; Context也可以是nearby one word

**Skip-grams**:

比如句子: I want a glass of orange juice to go along with my cereal; 先去<span style="color: red">context word</span> 比如选取了word: orange, 随机pick another word within some window as <span style="color: red">target word</span>  比如前后的5个或者10个词; 比如 context: orange -> target: juice; context: orange -> target: glass; context: orange -> target: my; 

Goal: learn from content to target;  vocabulary size  = 10,000, context: orange (vector index 6257) ->  target: juice (4834)  

Model:  $$ O_c \rightarrow E \rightarrow e_c \rightarrow softmax \rightarrow \hat y$$   <br/>
Softmax: $$ p(t |c) = \frac{ \theta_t^T e_c }{ \sum_{j=1}^{10,000} { e^{ \theta_j^T e_c  }  } } $$  $$\theta_t$$ is parameter associated with output t <br/>
Loss function: $$ L \left(\hat y , y \right) = - \sum_{i=1}^{10,000} { y_i log\hat{y_i}  }$$ 

<span style="background-color: #FFFF00">Problem with softmax classification</span>: softmax的分母每次都要sum over all words in vocabulary; solution1: hierarchical softmax:有点像segment tree, 把所有的单词分成一半，再分一半。。。每一个parent 记录所有的softmax的和of all childs; complexity: log|v| ; 通常不是balanced tree, common words 在top, less common 在deeper(因为不common的，通常不用go that deep in the tree)
![](/img/post/Deep_Learning-Sequence_Model_note/week2pic5.png)

How to find context c: 如果我们random 选择from training corpus, 可能会选择很多the, a, of, and, to,但我们更想让model训练比如orange, durian这样的词 


**Negative Sampling**:


Given word: orange & juice. Is context - target pair?<br/>
比如: I want a glass of orange juice to go along with my cereal. 

| Context | Target | target? |
| ------:| -----------:| ------:|
|orange | juice | 1  |
|orange | king | 0 |
|orange | book | 0 |
|orange | of | 0 |

sample context and target word; <span style="color: red">Positive example</span> generated: look at context within windows (5 or 10 word around); <span style="color: red">Negative example: take the same context word. then pick a word randomly from dictionary </span>; 注意: 上面最后一个例子，"of" is zero even if we have "of"; <br/>
<span style="background-color: #FFFF00">Generate training set</span>: 先generate positive example. 再生成k个negative examples, it is okay 如果生成的negative example 在context +-5，+-10 window出现; k = [5,20] for small dataset, k = [2,5] for large dataset

**Model**: $$ \theta_t^{T} $$ one parameter theta for each target word, $$ e_c $$ for embedding vector. Instead of 10000 way softmax which is expensive to compute, <span style="background-color: #FFFF00">instead we have 10000 binary classification problem</span>

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic6.png)

Select examples: If you choose words 根据its frequence, 可能end up with the, of, and; use $$ P(W_i) =  \frac{ f \left(w_i \right)^{3/4} }{ \sum_{j=1}^{10,000} { f\left(w_i \right)^{3/4}  } } $$ 这个分布选取



**GloVe**:

$$X_{ij} $$ = times  i (target) appears in context of j (context)， i 在j的上下文出现多少次; 如果上下文是前后10个词的话,  也许得到symmetric relationship $$X_{ij} = X_{ji} $$; 当如果只选word before it, may not get symmetric relation ship

Model:  use gradient descent to minimize below function; 为了避免log0 出现, 乘以weight term $$f\left(X_{ij}\right)$$; $$\theta_j$$ 和 $$e_j$$是symmetric的，可以reversed or 对调，会得到同样的目标函数， when do gradient descent, 所以可以取个平均值; <span style="background-color: #FFFF00">Initialize</span> both $$\theta_i$$ 和 $$ e_j $$ randomly uniformly at beginning 

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic7.png)

Aslo cannot 保证embeded vector是可以解释的,parallelogram for analogies still works


![](/img/post/Deep_Learning-Sequence_Model_note/week2pic8.png)


#### Embedding Application

**Sentiment Classification**

<span style="background-color: #FFFF00">Chanllenge: </span> not have a huge dataset

Simple Sentiment Classification Model: 用embedded vector which from large training set: so 不通常出现的word 也可以label 他们


1. use average of each words output: used for review that are short or long; <span style="background-color: #FFFF00">Problem: Ignore order </span>：比如: completely lacking good service an dgood ambience, 即使有两个good，也是1星review
2. RNN for sentiment Classification:  <span style="background-color: #FFFF00">many-to-one architecture </span>


![](/img/post/Deep_Learning-Sequence_Model_note/week2pic9.png)
![](/img/post/Deep_Learning-Sequence_Model_note/week2pic10.png)

**Debiasing Word Embeddings**

比如消除性别的歧视， 比如 man: programmer as Woman: Homemaker; 比如 man: Doctor as Mother: Nurse; Word embeddings可以reflect gender, ethnicity ages... biases of text used to train to model; 

Address bias: 

1. Identiy bias direction； 比如用 embeded vector $$ e_{he} - e_{she}; e_{male} - e_{female} $$... averge them 得到bias direction(1 dimension), 垂直的bias direction是 non-bias direction(299 dimension)
2. Neutralize: 对于不是definitional 的word (_definitional的是grandmother, grandfather, 不是definitional比如 doctor, babysitter_), project to get rid of bias, project them到non-bias direction; 对于如何选取什么word neutralized, author的看法；train a classifier to try to figure out 什么word是definitional 什么不是; 大多数英语单词都是non-definitional的
3. Equalize pairs: 比如 grandfather vs grandmother, boy vs girl, 比如下图中 babysitter 的project的点距离grandmother比grandfather 更近, which is a bias; 所以移动grandfather 和 grandmother to pair points (到距离Non-bias direction的距离一样的点); 选取equalized pairs不会很多，可以hand-picked


![](/img/post/Deep_Learning-Sequence_Model_note/week2pic11.png)

<br/><br/><br/>


## Week3 Sequence Models & Attention Mechanism

**Sequence to Sequence Model**

Machine translation: RNN先用<span style="color: red">encoder network</span> (input one word 每次), figure out some representation of sentence. RNN 最后 output 一个 vector代表input sentence，用这个vector作为<span style="color: red">decode netork</span>的开始 再用decode network 一个一个output 翻译的单词，  <span style="background-color: #FFFF00">difference from synthesizing novel text using language model: 不需要randomly choose translation, want the most likely translation. </span>

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic1.png)

can think machine translation as building a conditional language model. Machine translation model的decode network 很接近language model. Encode network model the probability $$ P \left( y ^{<{1}>}, y ^{<{2}>},\cdots,  y ^{<{T_x}>} |  x ^{<{1}>}, \cdots \right) $$, output the probability of English Translation condition on some input French sentence

<span style="background-color: #FFFF00"> Finding the most likely translation </span>: 不能用random sample output from $$y ^{<{1}>$$ to $$y ^{<{2}>$$, 有时候可能得到好的，有时候得到不好的翻译; instead: the goal should maximize the probability  $$ P \left( y ^{<{1}>}, y ^{<{2}>},\cdots,  y ^{<{T_x}>} |  x \right) $$

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic2.png)

**Why not Greedy Search** Greedy Search: 在pick 第一个word 后，选择概率最高的第二个单词，再选择概率最高的第三个单词，我们需要的是最大化joint probability $$ P \left( y ^{<{1}>}, y ^{<{2}>},\cdots,  y ^{<{T_x}>} |  x \right) $$, 这么选出的word 不一定是接近最大的joint proability 的句子; 比如翻译的句子是 Jane is visiting Africa in September这个是perfect翻译, 但是greedy翻译出来的是 Jane is going to be visiting Africa in September. 因为Jane is goint 的概率大于Jane is visiting

不能run 全部combination of words，算那个概率最大， 比如有10000个词组成的字典，句子长度为10，总共有 $$10000^{10} $$种组合, 所以需要approximate search algorithm，可能不是总成功，不同 try to find sentences to maximize joint conditional probability.


#### Beam Search


**Beam Search Algorithm**, <span style="background-color: #FFFF00"> B = beam width</span>: 不像greedy search 每次只考虑最大可能的一个词，beam search 会考虑最大可能的B个词； 注: 当B=1, 相当于greedy search

Example： B = 3
1. Step1: evulate $$ P\left(y^{<{1}>} | x\right) $$, 发现 in, jane, september是根据概率最的可能的三个词, keep [in, jane, september]
2. Step2: evulate $$ P\left(y^{<{2}>} | y^{<{1}>}, x\right) $$, $$ P\left(y^{<{1}>},  y^{<{1}>} | x\right) = P\left(y^{<{1}>} | x\right) P\left(y^{<{2}>} | y^{<{1}>}, x\right) $$  比如字典有10000个词，考虑来自step1三个词作为开始，只用考虑10000*3个词, then pick top3; 比如发现算上第二词 最大可能性的三个词 [In september, jane is, jane visit] -> reject september 作为第一个词的可能

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic3.png)


**Length normalization**,

可能$$ P\left(y^{<{t}>} | x, y^{<{1}>}, y^{<{2}>}, \cdots, y^{<{t-1}>}    \right) $$概率的乘积越来的越小，不好记录，与其记录乘积，也可记录sum of log, more stable to avoid overflow and numeric rounding error;  <span style="background-color: #FFFF00">problem: 可能prefer更短的句子, 因为probability都是小于1，句子越长概率乘积越小，同样log都是0，句子越长加的负数越大</span>, <span style="background-color: #FFFF00">Solution: normalize 概率，除以句子长度</span> $$ 1/{T_y^\alpha}$$ maybe $$\alpha$$ = 0.7,当$$\alpha$$=1, complete normalize by length; 当$$\alpha$$=0, $$ 1/{T_y^\alpha} = 1/1$$: not normalized at all. 0.7是between full normalization and no normalization; <span style="color: red">同时alpha也可以作为hyperparameter 用来tune</span>

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic4.png)

比如beam = 3， 看所有top three possibilities of length 1, 2, ...,30 against 上面的normalized probability score, pick the one 有最高score的(highest normalized log likelihood objective) 作为final translation output

How to choose Beam width B? 在实际中可能选择around 10;  100 consider be large; 1000, 3000是not common的, 用越来越大的B, it is diminishing returns; 比如gain 很大从1->3->10, 但是gain 从1000->3000, 不是很大了
- large B: pro: better result， con: slower
- small B: pro: run faster,  con: worse result

<span style="background-color: #FFFF00"> 不像BFS, DFS, Beam Search runs faster 但是不确保find exact maximum for 最大化 P(y|x) </span>

**Beam Search Error Analysis**

Example: <br/>
Jane visite l'Afrique en septembre. <br/>
Human 翻译: Jane visits Africa in September ($$y^{*}$$) <br/>
Algorithm 翻译: Jane visited Africa last September ($$\hat y$$)

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic6.png)

用RNN 计算 $$P\left( y^{*}|x \right) $$, $$P\left(\haty |x \right) $$
1. Case 1:  $$P\left( y^{*}|x \right) $$ > $$P\left(\haty |x \right): Beam choose $$\hat y$$, 但是 $$y^{*}$$ attains 更高的 P(y|x); <span style="background-color: #FFFF00"> Beam search is at fault </span>
2. Case 2: $$P\left( y^{*}|x \right) $$ <=  $$P\left(\haty |x \right);  $$y^{*}$$  better translation than $$\hat y$$, 但是 RNN预测相反, <span style="background-color: #FFFF00"> RNN is at fault </span>


![](/img/post/Deep_Learning-Sequence_Model_note/week3pic5.png)


[pic3]: https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep_Learning-Sequence_Model_note/week1pic3.png
