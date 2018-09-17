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

fixed history: 比如I want a glass of orange ___ , 预测填入的是juice，把前四个word, a glass of orange 代入network, 每个词都是300维的embedded vector(来自same embedded matrix),把4个300 stack together, 带入hidden layer, 再用softmax predict;  <span style="background-color: #FFFF00">advantage</span> can deal with arbitrary long 句子，因为input size is fixed 

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic4.png)

Context/target pairs:   Context可以是 last 4 words; Context也可以是4 word on left & right; Context也可以是nearby one word

**Skip-grams**:

比如句子: I want a glass of orange juice to go along with my cereal; 先去<span style="color: red">context word</span> 比如选取了word: content, 随机pick another word within some window as <span style="color: red">target word</span>  比如前后的5个或者10个词; 比如 context: orange -> target: juice; context: orange -> target: glass; context: orange -> target: my; 

Goal: learn from content to target;  vocabulary size  = 10,000, context: orange (vector index 6257) ->  target: juice (4834)  

Model:  $$ O_c \rightarrow E \rightarrow e_c \rightarrow softmax \rightarrow \hat y$$   <br/>
Softmax: $$ p(t |c) = \frac{ \theta_t^T e_c }{ \sum_{j=1}^{10,000} { e^{ \theta_j^T e_c  }  } } $$  $$\theta_t is parameter associated with output t <br/>
Loss function: $$ L \left(\hat y , y \right) = - \sum_{i=1}^{10,000} { y_i log\hat{y_i}  }$$ 

<span style="background-color: #FFFF00">Problem with softmax classification</span>: softmax的分母每次都要sum over all words in vocabulary; solution1: hierarchical softmax:有点像segment tree, 把所有的单词分成一半，再分一半。。。每一个parent 记录所有的softmax的和of childs; complexity: log|v| ; 通常不是balanced tree, common words 在top, less common 在deeper(因为不common的，通常不用go that deep in the tree)
![](/img/post/Deep_Learning-Sequence_Model_note/week2pic5.png)

How to find context c: 如果我们random 选择from training corpus, 可能会选择很多the, a, of, and, to,但我们更想让model训练比如orange, durian这样的词 

[pic3]: https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep_Learning-Sequence_Model_note/week1pic3.png