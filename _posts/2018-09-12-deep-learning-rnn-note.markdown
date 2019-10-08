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
    - Machine Learning
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
#### Notation: 

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic1.png)


example: given a sentence 判断哪个是人名<br/> 
$$x^{({i})<{t}>}$$:  表示第i个training example 中第t个word <br/> 
$$y^{({i})<{t}>}$$:  表示第i个training example 中第t个word的输出label<br/> 
$$T_x^{i}$$:  表示第i个training example的长度<br/> 
$$T_y^{i}$$:  表示第i个training example的ouput长度<br/> 


**representing words:** <br/>

use dictionary and give each word an index, <br/>
$$x^{<{t}>}$$:  是one hot vector(meaning: only one in one position, everywhere else 0), 比如字典的长度是10000, x = apple, apple出现在字典的100位, $$x^{<{t}>} = \begin{bmatrix}
    0 \\
    \vdots \\
    1  \\
	\vdots\\
    \end{bmatrix}
$$ 只有第100位是1，剩下都是0. if 遇见了word不在字典中，create a new token or a new fake word called unknown word e.g. ```<unk>```

Note: Some internet company use dictionary maybe 1 million or een bigger than that 

比如下面看是不是name的，output是长度为9，0代表不是name, 1代表是name
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic2.png)


#### Recurrent Neural Network Model:
<span style="background-color: #FFFF00">Why not a standard network?</span>(e.g. sentiment in NLP ) <br/>
problems:
1. Input, output can be <span style="color:red">different lengths</span> in different example (不是所有的input的都是一样长度)
2. Doesn't share features learned across <span style="color:red">**different positions**</span> of text(也许word Harry在位置1，但是也许Harry也许出现在位置7)



- At time 0, have some either made-up activation( initialized randomly or 全部是0的vector) as $$a^{<{0}>}$$. <br/>
- step 1: Take a word(first word) to a neural network layer, then try to predict if this word is name or not. <br/>
- step 2: Use activation value from step 1 and $$x_{2}$$ to predict $$y_2$$. Then take activation value from step 2 to step 3. 
- The <span style="color:red">activation parameters</span> () used in each step are <span style="color:red">**shared**</span>. 
  - $$W_{ax}$$, (from x to activation) govern the connection between $$x_{<i+1>}$$ $$x_{<i>}$$
  - $$W_{aa}$$, (from activation to activation) govern the horizontal activation connection  
  - $$W_{ya}$$ (from activition to y) (用x得到y like quantity) 控制governs the output prediction

Below structure $$T_x = T_y$$,  e.g.  $$y_{<3>}$$ not only get infomation from $$x_{<3>}$$ but aslo from $$x_{<1>}$$ and $$x_{<2>}$$

One <span style="color:red">**weakness**</span> for RNN: only use <span style="color:red">information</span> that is <span style="color:red">earlier</span> in the sequence to make a prediction （Bidirection RNN (BRNN) 可以解决这个问题）e.g. when prediciting  $$y_{<3>}$$ not use  $$x_{<4>}$$


![][pic3]


**Forward Propagation**:

$$\begin{align} a^{<{0}>} &= \vec0  \\
a^{<{1}>} &= g_1\left(W_{aa}\cdot a^{<{0}>}+ W_{ax}\cdot X^{<{a}>} + b_a \right) \\
y^{<{1}>} &= g_2\left(W_{ya}\cdot a^{<{1}>} + b_y \right)
\end{align}$$ <br/>
从$$a^{<{t-1}>} $$和 $$x^{<{t}>}$$ 生成$$a^{<{t}>}$$ 的可以是<span style="color: red">tanh/Relu</span>, 从$$a^{<{t}>}$$ 到$$y^{<{t}>}$$的是<span style="color: red">softmax</span>

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic4.png)

简化符号
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic5.png)



#### Backpropagation Through Time

Single Loss Function: $$ L^{<{t}>}\left( \hat y^{<{t}>},  y^{<{t}>} \right) = - y^{<{t}>}log \left( \hat y^{<{t}>} \right) - 
    \left( 1- y^{<{t}>} \right) log\left(1- \hat y^{<{t}>} \right) $$<br/>

Overall Loss Function:  $$ L \left(  \hat y , y \right) =  \displaystyle \sum_{t=1}^{T_x} {L^{<{t}>} \left( \hat y^{<{t}>}, y^{<{t}>} \right) } $$



foward propation goes from left to right. back propagation go from right to left 
![](/img/post/Deep_Learning-Sequence_Model_note/week1pic6.png)


#### RNN Architectures

<span style="color: red">**Many to Many** Architectures</span>: 比如word识别名字，输入的每word，都有输出0，1; 注：many-to-many, input length 和 Output length可以相同，也可以不同，比如翻译先把法语(encoder)句子读完，然后一个一个generate 英语(decoder), English and French sentences can be different length <br/>
<span style="color: red">**Many to One** Architectures</span>:  Sentiment Classification: 给一个word，只最后输出0-5代表几个星<br/>
<span style="color: red">**One to One** Architectures</span>: standard neural network<br/>
<span style="color: red">**One to Many** Architectures</span>: e.g. music generation output set of notes 代表a piece of music (x 可以是null)

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic7.png)

#### Sequence generation
Language Model: the way that speech recognition pick words based on probability. Output only sentences that are likely. 比如, then pick the second sentences<br/>
P(The apple and pair salad) = $$3.2\times10^{-13} $$<br/>
P(The apple and pair salad) = $$5.7\times10^{-10} $$

Training Set: large corpus of English text; corpus: NLP terminalogy means a large body/set. 
- 首先<span style="background-color: #FFFF00"> **tokenize** </span>把word map到字典上，生成vector. 有时add extra token EOS 表示句子的结尾。 也可以决定是否把标点符号也tokenize. 如果word 不在字典中，用<span style="color: red">UNK</span> substitue for unknown word
 
RNN Model:

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic8.png)

Training

1. 从$$a^{<{1}>} $$到 $$\hat y^{<{1}>}$$ 是softmax matrix, <span style="color:red">得到字典中每个字的概率</span>， $$y^{<{1}>}$$是一个10002(10000 + unknown + EOS) vector，到了$$a^{<{2}>}$$, 
2. At $$a^{<{2}>}$$, <span style="color:red">given the first correct answer</span>, what is the distribution of P(__ \| cats); 
3. At $$a^{<{3}>}$$, given the first correct answer, P(__ \| cats, average);
4. At the last one, predict P(_ \|....前面所有的), 

cost function is softmax cost function; $$L\left( \hat y^{t}, y^{t} \right) = - \sum_{i} {y_i^{t} log \hat y_i^{t} } $$, $$ L = \sum_{t} {L^{t} \left( \hat y^{t}, y^{t} \right)}

given the first word, $$P\left( y^{<{1}>}, y^{<{2}>}, y^{<{3}>} \right) = P\left( y^{<{1}>}\right) \cdot P\left(y^{<{2}>} \mid y^{<{1}>} \right)\cdot  P\left(y^{<{3}>} \mid y^{<{1}>}, y^{<{2}>} \right) $$


#### Sampling novel Sequence:

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic9.png)

Sample from distribution to generate noble sequences of words
 
1. 得到$$\hat y^{<{1}>}$$后，random sample according to softmax的distribution(a的概率多大，aaron的概率多大)，```np.random.choice``` to sample according to this distribution. 
2. take the $$\hat y^{<{1}>}$$ you just sample in step 1 to pass as input to next timestamp.再得到$$\hat y^{<{2}>}$$; 
    - 比如$$\hat y^{<{1}>}$$sample = The, 把the 作为input，得到另一个softmax distribution P( _ \| the), 再sample $$\hat y^{<{2}>}$$,  把sample的 pass 到next time step. 
3. <span style="color: red">**When to end**:</span>,
   - keep sampling until generate EOS token. 
   - 如果没有设置EOS. then decide to sample 20 个或者100个words 知道到达这个次数(20 or 100 words). 有时可能生成unknown word token, 可以确保algorithm 生成sample 不是unknown token，遇到unknown token, reject and keep sampling until get non-unknown word. can leave it in output if don't mind having unknown word output

字典除了是vocabulary，也可以是character base， 如果想build character level 而不是word level 的，$$y^{<{1}>}, y^{<{2}>}, y^{<{3}>}$$是individual characters， E.g. Cat average. $$y^{<1>} = c$$, $$y^{<2>} = a$$ , $$y^{<3>} = t$$, $$y^{<4>} = space$$   

Advantage: 
- <span style="background-color: #FFFF00">character 就不会遇见unknown word的情况</span>. 比如 Mau, not in vocabulary, assign unknown, for character letter, 不会是unknown
  
Disadvantage: 

- end up much **longer sequence**.  一句话可能有10个词，但会有很多的characters，
- Character level 不如word level 能capture long range dependencies between how the earlier parts of sentence aslo affect the later part of the sentence.
-  Character level more <span style="color:red">**computationally expensive**</span> to train. 当计算机变得越来越快，more people look at character level models (not widespread today for character level)


#### Vanishing gradients

languages that comes earlier 可以影响 later的， e.g.  choose was or were

The cat which .... was full <br/>
The cats which .... were full

The basic RNN <span color="style:red">not very good at capturing very long-term dependency</span>.  because for very deep neural network e.g. 100 layers, <span style="color:red">later layer</span> had <span style="color:red">hard time propagating back</span> to affect the weights of these earlier layers. 

It means the output only influenced by close input, $$y^{<20>}$$ is affected by $$x^{<20>},x^{<19>}, x^{<18>} $$, not $$x^{<1>}$$. The errors associated at latter timestep to affect computation that are eariler. e.g. cats or cat affect was, were.

Exploding Gradient: aslo happen for RNN, increase exponentially with the number of layers go through. Whereas Vanish Gradient tends to a bigger problem for RNN 
    - 导致 parameters blow up, often see NaNs, have overflow in neural network computation,  
    - <span style="background-color: #FFFF00"> exploding gradient 可以用**gradient clipping**</span>，<span style="color: red">当超过某个threshold得时候，rescale避免too large. thare are clips 不超过最大值</span>, 比如gradient超过$$\left[-10,10\right]$$, 就让gradient 保持10 or -10




#### GRU && LSTM

**GRU**: Gated Recurrent Unit, capture long range connection and solve Vanishing Gradient

 $$\begin{align} \tilde c^{<{t}>} &= tanh \left( W_c \left[ \Gamma_r \times c^{<{t-1}>}, x^{<{t}>}  \right] + b_c \right) \\ \Gamma_r &= \sigma \left( W_r \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_r \right) \\  \Gamma_u &= \sigma \left( W_u \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_u \right) \\ c^{<{t}>} &= \Gamma_u \cdot \tilde c^{<{t}>}  + \left( 1 - \Gamma_u \right) \cdot  c^{<{t-1}>}  \\ a^{<{t}>} &= c^{<{t}>}  \end{align}$$  


1. c是memory cell, a 是output cell, c = memory cell 比如记录cat 是单数还是复数, 用于后面记录是was or were 
2. $$\tilde c^{<{t}>}$$是candidate value 代替$$c^{<{t}>}$$， 
3. $$\Gamma_u$$是表示gate, value between 0 and 1, For most of possible range, it will very close to 0 or very close to 1. The job of $$\Gamma_u$$ is to decide when to update $$c^{<t>}$$ value, 如果gate = 1, $$c^{<{t}>}$$ 更新值为 candidate 值 $$\tilde c^{<{t}>}$$, 比如遇到cat gate = 1更新 $$c^{<{t}>}$$为1表示单数, the cat, which already ate.... was full, 从cat 到was, gate =0, means don't update, 直到was, $$c^{<{t}>}$$还为1 . Because $$\Gamma_u$$ can be so close to zero, <span style="color:red">it won't suffer that vanish gradient problem </span>, allow nerual network to learn long range dependency
4. sigmoid function for $$\Gamma_u$$ easy to set zero, 只要 $$ W_u \left[ c^{<{t-1}>}, x^{<{t}>}  \right] + b_u $$ 是非常大的负数
5. $$c^{<{t}>}$$可以是vector (比如100维，100维都是bits), then$$\Gamma_u$$,$$\tilde c^{<{t}>}$$都是same dimension,  $$ \Gamma_u \cdot \tilde c^{<{t}>}  + \left( 1 - \Gamma_u \right) \cdot  c^{<{t-1}>} $$ 是 <span style="color:red">**element wise operation**</span>, to tell bit需要update, to keep some bits as before and update other bits，哪个保持上一个value，比如用第一个维度代表单数复数，第二维度代表是不是food
6. $$\Gamma_r $$: relevance, how relevant $$c^{<{t-1}>}$$ to update $$c^{<{t}>}$$


**LSTM**: Long Short Term Memory

 $$\begin{align} \tilde c^{<{t}>} &= tanh \left( W_c \left[ \Gamma_r \times a^{<{t-1}>}, x^{<{t}>}  \right] + b_c \right) \\ \Gamma_u &= \sigma \left( W_u \left[ a^{<{t-1}>}, x^{<{t}>}  \right] + b_u \right) \\ 
 \\  \Gamma_f &= \sigma \left( W_f \left[ a^{<{t-1}>}, x^{<{t}>}  \right] + b_f \right) 
 \\  \Gamma_o &= \sigma \left( W_o \left[ a^{<{t-1}>}, x^{<{t}>}  \right] + b_o \right) \\ c^{<{t}>} &= \Gamma_u \cdot \tilde c^{<{t}>}  + \Gamma_f  \cdot  c^{<{t-1}>}  \\ a^{<{t}>} &= \Gamma_o \cdot tanh\left(c^{<{t}>} \right) \end{align}$$  



1. $$\Gamma_u$$是表示update gate,  $$\Gamma_f$$是表示forget gate, $$\Gamma_o$$是表示output gate. Different from GRU, <span style="color: red">LSTM use separate update and forget gate</span>.
2. One variation: **peephole connection** ($$c^{<{t-1}>}$$): gate value may not only depend on $$a^{<{t-1}>}$$ & $$x^{<{t}>}$$, 也可能depend on $$c^{<{t-1}>}$$, $$\Gamma_o = \sigma \left( W_o \left[ a^{<{t-1}>}, x^{<{t}>}, c^{<{t-1}>}  \right] + b_o \right)$$


![](/img/post/Deep_Learning-Sequence_Model_note/week1pic10.png)
 
 上图 四个小方块依次是 forget gate, update gate, tanh, and output gate

| GRU | LSTM |
| ------:| -----------:|
| $$c^{<{t}>} $$ 等于 $$a^{<{t}>} $$ | $$c^{<{t}>} $$ 不等于 $$a^{<{t}>} $$ |
| update $$c^{<{t}>} $$是由gate $$\Gamma_u$$控制，如果不update, gate = 0, $$c^{<{t}>} $$ = $$c^{<{t-1}>} $$   | 有三个gate  $$\Gamma_u$$,$$\Gamma_f$$,$$\Gamma_o$$ 分别控制update, forget, 和output |

when use GRU or LSTM: isn't widespread consensus in this(some problem GRU win and some problem LST win); Andrew: GRU is simpler model than LSTM and GRU is recently invention than LSTM, <span style="background-color: #FFFF00">easy to build much bigger network</span> than LSFT, LSTM is <span style="background-color: #FFFF00">more powerful and effective</span> since it has three gates instead of two. LSTM is more historical proven， default first try. Now more and more team use GRU, more simpler but work as well.


#### Bidirection RNN && Deep RNNS:

单向的RNN的问题，比如 

He said "Teddy bears are on sale"; <br/>
He said “Teddy Roosevelt was a great President".<br/>
Teddy都是第三个单词且前两个都一样，而只有第二句话的Teddy表示名字

Bidirection RNN: forward prop从左向右 and 从右向左, 每个Bidirection RNN block还可以是GRU or LSTM的block

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic11.png)

 $$ \tilde y^{<{t}>} = g\left( W_y\left[ \overrightarrow a^{<{t}>}, \overleftarrow a^{<{t}>}   \right] + b_y \right)$$  

Lots of NLP problem, BRNN with LSTM are commonly used

<span style="background-color: #FFFF00">Disadvantage</span>: 需要entire sequence of data before you can make prediction; 比如speech recognition: 需要person 停止讲话 to get entire utterance before process and make prediction

![](/img/post/Deep_Learning-Sequence_Model_note/week1pic12.png)

For RRN,  <span style="color:red">三层已经是very deep</span>, $$a^{\left[{1}\right]<{0}>}$$表示第1层第0个input，在output layer也可以有stack recurrent layer，但这些layer没有horizon connection， 每个block 也可以是GRU, 也可以是LSTM, 也可以build deep version of bidirectional RNN, 

比如计算$$a^{\left[{2}\right]<{3}>}$$:   $$a^{\left[{2}\right]<{3}>} = g\left( W_a^2 \left[a^{\left[{2}\right] <{2}>}, a^{\left[ {1}  \right] <{3}>}  \right] \right)$$

<span style="background-color: #FFFF00">**Disadvantage**: computational expensive to train</span>

<br/><br/><br/>

***

## Week2 NLP & Word Embedding


#### Word Embedding:
<span style="background-color: #FFFF00">Word Embedding: </span> 让algorithm学会同义词：比如男人vs女人，king vs queen<br/> 
<span style="background-color: #FFFF00">One hot vector的缺点</span>: 10000中(10000是字典)，只有1个为1表示这个词，不能表示gender. age, fruit..., 因为任何两个one-hot vector的inner product是zero and Eludian distance between any pair is the same.

![](/img/post/Deep_Learning-Sequence_Model_note/week2pic1.png)

- featurized representation (**embeddings**) with each of these words, 比如一个 vocabulary dictionary size 10000, 而每个word 比如有3000 feature, feature vector size = 3000, 
   - e.g. Gender 在 第0个位置, Age 在第2个位置, 比如man 的feature vector \[0] = -1,  woman 的feature vector \[0] = 1, King 的 feature vector \[2] = 0.7, Queen 的 feature vector \[2] = 0.69.  
- Featurized representaion will be similar for analogy. e.g. Apple vs Orange, Kings vs Queens
- In practice, feature(word embedding)  that used for learning won't have a easy interpretation like gender, age, food ...
- <span style="background-color: #FFFF00">T-SNE</span> 把3000vector visualize 到2-D, analogy tends to close

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

Machine translation: RNN先用<span style="color: red">encoder network</span> (input one word 每次), figure out some representation of sentence. 再output 一个 vector代表input sentence，用这个vector作为<span style="color: red">decode netork</span>的开始, 再用decode network 一个一个output 翻译的单词，  <span style="background-color: #FFFF00">difference from synthesizing novel text using language model: 不需要randomly choose translation, want the most likely translation. </span>

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic1.png)

can think machine translation as building a conditional language model. Machine translation model的decode network 很接近language model. Encode network model the probability $$P \left( y ^{<{1}>}, y ^{<{2}>},\cdots,  y ^{<{T_x}>}\vert x ^{<{1}>}, \cdots \right)$$, output the probability of English Translation condition on some input French sentence

<span style="background-color: #FFFF00"> Finding the most likely translation </span>: 不能用random sample output from $$y^{<{t-1}>}$$ to $$y^{<{t}>}$$, 有时候可能得到好的，有时候得到不好的翻译; instead: the goal should maximize the probability  $$P \left( y ^{<{1}>}, y ^{<{2}>},\cdots,  y^{<{T_x}>}\vert x \right)$$

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic2.png)

**Why not Greedy Search** Greedy Search: 在pick 第一个word 后，选择概率最高的第二个单词，再选择概率最高的第三个单词，我们需要的是最大化joint probability $$ P \left( y ^{<{1}>}, y ^{<{2}>},\cdots,  y ^{<{T_x}>} \vert  x \right) $$, 这么选出的word组成的句子 不一定是接近最大的joint proability 的句子; 比如翻译的句子是 Jane is visiting Africa in September这个是perfect翻译, 但是greedy翻译出来的是 Jane is going to be visiting Africa in September. 因为Jane is goint 的概率大于Jane is visiting

不能run 全部combination of words，算哪个概率最大， 比如有10000个词组成的字典，句子长度为10，总共有 $$10000^{10} $$种组合, 所以需要approximate search algorithm，可能不是总成功，不同 try to find sentences to maximize joint conditional probability.


#### Beam Search


**Beam Search Algorithm**, <span style="background-color: #FFFF00"> B = beam width</span>: 不像greedy search 每次只考虑最大可能的一个词，beam search 会考虑最大可能的B个词； 注: 当B=1, 相当于greedy search

Example： B = 3
1. Step1: evulate $$ P\left(y^{<{1}>} \vert x\right) $$, 发现 in, jane, september是根据概率最的可能的三个词, keep [in, jane, september]
2. Step2: evulate $$ P\left(y^{<{2}>} \vert y^{<{1}>}, x\right) $$, $$ P\left(y^{<{1}>},  y^{<{1}>} \vert x\right) = P\left(y^{<{1}>} \vert x\right) P\left(y^{<{2}>} \vert y^{<{1}>}, x\right) $$  比如字典有10000个词，考虑来自step1三个词作为开始，只用考虑10000*3个词, then pick top3; 比如发现算上第二词 最大可能性的三个词 [In september, jane is, jane visit] -> reject september 作为第一个词的可能

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic3.png)


**Length normalization**,

可能$$ P\left(y^{<{t}>} \vert x, y^{<{1}>}, y^{<{2}>}, \cdots, y^{<{t-1}>}    \right) $$概率的乘积越来的越小，不好记录，与其记录乘积，也可记录sum of log, more stable to avoid overflow and numeric rounding error;  <span style="background-color: #FFFF00">problem: 可能prefer更短的句子, 因为probability都是小于1，句子越长概率乘积越小，同样log都是小于0，句子越长sum越小</span>, <span style="background-color: #FFFF00">Solution: normalize 概率，除以句子长度</span> $$ 1/{T_y^\alpha}$$, maybe $$\alpha$$ = 0.7, 当$$\alpha$$=1, complete normalize by length; 当$$\alpha$$=0, $$ 1/{T_y^\alpha} = 1/1$$: not normalized at all. 0.7是between full normalization and no normalization; <span style="color: red">同时alpha也可以作为hyperparameter 用来tune</span>

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic4.png)

比如beam = 3， 看所有top three possibilities of length 1, 2, ...,30 against 上面的normalized probability score, pick the one 有最高score的( <span style="background-color: #FFFF00">highest normalized log likelihood objective</span>) 作为final translation output

How to choose Beam width B? 在实际中可能选择around 10;  100 consider be large; 1000, 3000是not common的, 用越来越大的B, it is diminishing returns; 比如gain很大 当beam从1->3->10, 但是gain不是很大了, 当beam 从1000->3000,
- large B: pro: better result, con: slower
- small B: pro: run faster,  con: worse result

<span style="background-color: #FFFF00"> 不像BFS, DFS. Beam Search runs faster 但是不确保find exact maximum for 最大化 P(y\|x) </span>

**Beam Search Error Analysis**

Example: <br/>
Jane visite l'Afrique en septembre. <br/>
Human 翻译: Jane visits Africa in September ($$y^{*}$$) <br/>
Algorithm 翻译: Jane visited Africa last September ($$\hat y$$)

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic6.png)

用RNN 计算 $$P\left( y^{*} \vert x \right) $$, $$P\left( \hat y \vert x \right) $$
1. Case 1:  $$P \left( y^{*} \vert x \right) $$ > $$P\left(\hat y \vert x \right)$$ : Beam choose $$\hat y$$, 但是 $$y^{*}$$ attains 更高的 P(y\|x); <span style="background-color: #FFFF00"> Beam search is at fault </span>
2. Case 2: $$P\left( y^{*}\vert x \right) $$ <=  $$P\left(\hat y \vert x \right)$$:  $$y^{*}$$  better translation than $$\hat y$$, 但是 RNN预测相反, <span style="background-color: #FFFF00"> RNN is at fault </span>


![](/img/post/Deep_Learning-Sequence_Model_note/week3pic5.png)



#### Bleu Score

given French sentence, 有几个英语翻译，how to measure? Bleu: Bilingual evalutation understudy

French: Le chat est sur le tapis <br/>
Reference 1: The cat is on the mat.<br/>
Reference 2: There is a cat on the mat.<br/>
MT output: the the the the the the the.<br/>

**Precision**: each word either appear in reference 1 or reference 2 / total word.  MT = $$\frac{7}{7} = 1 $$  <span style="background-color: #FFFF00"> (not a particularly useful measure) </span><br/>
**Modified Precision**: credit only up to maximum appearance in reference 1 or reference. 上面MT翻译中 the 在1中出现了2回, MT = $$\frac{2}{7} $$

French: Le chat est sur le tapis <br/>
Reference 1: The cat is on the mat.<br/>
Reference 2: There is a cat on the mat.<br/>
MT output: the cat the cat on the mat.<br/>

**Bleu score on bigrams**: 两个两个词连在一起看有没有在reference 1 or 2中出现， 比如the cat, cat the, cat on...   MT $$ = \frac{4}{6} $$,


| Context | Count | Count Clip |
| ------:| -----------:| ------:|
|The cat | 2 | 1  |
|cat the | 1 | 0 |
|cat on | 1 | 1 |
|on the | 1 | 1 |
|the mat | 1 | 1 |


| unigram | n-gram |
| ------:| -----------:|
|$$\displaystyle p_1 = \frac{ \sum_{unigram \in \hat y }^{} { Count_{clip} \left( unigram \right)} }{ \sum_{unigram \in \hat y }^{} { Count\left( unigram \right)} }  $$ | $$ \displaystyle p_n = \frac{ \sum_{unigram \in \hat y }^{} { Count_{clip} \left( n-gram \right)} }{ \sum_{unigram \in \hat y }^{} { Count\left( n-gram \right)} }  $$ |

 <span style="background-color: #FFFF00"> 如果机器翻译的跟reference 1 or reference 2完全一样, $$P_1$$ and $$P_n$$ 都等于1</span>

 BP: 表示brevity penalty: if output is short, 容易得到high precision; BP is adjustment factor 避免too short

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic7.png)

<span style="background-color: #FFFF00"> Bleu Score 应用于machine translation or 给图片起标题 (image caption); not use in speech recognition, 因为speech recognition一般都有one ground truth </span>


#### Attention Model

<span style="background-color: #FFFF00">problem with encoder & decoder network:</span> given long sentence, encode 只能读完句子所有内容后, 再通过decoder进行翻译输出;  encoder & decoder network 对于<span style="background-color: #FFFF00"> 短的句子和很长的句子效果不好</span>。

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic8.png)


- 用bidirectional RNN, 对于不同位置, 可以得到rick features around the word; 
- 再用另一组rnn generate translation, 用$$s^{<{t}>}$$ 表示hidden state,  $$s^{<{2}>}$$ 需要 $$s^{<{1}>}$$ (generate的第一个词） 作为input。 
- 比如当生成第一个词时, 不太用着at the end of 句子的word, 用attention weight 比如$$\alpha^{<{1,1}>}$$表示产生第一个词时，来自一个features (bidirection rnn output的) 的weight, $$\alpha^{<{1,2}>}$$ how much weight(attention) need to put on second input to generate first word;  $$\alpha^{<{t,t'}>}$$ amount of attention $$y^{<{t}>}$$ should pay to $$a^{<{t'}>}$$
- 最后generate EOS

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic9.png)

- $$ \overrightarrow a^{<{0}>}$$,  $$\overrightarrow a^{<{6}>}$$是zero vector, 用$$ a^{<{t}>}$$ 表示foward 和backword features
- $$ \sum_{ t }^{} {\alpha^{<{1, t'}>}} = 1$$ all weights which used to generate 第一个的词的和等于1 (适用于每个词)
- content 是weight sum of activation ($$a^{<{t}>}$$)
- compute alpha  $$\alpha^{<{t, t'}>}$$用softmax 
- generate $$e^{<{t, t'}>}$$用smaller neural network(通常只有一个hidden layer): input: feature from $$\alpha^{<{t'}>}$$  and $$s^{<{t-1}>}$$ is hidden state 来自上个rnn output, $$s^{<{t-1}>}$$也是现在rnn的input
- <span style="background-color: #FFFF00">Downside</span>: take quadratic time to run this algorithm

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic10.png)

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic11.png)

#### Speech recognition


**CTC cost for speech recognition (CTC: connectionist temporal classification)** Rule: collapse repeated characters not separated by "blank"

In speech recognition, input time steps are much bigger than output time steps; 比如10 seconds audio, feature come at 100 hertz so 100 samples每秒; 10 seconds audio clip has 1000 inputs; \_ : called special character, \|\_\|: space character; 为了让 1000 inputs has 1000 output,生成words like 下面图片中, 但把output word ( ttt_h_eee \_ \_ \_ \|_\| \_ \_ \_ qqq \_ \_ ) collapse一起 (the q), end up much shorter output 文本

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic12.png)

**Trigger Word Detection** 比如amazon echo; 用audio clip 计算spectrogram to generate features; to define target label y before trigger word as 0, after as 1

![](/img/post/Deep_Learning-Sequence_Model_note/week3pic13.png)

[pic3]: https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep_Learning-Sequence_Model_note/week1pic3.png
