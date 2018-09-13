---
layout:     post
title:     "Deep Learning - Sequence Model 笔记"
subtitle:   ""
date:       2018-09-12 19:00:00
author:     "Becks"
header-img: "img/post-bg-nextgen-web-pwa.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Coursera
    - Deep Learning
	- Machine Learning
    - Sequence Model
---

##Week1 Recurrent Neural Networks
Examples of sequence data:
                
1. Speech Recognition: given audio clip X ----> text Y (both input and output sequence data)
2.  music generation:  output is sequence(音乐), input maybe music 的类型 or nothing
3. sentiment classfication: 像影: "there is nothing to like in this move" ---> 这样的review是几分
4. DNA sequence Analysis:  given DNA AGCCTGA... ---> label which part of DNA sequence corresponds to a protein
5. machine translation
6. video activty recogntion : 给一段录像 -> 在running
7. Name entity recognation：给一段话 --> 挑出人名


sometimes 输入X 和 输出Y 可以是不同的长度，sometimes X和Y(example 4,7)是同样长度的, sometimes 只有X或者只有Y是sequence的 (example 2)

#### notation: 
[![](https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep%20Learning%20-%20Sequence%20Model%20note/week1pic1.png)](https://raw.githubusercontent.com/beckswu/beckswu.github.io/master/img/post/Deep%20Learning%20-%20Sequence%20Model%20note/week1pic1.png)
example: given a sentence 判断哪个是人名
$$x^{({i})<{t}>}$$:  表示第i个training example 中第t个word, t 表示temporal sequences althought whether sequences are temporal one or not
$$y^{({i})<{t}>}$$:  表示第i个training example 中第t个word的输出label
$$T_x^{i}$$:  表示第i个training example的长度
$$T_y^{i}$$:  表示第i个training example的ouput长度




