---
layout:     post
title:      "Mathematics for Machine Learning 笔记"
subtitle:   "机器学习数学基础  —— 学习笔记"
date:       2018-10-20 19:00:00
author:     "Becks"
header-img: "img/post-bg2.jpg"
catalog:    true
tags:
    - Coursera
    - Machine Learning
    - 学习笔记
---
> note from Coursera Mathematics for Machine Learning
> 

## Course 1: Linear Algebra

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#### vector

<span style="background-color: #FFFF00">property: </span>

Communicative:  $$ \vec v \cdot \vec w  =  \vec w \cdot \vec v $$ <br/>
Distributive:  $$ \left(\vec w + \vec v \right)\cdot \vec x  =  \vec w \cdot \vec x + \vec v \cdot \vec x  $$<br/>
Associative over scaler multiplication:  $$ \left( c \vec v \right) \cdot \vec w  =  c \left( \vec v \cdot \vec w \right)  $$<br/>
Dot product self is the length square: $$ c\vec v \cdot \vec v  =  ||v||^2 = v_1^2 + v_2^2 + ... + v_n^2  $$<br/>
Cosine:  $$  \vec a \cdot \vec b  =  ||a||^2||b||^2 cos\theta  $$<br/>
projection length:  $$  \vec proj_{L} \left(\vec x\right) =   \frac{ \vec x \cdot \vec v  }{ ||\vec v|| } $$<br/>
vector projection:  $$  \vec proj_{L} \left(\vec x\right) =  c \vec v =  \frac{ \vec x \cdot \vec v  }{ \vec v \cdot \vec v } \vec v $$<br/>