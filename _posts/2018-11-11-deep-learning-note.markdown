---
layout:     post
title:      "Deep Learning Summary"
subtitle:   "深度学习 Deep Learning - Note"
date:       2018-11-11 12:00:00
author:     "Becks"
header-img: "img/post-bg-city-night.jpg"
catalog:    true
tags:
    - Coursera
    - Deep Learning
    - Machine Learning
    - 总结
    - 学习笔记
---

> note from Coursera Deep Learning

| Input(x)  | Output(y)   |  application  | Type of Neuron Network  |
| ------------ | ------------ | ------------ | ------------ |
| Home Features  | Price   | Real Estate   | Standard Neural Network   |
| Ad, User Info  | Click on ad(0,1)   | Online Advertising   | Standard Neural Network   |
| Image | Object(1,...,1000)  | Photo Tagging  | Convolution NN  |
| Audio | Text Transcript  | Speech recognition  | Recurrent NN  |
| ENglish | Chinese  | Machine Translation  | Recurrent NN  |
| Image, Radar Info | Position of cars  | Autonomous Driving  | Custom / Hybrid NN  |

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


Structured Data: Database data: 比如housing price prediction 输入值是size, #bedroom, 输出值是 price
Unstructure Data: 比如audio, image, Text

## Standard Neural Network

#### Cost Function

$$\mathscr{L} \left(\hat y, y \right) = - ylog\left( \hat y \right) + \left( 1- y \right) log\left( 1 - \hat y \right) $$

Loss function measures how good $$\hat y $$ when true label is y

Loss (error) function 不用 $$L\left(\haty, y) = \frac{1}{2}\left(\hat y - y\right)^2$$ (Optimization is not a convex function having many local optimum, so Grandient Descent may not find global optimum)
