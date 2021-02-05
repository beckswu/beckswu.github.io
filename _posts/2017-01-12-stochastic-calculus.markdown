---
layout:     post
title:      "Stochastic Calculus"
subtitle:   "Stochastic Calculus, Itô's lemma"
date:       2017-01-12 20:00:00
author:     "Becks"
header-img: "img/post-bg-city-night.jpg"
catalog:    false
tags:
    - Stochastic Calculus
    - Financial Mathematics
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


## 1 Stochastic process

In probability theory and related fields, **a stochastic or random process** is a mathematical object usually defined as a family of random variables. Many stochastic processes can be represented by time series. However,<span style="color:red"> a stochastic process is by nature continuous while a time series is a set of observations indexed by integers. A stochastic process may involve several related random variables.</span>


A stochastic process is defined as a collection of random variables defined on a common probability space $$ {\displaystyle (\Omega ,{\mathcal {F}},P)}$$, where $${\displaystyle \Omega }$$  is a sample space, $${\displaystyle {\mathcal {F}}}$$ is a $${\displaystyle \sigma }$$-algebra, and $${\displaystyle P}$$ is a probability measure; and the random variables, indexed by some set $${\displaystyle T}$$, all take values in the same mathematical space $${\displaystyle S}$$, which must be measurable with respect to some $${\displaystyle \sigma }$$ -algebra $${\displaystyle \Sigma }$$ .

In other words, for a given probability space {\displaystyle (\Omega ,$${\mathcal {F}},P)}$$ and a measurable space $${\displaystyle (S,\Sigma )}$$, a stochastic process is a collection of $${\displaystyle S}$$-valued random variables, which can be written as:

$${\displaystyle \{X(t):t\in T\}.}$$


#### Random walk


Random walks are stochastic processes that are usually defined as <span style="color:red">sums of iid(Independent and identically distributed) random variables</span> or random vectors in Euclidean space, so they are processes that change in discrete time. 

A classic example of a random walk is known as the <span style="background-color:#FFFF00">**simple random walk**</span>, which is a stochastic process in discrete time with the integers as the state space, and is based on a Bernoulli process, where each Bernoulli variable takes either the value positive one or negative one. In other words, the simple random walk takes place on the integers, and its value increases by one with probability, say, $${\displaystyle p}$$, or decreases by one with probability $${\displaystyle 1-p}$$, so the index set of this random walk is the natural numbers, while its state space is the integers. If the $${\displaystyle p=0.5}$$, this random walk is called a <span style="color:red">symmetric random walk</span>.


#### Wiener process


In mathematics, the Wiener process is also called **Brownian motion** is one of the best known Lévy processes (càdlàg stochastic processes with stationary independent increments) and occurs frequently in pure and applied mathematics, economics, quantitative finance, evolutionary biology, and physics.



## Geometric Brownian motion

Ito’s lemma is the chain rule:

$$du \left(X_t, t \right) = \frac{ \partial du }{ \partial t} dt + \frac{ \partial du }{ \partial x} dX_t + \frac{1}{2} \partial^2_x u dX_t^2 dt$$


Geometric Brownian motion is the solution to the SDE

$$ dS = \mu S_t dt + \sigma S_t dW_t$$

$$dlog\left( S_t \right) = \frac{} $$