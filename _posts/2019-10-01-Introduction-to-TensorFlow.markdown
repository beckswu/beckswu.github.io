---
layout:     post
title:      "Introduction to TensorFlow"
subtitle:   "Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning"
date:       2019-10-01 20:00:00
author:     "Becks"
header-img: "img/post-bg3.jpg"
catalog:    true
tags:
    - Machine Learning
    - Deep Learning
    - Coursera
    - 学习笔记
    - Learning Note
---


## A New Programming Paradigm



Keras: API in tensorflow, make it easy to define neural network

- ```Dense``` to define a layer of connected neurons
- loss function and  optimiziers ```compile```
- trainning model ```fit```
- predict ```predict```

```python
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) 
# only one neuron 
# only one Dense here, so there are only one layer and only one unit 
```

loss function and  optimiziers

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
#sgd: stochastic gradient descent 
```

Nerual network has no idea of relationship between X and Y. So it guess, e.g. ```Y = 10X - 1```. Loss function measure how guess is good or bad, then give data to optimizer to figure out next guess. The logic is each guess should be better than the one before. When guess better and bettern, the term convergence is used


```python
xs = np.array([-1,0,1,2,3,4], dtype = float)
ys = np.array([-3,-1,1,3,5,7], dtype = float)
model.fit(xs,ys,epochs = 500)
#describe training loop 500 times
print(model.predict([10,0]))
```

training result is not 19, but close to 19, 因为 一: training data is few. 二: when try to figure out answer, deal in probability 


## Introduction to computer Vision

比如 has images 28 by 28 pixels = 786 bytes are needed to store entire image. each pixels can have value from 0 to 255 

- ```fashion_mnist.load_data()``` return 4 lists to us
- fashion_mnist dataset, 60000 of 70000 used to train, 10000 can be used for testing
- train_labels, test_labels are number rather than text, which can reduce bias 

```python
fashion_mnist = keras.datasets.fashion_mnist
(training_iamges, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

注意最后一个layer has 10 neurons because we have 10 classes in dataset. First layer is 28 by 28, because input layer is 28 by 28 pixel. <span style="background-color:#FFFF00">Flatten is to take 28 by 28 square into a simple linear array</span>. Middle layer(128 neurons) is hdden layer

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
```

for every epoch, can callback to a code function. 比如到了一定accuracy, can cancel the training 

- callback function doesn't need to be in a separate file. 
- init will implement ```on_epoch_end``` function which get called by the callback whenever epoch ends. It aslo sends log object which contains lots of information about current state of training

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4>)：
            print('\nLoss is low so cancelling training')
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
#normalized data
training_iamges = training_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5. callbacks = [callbacks])
```
