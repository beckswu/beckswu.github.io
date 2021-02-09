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

## Convolutional NN

####  keras.layers.Conv2D

- `filters`:	Integer, the dimensionality of the output space 
- `kernel_size`	An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
- `strides`	An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
- `padding`	one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input
- `input_shape` (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

```
tf.keras.layers.Conv2D(
    filters, # Integer, the dimensionality of the output space
    kernel_size,#filter   specifying the height and width of the 2D convolution window. 
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
```

#### CONV2D

-  filter order is  [filter_height, filter_width, in_channels, out_channels]
-  input order is[batch_shape, in_height, in_width, in_channels]

```python
initializer=tf.initializers.GlorotUniform(seed = 0)
W1 = tf.Variable(initializer([4, 4, 3, 8]),name="W1")
X = np.random.randn(2,64,64,3).astype(np.float32)

Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
# Z1 Shape: (2,64,64,8)
```

#### Dense

Just your regular densely-connected NN layer.

Dense implements the operation:` output = activation(dot(input, kernel) + bias)` where `activation` is the element-wise activation function passed as the `activation` argument, kernel is a weights matrix created by the layer, and bias is a `bias` vector created by the layer (only applicable if `use_bias` is `True`).

Note: If the input to the layer has a rank greater than 2, then `Dense` computes the dot product between the `inputs` and the `kernel` along the last axis of the `inputs` and axis 1 of the `kernel` (using `tf.tensordot`). For example, if input has dimensions `(batch_size, d0, d1)`, then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are `batch_size * d0` such sub-tensors). The output in this case will have shape `(batch_size, d0, units)`.

```
tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```



#### Flatten
Flattens the input. Does not affect the batch size.

```
tf.keras.layers.Flatten()(X)
```

#### Max_pool

```python
#A1 Shape:  (2, 64, 64, 8)
P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
# P1 Shape (2,8,8,8)
```

[Max_pool layer padding Difference between `padding='SAME'` and `padding = 'VALID'`](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)

"VALID" = without padding:

```
   inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                  |________________|                dropped
                                 |_________________|
```
"SAME" = with zero padding:

```
               pad|                                      |pad
   inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
               |________________|
                              |_________________|
                                             |________________|
```

In this example:

 - Input width = 13
 - Filter width = 6
 - Stride = 5
  
Notes:

"VALID" only ever drops the <span style="color:red">right-most columns</span> (or bottom-most rows).
"SAME" tries to pad evenly left and right, but <span style="color:red">if the amount of columns to be added is odd, it will add the extra column to the right,</span> as is the case in this example (the same logic applies vertically: there may be an extra row of zeros at the bottom).

About the name:

- With "SAME" padding, if you use a stride of 1, the layer's outputs will have the **same** spatial dimensions as its inputs.
- With "VALID" padding, there's <span style="color:red">no "made-up" padding inputs</span>. The layer only uses **valid** input data.

#### MaxPool2D

- `pool_size`	integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
- `strides`	Integer, tuple of 2 integers, or None. Strides values. Specifies how far the pooling window moves for each pooling step. If None, it will default to pool_size.
- `padding`	One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.

```
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(1, 1), padding='same')
```



## Model


#### Model

Note <span style="color:red">Input dimension is one training example dimension, should not include batch size</span>

Two way to initiate Model

1 - With the "Functional API"

```
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

2 - By subclassing the Model class: in that case, you should define your layers in `__init__` and you should implement the model's forward pass in `call`.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

"""
If you subclass Model, you can optionally have a training argument (boolean) in call, which you can use to specify a different behavior in training and inference:
"""

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)


model = MyModel()
```


#### Compile

- `optimizer`:	String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
- `loss`: String (name of objective function), objective function or tf.keras.losses.Loss instance. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`. You can also pass a list (len = len(outputs)) of lists of metrics such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or `metrics=['accuracy', ['accuracy', 'mse']]`
- `metrics`: 
```python
compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
)

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

#### fit 

- Return: A `History` object. Its History.history attribute is a record of training <span style="color:red">loss values and metrics values</span> at successive epochs, as well as validation loss values and validation metrics values (if applicable).

```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
```

```python

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

# history.history
"""
{'loss': [0.344687819480896, 0.15941613912582397],
 'sparse_categorical_accuracy': [0.9019200205802917, 0.9523000121116638],
 'val_loss': [0.1892719268798828, 0.1630939543247223],
 'val_sparse_categorical_accuracy': [0.9483000040054321, 0.9488000273704529]}
 """
```


#### evaluate

- return Scalar <span style="color:red">test loss</span> (if the model has a single output and no metrics) or <span style="color:red">list of scalars</span> (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.

```
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# test loss, test acc: [0.17500483989715576, 0.9459999799728394]


predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
# predictions shape: (3, 10)
```

#### Predict

- `batch_size`: Integer or None. Number of samples per batch. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of dataset, generators, or keras.utils.Sequence instances (since they generate batches).
verbose	Verbosity mode, 0 or 1.
- `steps`	Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None. If `x` is a tf.data dataset and steps is None, predict will run until the input dataset is exhausted.
- `return`: Numpy array(s) of predictions.


```python
return = predict(
    x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)
```


## Image

#### load_img

Loads an image into PIL format.


- `color_mode`	One of "grayscale", "rgb", "rgba". Default: "rgb". The desired image format.
- `target_size`	Either None (default to original size) or tuple of ints (img_height, img_width).
- `interpolation`	Interpolation method used to resample the image if the target size is different from that of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are also supported. By default, "nearest" is used.

```
tf.keras.preprocessing.image.load_img(
    path, color_mode='rgb', target_size=None,
    interpolation='nearest'
)
```


#### img_to_array

Converts a PIL Image instance to a Numpy array.

- `img`:	Input PIL Image instance.
- `data_format`:	Image data format, <span style="color:red">can be either "channels_first" or "channels_last"</span>. Defaults to None, in which case the global setting tf.keras.backend.image_data_format() is used (unless you changed it, it defaults to "channels_last").
- `dtype`:	Dtype to use. Default to None, in which case the global setting tf.keras.backend.floatx() is used (unless you changed it, it defaults to "float32")

```
tf.keras.preprocessing.image.img_to_array(
    img, data_format=None, dtype=None
)

from PIL import Image
img_data = np.random.random(size=(100, 100, 3))
img = tf.keras.preprocessing.image.array_to_img(img_data)
array = tf.keras.preprocessing.image.img_to_array(img)

```


#### preprocess_input

The images are converted from **RGB** to **BGR**(tenserflow model process based on BGR), then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.

```
tf.keras.applications.resnet.preprocess_input(
    x, data_format=None
)
```