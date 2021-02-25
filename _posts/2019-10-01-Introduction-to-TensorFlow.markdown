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


## Convolution NN 

**fashion_mnist**: [Github](https://github.com/zalandoresearch/fashion-mnist): 


- include 70k images spread across 10 categories of clothing, image are scaled at `28 x 28` gray scale (只有一个channel, instead of 3)
- fashion_mnist dataset, 60000 of 70000 used to train, 10000 can be used for testing
- train_labels, test_labels are number rather than text, which can reduce bias 

```python
# fashion_mnist.load_data() return 4 lists

fashion_mnist = keras.datasets.fashion_mnist
(training_iamges, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
```

#### Callback

- callback function doesn't need to be in a separate file. 
- init will implement ```on_epoch_end``` function which get called by the callback whenever epoch ends. It aslo sends log object which contains lots of information about current state of training

```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
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

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

```



#### Intermidate Layer Output 

```python
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

for i in range(0,4):
    f1 = activation_model.predict(test_images[0].reshape(1, 28, 28, 1))[i]
    plt.imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
    print("f1.shape",f1.shape)
"""

 f1.shape (1, 26, 26, 64)  -> Conv2D

 f1.shape (1, 13, 13, 64)  -> MaxPooling2D

 f1.shape (1, 11, 11, 64)  -> Conv2D

 f1.shape (1, 5, 5, 64)    -> MaxPooling2D

"""
```

#### Visualize Intermediate Representations

```python
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = tf.keras.models.Sequential([
    # This is the first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

successive_outputs = [layer.output for layer in model.layers]
layer_names = [layer.name for layer in model.layers]

visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.

train_human_names = os.listdir(train_human_dir)
img_path = os.path.join(train_human_dir, train_human_names[0])

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image

x = img_to_array(img)  # Numpy array with shape (300, 300, 3)

x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 300, 300, 3)


# Rescale by 1/255

x /= 255
successive_feature_maps = visualization_model.predict(x)

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers

    n_features = feature_map.shape[-1]  # number of features in feature map

    print(layer_name, feature_map.shape)

    # The feature map has shape (1, size, size, n_features)

    size = feature_map.shape[1]
    # We will tile our images in this matrix

    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      #  把一个feature 都horizontal display 出来, normalize feature

      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid

    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

output:

```
/tmp/horse-or-human/horses/horse49-5.png
(1, 300, 300, 3)
conv2d (1, 298, 298, 16)
max_pooling2d (1, 149, 149, 16)
conv2d_1 (1, 147, 147, 32)
max_pooling2d_1 (1, 73, 73, 32)
```

![](/img/post/tensorflow/c1_2.png)


#### Image Generater

have sub-directory for training and validation, img generator will create for those images and auto label. <span style="background-color:#FFFF00">The name of subdirectory will be the label </span>

![](/img/post/tensorflow/c1_1.png)

- `steps_per_epoch = TotalTrainingSamples / TrainingBatchSize`
- `validation_steps = TotalvalidationSamples / ValidationBatchSize`

`batch_size` 和 `steps_per_epoch` 是为了spare computer memory,比如想run 一个epoch 用1024个training examples, 但computer memory 不允许，所以就每次run 128个，run 8回


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images, always point to directory not subdirectory

        target_size=(300, 300),  # All images will be resized to 300x300

        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')


test_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator

train_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),  # All images will be resized to 300x300

        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

#print accuracy, loss for training and validation

history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  #load 8 batches (8 x 128) from train_generator into one model training

      epochs=15,
      validation_data = validation_generator,
      validation_steps = 8, #When each epoch ends, the validation generator will yield validation_steps batches, run 8个 validation batches

      verbose=2) # epoch = 2 hide a little animation hiding epoch progress
      
```


#### Process Image

```
from tensorflow.keras.preprocessing import image

img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x, batch_size=10)

```

#### Data Augmentation


- Not to add image directly on the drive. augmentation will take place in memory and not overriding the data 
- Data Augmentation **doesn't add more picture**, but for each training epoch to change the image based on parameter specified to *generate more scenarios in the images* 
- 比如下图猫的耳朵是竖的，如果在test set 耳朵不是完全竖的, 可能认不出来了 => overfitting. But if training set include rotated image, 也许就可以认出耳朵斜的的 

![](/img/post/tensorflow/c2_1.png)

**Shear Image**

下图左侧的是在training set, 右侧的人姿势(orientation)不在training set中，所以把左侧的image shear 一下会得到类似的姿势（pose)

![](/img/post/tensorflow/c2_2.png)

![](/img/post/tensorflow/c2_3.png)


**Zoom Image**: 左侧image in training set, 右侧的人face right, 如果zoom 左侧的会得到类似右侧的图片. If zoom, may see some generalized example 

![](/img/post/tensorflow/c2_4.png)

![](/img/post/tensorflow/c2_5.png)

**Horizontal Flipping**: training set 只有left-arm raiser 没有right-arm raiser, flip horizontally 可以得到right-arm raiser

![](/img/post/tensorflow/c2_6.png)

![](/img/post/tensorflow/c2_7.png)

```python
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 40, #image  will rotate between 0 - 40 degree

    width_shift_range = 0.2,  # randomly shift up to 20% horizontal

    height_shift_range = 0.2, # randomly shift up to 20% vertically

    shear_range = 0.2, #shear randomly up to  20% percent of the image

    zoom_range =0.2, #0.2 is relative image you will zoom in,random amount up to 20% of image

    horizontal_flip = True, # image will be flip randomly 

    fill_mode = 'nearest' # fill nay pixels that maybe lost by the operations. nearest: use neighbors of that pixel to try to keep uniformity

    #with the neighbors of that pixels to try and keep uniformity

)
```

<span style="background-color:#FFFF00">如果training set introduce **randomness** by data augmentation but validation set doesn't have the same randomness, validation loss will fluctuate around </span> the result 就会像下图 (training set 和 test set 的image 很像很接近, 比如train set & test set 人都是center and stand in the image, zoom 和rotate 后的图片会显得人不是站着的，cannot generalize result) 
- <span style="background-color:#FFFF00">**so must have broad image for training and testing, 否则data augmentation won't help**</span>



![](/img/post/tensorflow/c2_8.png)


#### Transfer Learning 

Lock all layers which are are trained on very large datasets by others, and only train Dense layer with your own data

Below use [`ImageNet`](http://image-net.org/) which are trained on 1.4 million images in a 1000 different classes

```python

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150,150,3), 
                                include_top = False, # specify that ignore fully-connected layer at the top and straight to the convolution (Inception V3 has a fully-connected layer at the top)

                                weights = None) # you don't want to use the built-in weights but use the snapshot you just download it
                                
pre_trained_model.load_weights(local_weights_file)


#****************************************# 

# Iterate through Layers  to lock layers without changing parameters

#****************************************# 

for layer in pre_trained_model.layers:  
  layer.trainable = False # specify not to train within the local

print(pre_trained_model.summary())
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

```

#### Dropout 

The idea behind Dropouts is that they remove a random number of neurons in your neural network. This works very well for two reasons: 
- <span style="color:red">Neighboring neurons often end up with similar weights</span>, which can lead to overfitting, so dropping some out at random can remove this. 
- <span style="color:red">Often a neuron can over-weigh the input from a neuron in the previous layer.  Can’t rely on any one feature and weights won't put much weight on any one input, so have to spread out weights</span>, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit! 

<span style="background-color:#FFFF00">**Disadvantage**: cost function J is no-longer well defined</span>, at every iteration, you randomly kill some node。 It is hard to double check gradient descent   function 每个 iteration 都 decrease

```Python
x = layers.Dropout(0.2)(x) # Parameter is between 0 and 1, it's fraction unit to drop

```

<span style="background-color:#FFFF00">比如遇到下图, validation accuracy diverge from training accuracy through the time，it worths trying **Dropout**</span>

![](/img/post/tensorflow/c2_9.png)

After applying Dropout

![](/img/post/tensorflow/c2_10.png)



## Convolutional NN Layers

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


## Link / Reference 

- [AI Google:  Responsible AI practices](https://ai.google/responsibilities/responsible-ai-practices/)
- [Image Filtering](https://lodev.org/cgtutor/filtering.html)
- [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
- [Understanding RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [Image Augmentation](https://github.com/keras-team/keras-preprocessing ): like rotating 90 degrees of images. It doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset