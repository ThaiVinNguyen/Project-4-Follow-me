[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Project 4: Follow Me
##### Udacity Robotics Nanodegree
###### Fuly 2018

[image 0]: ./docs/misc/sim_screenshot.png
[image 1]: ./docs/misc/fcn.png
[image 2]: ./docs/misc/3layer.png
[image 3]: ./docs/misc/epoch50.png
![simulation][image 0]


### 1. Overview

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

## 2. Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## 3. Implement the Segmentation Network

1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network by **1080Ti Nvidia GPU** that I had.
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## 4. My model network

##### Build the Model
![model][image 1]
Because we're challenged with the task of not just understanding what is in an image, but we need to figure out where the object is in the image, we're going to be using a network fully of convolutional layers ***(FCN)***.

We'll use a method of first extracting important features through **encoder**, then upsample into an output image in **decoder**, finally assigning each pixel to one of the classes

###### Network Architecture
![architecture][image 2]
I make 3 options and i choice that the best network.
  1. Input > 32 > 64 > 128 > 64 > 32 > Output` and choice **learning rate = 0.001**
Link code: [Model training 02](code/model_training_02.ipynb)
  2. Input > 64 > 128 > 256 > 128 > 64 > Output` and choice **learning rate = 0.001**
Link code: [Model training final](code/model_training_final.ipynb)
  3. Input > 64 > 128 > 256 > 128 > 64 > Output` and choice **learning rate = 0.01**
Link code: [Model training 01](code/model_training_01.ipynb)

###### Input

We know our original `256x256x3` images. But I resized and Input layer size `input =  160,160,3`.

###### Encoder
```
def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

The start of our model looks like this:
```
def fcn_model(inputs, num_classes):

    # ENCODER
    encoder1 = encoder_block(input_layer=inputs, filters=64, strides=2)
    encoder2 = encoder_block(input_layer=encoder1, filters=128, strides=2)
```
Then to the middle portion of the network.

###### 1x1 Convolution

Then we've used a **1x1 convolutional** layer in the middle of the network, it allows use to take in any sized image, as opposed to a fully connected layer that would require a specific set of input dimensions.

The next portion of our `fcn_model` function is:
```
def fcn_model(inputs, num_classes):
    # See encoder portion from above!

    # 1x1 Convolution layer using conv2d_batchnorm()
    conv_norm = conv2d_batchnorm(input_layer=encoder2, filters=256, kernel_size=1, strides=1)
```

Finally to the next section of the model.

###### Decoder

```
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```


In addition to `bilinear_upsample`, we'll use concatenation with `layers.concatenate`, which will help us implement skip connections, as well as the `separable_conv2d_batchnorm` function. These functions will make up our `decoder_block` function:
```
def decoder_block(small_ip_layer, large_ip_layer, filters):

    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)

    # Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsampled_layer, large_ip_layer])

    # Add some number of separable convolution layers
    separable_layer = separable_conv2d_batchnorm(concat_layer, filters)
    output_layer = separable_conv2d_batchnorm(separable_layer, filters)


    return output_layer
```


The choice for the **decoder** was simple, in that I simply scaled back up from the middle **1x1** convolutional layer to the final output image size. In this moment, the term deconvolution feels fitting, as it is what I had in mind while building the decoder(however, I know it is a contentious term).

The next portion of the `fcn_model` function is:
```
def fcn_model(inputs, num_classes):
    # See Encoder and 1x1 convolutional sections from above!

    # DECODER
    decoder1 = decoder_block(small_ip_layer=conv_norm, large_ip_layer=encoder1, filters=128)
    decoder2 = decoder_block(small_ip_layer=decoder1, large_ip_layer=inputs, filters=64)
```

###### Output

We finally apply our favorite activation function, **Softmax**, to generate the probability predictions for each of the pixels.
```
return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder2)
```
This completes our `fcn_model` function to be...
```
def fcn_model(inputs, num_classes):
    # Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder1 = encoder_block(input_layer=inputs, filters=64, strides=2)
    encoder2 = encoder_block(input_layer=encoder1, filters=128, strides=2)


    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_norm = conv2d_batchnorm(input_layer=encoder2, filters=256, kernel_size=1, strides=1)

    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder1 = decoder_block(small_ip_layer=conv_norm, large_ip_layer=encoder1, filters=128)
    decoder2 = decoder_block(small_ip_layer=decoder1, large_ip_layer=inputs, filters=64)


    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder2)
```

This final layer will be the same height and width of the input image, and will be `c-classes` deep. In our example, we have 3 classes deep because we are aiming to segment our pixels into one of the three classes!

## 5.Training

###### Hyperparameters
After trying a variety of values for `num_epochs`. I decided on the following training values:
```
learning_rate = 0.001
batch_size = 64
num_epochs = 50
steps_per_epoch = 65 #(4131/batch_size)
validation_steps = 50
workers = 4
```
**learning_rate**: The learning rate is what we multiply with the derivative of the loss function, and subtract it from its respective weight. It is what fraction of the weights will be adjusted between runs.

**batch_size**: The batch size is the number of training examples to include in a single iteration. I set this as 64. I could have also used 128 or 32, as long as I adjusted steps per epoch also.

**num_epochs**: An epoch is a single pass through all of the training data. The number of epochs sets how many times you would like the network to see each individual image.

**steps_per_epoch**: This is the number of batches we should go through in one epoch. In order to utilize the full training set, it is wise to make sure `batch_size*steps_per_epoch = number of training examples`.

**validation_steps**: Similar to steps per epoch, this value is for the validation set. The number of batches to go through in one epoch.

**workers**: This is the number of processes we can start on the CPU/GPU.

With these parameters, anything beyond 50 epochs would be excessive and wasteful. Each epoch was running about 93-94s on average.
![Epochs 50][image 3]

## 6. Evaluation

For the first evaluation, I choice model: Input > 32 > 64 > 128 > 64 > 32 > Output with **learning_rate=0.001** and the number of epochs to 50. The final_score came out as **~0.365**.So I continued training with more epochs. Link code: [Model training 02](code/model_training_02.ipynb)

Next, I kept the `learning_rate=0.001` I choice model: Input > 64 > 128 > 256 > 128 > 64 > Output and the number of epochs to 50. The `final_score` came out as **~0.427**, Success! Link code: [Model training final](code/model_training_final.ipynb)
Then, I am trying to change `learning_rate=0.01` with model: Input > 64 > 128 > 256 > 128 > 64 > Output and the number of epochs to 50. The `final_score` came out as **~0.415**, Also success! Link code: [Model training 01](code/model_training_01.ipynb)

### 7. Future Enhancements

Adding data to train on would be a significant improvement. This would give the network more opportunity to learn.

In order for this Deep Neural Network to be used to follow another target: such as a cat or a dog, it would just need to be trained on a new set of data. Also, the encoder and decoder layer dimensions may have to adjusted depending on the overall depth of the network.



