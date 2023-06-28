from clearml import Task

hyperparams = {'TRAIN_SIZE':1000, 
               'BATCH_SIZE':10,
               'EPOCHS':30,
               'HIDDEN_LAYERS':1,
               'HIDDEN_LAYER_SIZE':90,
               'LEARNING_RATE_COEFF':1.10,
               'TEST_SIZE':100}

task = Task.init(project_name='examples', task_name=str(hyperparams))

task.set_parameters_as_dict(hyperparams)

print(task.get_parameters())

# Imports to bring in libraries we need and sometimes give them shorthand aliases 
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tensorflow as tf
import datetime
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler

# load data
# This uses a method, e.g. code, provided with the data set
# It automatically loads both training and validation <image, label> tuples
(training_images, training_labels), (validation_images, validation_labels) = mnist.load_data()

# This limits the amount of data we use from each of the training and test sets to 
# the amount requested by the parameters provided for you to edit.

training_images = training_images[0:int(task.get_parameters()['General/TRAIN_SIZE'])]
training_labels = training_labels[0:int(task.get_parameters()['General/TRAIN_SIZE'])]
validation_images = validation_images[0:int(task.get_parameters()['General/TEST_SIZE'])]
validation_labels = validation_labels[0:int(task.get_parameters()['General/TEST_SIZE'])]

# Normalize inputs from 0-255 to 0.0-1.0
# In other words divide every pixel by 255 so the range is from 0-1 instead of 0-255
# This is because the neural netowrk expects inputs to be in the range from 0-1
training_images = training_images / 255
validation_images = validation_images / 255

# Convert numerical labels to "One-Hot Encoding"
# This means we encode each category's values as a separate boolean output variable
# So,instead of having one output encoded as 0:0/255, 1:1/255, 2:2/255, etc, 
# each category gets a different output, so for example, consider 0, 1, and 2:
# 0: {1, 0, 0}
# 1: {0, 1, 0}
# 2: {0, 0, 1} 
# This is important to do because There is no numerical relationship between the 
# categories. In other word, a 1.5 wouldn't mean half "1" and half "2". It would
# just be hard to interpret and not good for training. 
number_of_classes = 10
training_labels = np_utils.to_categorical(training_labels, number_of_classes)
validation_labels = np_utils.to_categorical(validation_labels, number_of_classes)

# Define the architecture of the neural network model

# Define the model sequentially, layer by layer
model = Sequential()

# Add input layer, flattening out the 28x28 2D image into a 1D vector on its way into the network
model.add(Flatten())

# Add hidden layer(s)
# A dense layer is trhe basic fully connected later. 
# "relu" is a "rectified linear unit", which is a long way of saying it makes anything negative into a 0
# There isn't such a thing as a negative color, so inhibition (negative weight) isn't likely to be helpful.
for dense_layer in range (int(task.get_parameters()['General/HIDDEN_LAYERS'])):
  model.add(Dense(int(task.get_parameters()['General/HIDDEN_LAYER_SIZE']), activation='relu'))

# Add output layer
# The output layer is "softmax", which basically means that it takes the 
# various outputs and makes them sum up to 1.0, so that they can be interpreted sort of like probabilities
# rather than us having to look at the whole set to interpret how much stronger or weaker one output is than others
model.add(Dense(number_of_classes, activation='softmax'))

# Compile model
# This is, in some ways, like compiling a program. It takes the model above from a definiton to a usable instance

# Categorical cross-entropy is a way of measuring error in situations where, as is the case with one-hot encoding, outputs are boolean and indicate membership in a single category
# The "adam" optimizer is a form of gradient descent, i.e. a way for the network to assign blame for error and adjust weights by back propogation
# Accuracy is a metric that measures the correctness of the categorization
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Create a log directory where the neural network can store data for later visualization by TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Set up "callback" so that model.fit feeds data into the log diretory for TensorBoard as it trains
# Note that this initializes the callback with the log directory created above and will log with each epoch, i.e. histogram_freq = 1
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit, a.k.a. train, the model

def custom_learning_rate(epoch, lrate):
	return float(task.get_parameters()['General/LEARNING_RATE_COEFF'])*lrate
 
lrs_callback = LearningRateScheduler(custom_learning_rate)
model.fit(training_images, training_labels, 
          validation_data=(validation_images, validation_labels), 
          epochs=int(task.get_parameters()['General/EPOCHS']), shuffle=True, 
          batch_size=int(task.get_parameters()['General/BATCH_SIZE']), 
          callbacks=[tensorboard_callback,lrs_callback])

metrics = model.evaluate(validation_images, validation_labels, verbose=0)

task.connect({'test_loss':metrics[0], 'test_accuracy':metrics[1]}, name='test_data')