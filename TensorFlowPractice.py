#TensorFlow Neural Network Practice
## This incorporates the following:
## Creating a neural network that classifies images
## Training said Network using training data
## And evaluating its accuracy on a validation or test data set

import tensorflow as tf

#Preparing MNIST data and converting from integers to floating point numbers
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test /255.0

#Create a stacked layer model using sequential
#4 total layers
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)])

#For each example, (image) the model will return a score for the likelyhood it is part of each class
#This model function will return "logits" or "log odds" which are explained in
#https://developers.google.com/machine-learning/glossary#logits
predictions = model(x_train[:1]).numpy()
print(f'here are the raw logit values for the predictions \n {predictions}')

#These raw values need to get sent through the "softmax" function in order to be
#Converted to probabilities. Google Documentation will help explain this too.
#nn = neural network
tf.nn.softmax(predictions).numpy()

#Now we write a section to take the losses from each example
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Losses represent the negative log probability of the true (or correct) class
#classification. When this is 0, the model is 100% sure on the class.

loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

#model.fir will adjust the model and mimimize the loss
#Epoch is basically when the entire data set is passed forward and backward
#through the neural network ONE TIME.
model.fit(x_train,y_train,epochs=5)

#Model.evaluate checks performance on validation or test datasets
model.evaluate(x_test, y_test, verbose=2)
#Returned from this will be statistics on the accuracy and amount of losses
probability_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])
