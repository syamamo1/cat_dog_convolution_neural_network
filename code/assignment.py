from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        # TODO: Initialize all hyperparameters
        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # TODO: Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, 16], stddev=.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([3, 3, 16, 20], stddev=.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([5, 5, 20, 40], stddev=.1))
        self.filter4 = tf.Variable(tf.random.truncated_normal([7, 7, 40, 80], stddev=.1))

        self.conv1_bias = tf.Variable(tf.random.truncated_normal([16], stddev=.1))
        self.conv2_bias = tf.Variable(tf.random.truncated_normal([20], stddev=.1))
        self.conv3_bias = tf.Variable(tf.random.truncated_normal([40], stddev=.1))
        self.conv4_bias = tf.Variable(tf.random.truncated_normal([80], stddev=.1))

        self.W1 = tf.Variable(tf.random.truncated_normal([80, 128], stddev=.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([128], stddev=.1))

        self.W2 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([64], stddev=.1))

        self.W3 = tf.Variable(tf.random.truncated_normal([64, 64], stddev=.1))
        self.b3 = tf.Variable(tf.random.truncated_normal([64], stddev=.1))

        self.W4 = tf.Variable(tf.random.truncated_normal([64, 2], stddev=.1))
        self.b4 = tf.Variable(tf.random.truncated_normal([2], stddev=.1))

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        # Convolution Layer 1
        filter1output = tf.nn.conv2d(inputs, self.filter1, strides=[1,1,1,1], padding='SAME')
        bias1output = tf.nn.bias_add(filter1output, self.conv1_bias)
        # Batch Normalization 1
        conv1mean, conv1var = tf.nn.moments(bias1output, [0,1,2])
        batch_norm1 = tf.nn.batch_normalization(bias1output, conv1mean, conv1var, offset=0, scale=1, variance_epsilon=1e-5)
        # ReLU 1
        relu1 = tf.nn.relu(batch_norm1)
        # Max Pooling 1
        conv1output = tf.nn.max_pool(relu1, 3, 3, padding='SAME')

        # Convolution Layer 2
        filter2output = tf.nn.conv2d(conv1output, self.filter2, strides=[1,1,1,1], padding='SAME')
        bias2output = tf.nn.bias_add(filter2output, self.conv2_bias)
        # Batch Normalization 2
        conv2mean, conv2var = tf.nn.moments(bias2output, [0,1,2])
        batch_norm2 = tf.nn.batch_normalization(bias2output, conv2mean, conv2var, offset=0, scale=1, variance_epsilon=1e-5)
        # ReLU 2
        relu2 = tf.nn.relu(batch_norm2)
        # Max Pooling 2
        conv2output = tf.nn.max_pool(relu2, 2, 2, padding='SAME')  

        # Convolution Layer 3
        filter3output = tf.nn.conv2d(conv2output, self.filter3, strides=[1,2,2,1], padding='SAME')
        bias3output = tf.nn.bias_add(filter3output, self.conv3_bias)
        # Batch Normalization 3
        conv3mean, conv3var = tf.nn.moments(bias3output, [0,1,2])
        batch_norm3 = tf.nn.batch_normalization(bias3output, conv3mean, conv3var, offset=0, scale=1, variance_epsilon=1e-5)
        # ReLU 3
        relu3 = tf.nn.relu(batch_norm3)
        # Max Pooling 3
        conv3output = tf.nn.max_pool(relu3, 2, 2, padding='SAME')

        # Convolution Layer 4
        if is_testing:
            filter4output = conv2d(conv3output, self.filter4, strides=[1,1,1,1], padding='SAME')
        else:
            filter4output = tf.nn.conv2d(conv3output, self.filter4, strides=[1,1,1,1], padding='SAME')
        bias4output = tf.nn.bias_add(filter4output, self.conv4_bias)
        # Batch Normalization 4
        conv4mean, conv4var = tf.nn.moments(bias4output, [0,1,2])
        batch_norm4 = tf.nn.batch_normalization(bias4output, conv4mean, conv4var, offset=0, scale=1, variance_epsilon=1e-5)
        # ReLU 4
        relu4 = tf.nn.relu(batch_norm4)
        # Max Pooling 4
        conv4output = tf.nn.max_pool(relu4, 1, 2, padding='SAME')        

        conv_output = tf.reshape(conv4output, [conv4output.shape[0], -1])

        # Dense Layer 1
        Wx1 = tf.matmul(conv_output, self.W1) + self.b1
        drop1 = tf.nn.dropout(Wx1, rate=.3)
        dense1output = tf.nn.relu(drop1)

        # Dense Layer 2
        Wx2 = tf.matmul(dense1output, self.W2) + self.b2
        drop2 = tf.nn.dropout(Wx2, rate=.3)
        dense2output = tf.nn.sigmoid(drop2)

        # Dense Layer 3
        Wx3 = tf.matmul(dense2output, self.W3) + self.b3
        drop3 = tf.nn.dropout(Wx3, rate=.3)
        dense3output = drop3

        # Dense Layer 4
        Wx4 = tf.matmul(dense3output, self.W4) + self.b4

        logits = tf.nn.softmax(Wx4)
        # logits = tf.nn.relu(Wx4)

        #print('logits', logits.shape)
        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        av_loss = tf.math.reduce_mean(loss)

        return av_loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    inds = tf.random.shuffle(range(len(train_labels)))
    shuffled_inputs = tf.gather(train_inputs, inds)
    shuffled_labels = tf.gather(train_labels, inds)

    for i in range(0, int(len(shuffled_labels)/model.batch_size)):
        images = shuffled_inputs[i*model.batch_size:(i+1)*model.batch_size,:,:] # ,np.newaxis?
        labels = shuffled_labels[i*model.batch_size:(i+1)*model.batch_size]
        
        losses = []
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = model.loss(predictions, labels)
            losses.append(loss)
            if i//model.batch_size % 4 == 0:
                train_acc = model.accuracy(model(images), labels)
                print("Accuracy, {} training steps: {}".format(i, train_acc))

        #print(loss.shape, 'lossss')
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return losses

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    # Loop through 10000 training images
    probs = model.call(test_inputs, is_testing = True)
    acc = model.accuracy(probs, test_labels)
    
    print('Test accuracy: {}'.format(acc)) 

    return acc


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''

    fname_train = 'C:\\Users\smy18\workspace2\dl\hw2-cnn-syamamo1\hw2\data2\data\\train'
    fname_test = 'C:\\Users\smy18\workspace2\dl\hw2-cnn-syamamo1\hw2\data2\data\\test'
    cat = 3
    dog = 5
    clean_inputs, clean_labels = get_data(fname_train, cat, dog)
    test_inputs, test_labels = get_data(fname_test, cat, dog)
    M = Model()
    num_epochs = 25

    losses = []
    for j in range(num_epochs):
        print('Epoch:', j, '---------------------------------------------')
        loss = train(M, clean_inputs, clean_labels)
        losses = losses + loss
    print('Finished -----------------------------------------')

    test(M, test_inputs, test_labels)
    visualize_loss(np.array(losses).flatten())
    
    return None


if __name__ == '__main__':
    main()
