import pickle
import numpy as np
import tensorflow as tf
import os

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = np.array(unpickled_file[b'labels'])
	# TODO: Do the rest of preprocessing! 
	inds = np.logical_or(labels == first_class, labels == second_class)

	found_inputs = inputs[inds]/255
	reshaped = tf.transpose(tf.reshape(found_inputs, (-1, 3, 32, 32)), perm = [0,2,3,1]) # shape (num_examples, 32, 32, 3)
	clean_inputs = np.array(reshaped, dtype=np.float32)

	found_labels = labels[inds]
	binary_labels = np.where(found_labels==first_class, 0, 1)
	clean_labels = tf.one_hot(binary_labels, 2)

	print(len(clean_inputs), len(clean_labels))
	#print(len(clean_inputs), len(clean_inputs[0]), len(clean_inputs[0][0]), len(clean_inputs[0][0][0]),  type(clean_labels))

	return clean_inputs, clean_labels

# fname = 'C:\\Users\smy18\workspace2\dl\hw2-cnn-syamamo1\hw2\data2\data\\train'
# cat = 3
# dog = 5
# d = get_data(fname, cat, dog)
