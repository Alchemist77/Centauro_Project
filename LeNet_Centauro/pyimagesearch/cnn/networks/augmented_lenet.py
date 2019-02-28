# model from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dense
from keras.layers import Dense, Dropout, Flatten, Activation, Input

class AugmentedLeNet:
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(width, height, depth)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
	    # the model so far outputs 3D feature maps (height, width, features)
		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes, activation='softmax', name='preds'))
		
		if weightsPath is not None:
			model.load_weights(weightsPath)

			return model

