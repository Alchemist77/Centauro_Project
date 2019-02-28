#import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dense
from keras.layers import Dense, Dropout, Flatten, Activation, Input

class LeNet:
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):
     
     
            model = Sequential()
            
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(width, height, depth)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(classes, activation='softmax', name='preds'))
            
            if weightsPath is not None:
                model.load_weights(weightsPath)
            
            return model
            
            ## commento: con layer 20, 50 (kernel 5) acc. 82%
#            model = Sequential()
#            
#            model.add(Conv2D(20, kernel_size=(5, 5),
#                             activation='relu',
#                             input_shape=(width, height, depth)))
#            
#            model.add(Conv2D(50, (5, 5), activation='relu'))
#            
#            model.add(MaxPooling2D(pool_size=(2, 2)))
#            #model.add(Dropout(0.25))
#            
#            model.add(Flatten())
#            model.add(Dense(200, activation='relu'))
#            #model.add(Dropout(0.5))
#            model.add(Dense(classes, activation='softmax', name='preds'))
#            model.add(Activation('softmax'))
#            
#            if weightsPath is not None:
#                model.load_weights(weightsPath)
#            
#            return model

            
           # model = Sequential()
#            model.add(Conv2D(32, kernel_size=(3, 3),
#                             activation='relu',
#                             input_shape=(width, height, depth)))
#            
#            model.add(Conv2D(32, (3, 3), activation='relu'))
#            model.add(MaxPooling2D(pool_size=(2, 2)))
#            #model.add(Dropout(0.25))
#            
#	       model.add(Conv2D(64, (3, 3), activation='relu'))
#	       model.add(Conv2D(64, (3, 3), activation='relu'))
#            model.add(MaxPooling2D(pool_size=(2, 2)))
#         
#            model.add(Flatten())
#            model.add(Dense(640, activation='relu'))
#            #model.add(Dropout(0.5))
#            model.add(Dense(classes, activation='softmax', name='preds'))
#            model.add(Activation('softmax'))
#            
#            if weightsPath is not None:
#                model.load_weights(weightsPath)
#            
#            return model
     
#            model = Sequential()
#            
#            model.add(Conv2D(32, kernel_size=(3, 3),
#                             activation='relu',
#                             input_shape=(width, height, depth)))
#            
#            model.add(Conv2D(64, (3, 3), activation='relu'))
#            
#            model.add(MaxPooling2D(pool_size=(2, 2)))
#            model.add(Dropout(0.25))
#            
#            model.add(Flatten())
#            model.add(Dense(128, activation='relu'))
#            model.add(Dropout(0.5))
#            model.add(Dense(classes, activation='softmax', name='preds'))
#            model.add(Activation('softmax'))
#            
#            if weightsPath is not None:
#                model.load_weights(weightsPath)
#            
#            return model
    
     
		# initialize the model
#		model = Sequential()
#
#		# first set of CONV => RELU => POOL
#		model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(width, height, depth))) # tf ordering
#		model.add(Activation("relu"))
#		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
#		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_first"))
#		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#		# second set of CONV => RELU => POOL
#		model.add(Convolution2D(50, 5, 5, border_mode="same", input_shape=(width, height, depth))) # tf ordering
#		model.add(Activation("relu"))
#		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
#		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_first"))
#		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#		# set of FC => RELU layers
#		model.add(Flatten())
#		model.add(Dense(500))
#		model.add(Activation("relu"))
#
#		# softmax classifier
#		model.add(Dense(classes))
#		model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
#		if weightsPath is not None:
#			model.load_weights(weightsPath)
#
#		# return the constructed network architecture
#		return model
