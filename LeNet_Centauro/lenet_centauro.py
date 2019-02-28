# import the necessary packages
import sys
sys.path.append("/home/centauro/keras_ws/keras_repo/LeNet_Centauro/pyimagesearch/cnn/networks/")

from pyimagesearch.cnn.networks import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import cv2
import os
from PIL import Image
from scipy import misc
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# TODO AUGMENTED DATASET,
# follow this http://machinelearningmastery.com/image-augmentation-deep-learning-keras/

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())


print("[INFO] Loading dataset...")
#dir_name = "data/resized_180x180/"
#dir_name = "data"

imList = []
labelList = []

dir_name = './data/simple'
width = 0
height = 0 

for folder in os.listdir(dir_name):
    if os.path.isdir(os.path.join(dir_name, folder)):

        for imageName in sorted(os.listdir(os.path.join(dir_name, folder))):
            #img = load_img(dir_name + os.sep + imageName)  # this is a PIL image
            img = misc.imread(dir_name + os.sep + folder + os.sep + imageName) # this is a numpy    
            
            	#resize
            img=cv2.resize(img,(180,180))    
            
        #    print(img.shape)
        #    print(type(img))
            
            width, height, channels = img.shape
            #width, height = img.size
            	#new_size = width/2, width/2
            	#new_size = int(width/1), int(width/1)	
            	#img.thumbnail(new_size, Image.ANTIALIAS)
        
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB    
        
        #    print(img_array.shape)
        #    print(type(img_array))
        #    sys.exit(0)        	
                 
            #img_array = img_to_array(RGB_img)  # this is a Numpy array
            
            
            imList.append(img_array)
            # get the labels from the image name
            # the label is the folder name
            labelList.append(folder)
            	
            data = np.array(imList)
            labels_categorical = np.array(labelList)

#for image in imList:
#    cv2.imshow("OPI", image)
#    cv2.waitKey(0)
#

#print(data.shape)
#sys.exit(0)

# one hot encoder, encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels_categorical)
encoded_Y = encoder.transform(labels_categorical)
# convert integers to dummy variables (i.e. one hot encoded)
#print(labels_categorical)
labels = np_utils.to_categorical(encoded_Y)
#print(labels)

#sys.exit(0)

trainData, testData, trainLabels, testLabels = train_test_split(data / 255.0, labels, test_size=0.2, random_state=0)

#print trainData
#print (trainLabels)
#print (testData)
#print (testLabels)

print ("Train samples are: " + str(trainData.shape[0]))
print ("Test samples are: " + str(testData.shape[0]))
 
#sys.exit(0)

## TRAINING CONSTANT
BATCH_SIZE  = trainData.shape[0]/2 # train samples in the batch
EPOCH 	 = 1000
IMG_WIDTH   = width #180
IMG_HEIGHT  = height #180
IMG_DEPTH   = 3
NUM_CLASSES = trainLabels.shape[1]

print ("Classes are: " + str(NUM_CLASSES))
print ("BATCH: " + str(BATCH_SIZE))
print ("WIDTH: " + str(IMG_WIDTH))
print ("HEIGHT: " + str(IMG_HEIGHT))

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
#opt = 'adam'

model = LeNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH, classes=NUM_CLASSES,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
model.summary()
 
# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
#	print("[INFO] training...")
#	model.fit(trainData, trainLabels, batch_size=BATCH_SIZE, epochs=EPOCH,
#		verbose=1)
# 
#	# show the accuracy on the testing set
#	print("[INFO] evaluating...")
#	(loss, accuracy) = model.evaluate(testData, testLabels,
#		batch_size=BATCH_SIZE, verbose=1)
#	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    #-------------------------	
    # VALIDATION METHOD
    #-------------------------
    print("[INFO] training...")
    # First shuffle the train set, perche' altrimenti nel validation ho sempre e solo le ultime classi
    trainData, trainLabels = shuffle(trainData, trainLabels, random_state=0)
    
    # checkpoint
    filepath= "./output/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
		patience=10, min_lr=0.001)
				  
    callbacks_list = [checkpoint, reduce_lr]
		
    model.fit(trainData, trainLabels, batch_size=BATCH_SIZE, epochs=EPOCH, 
		verbose=1, callbacks=callbacks_list, validation_split=0.2)	

    #--------------------------
    # end VALIDATION METHOD
    #--------------------------
 
    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=BATCH_SIZE, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
	
# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=BATCH_SIZE, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(5,)):
	# classify the digit
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	
	# resize the image: make it bigger to better see it
	#image = (testData[i][0] * 255).astype("uint8")
	image = (testData[i] * 255).astype("uint8")
	image = cv2.resize(image, (IMG_WIDTH*3, IMG_HEIGHT*3), interpolation=cv2.INTER_LINEAR)
 
	# show the image and prediction
	print("[INFO] Class Index -> Predicted: {}, Actual: {}".format(prediction[0],
		np.argmax(testLabels[i])))
		
	index_predicted = prediction[0]
	index_actual = np.argmax(testLabels[i])
	
	predicted_class = encoder.inverse_transform(index_predicted)
	actual_class 	= encoder.inverse_transform(index_actual)
	
	print("[INFO] On label -> Predicted: {}, Actual: {}".format(predicted_class,
		actual_class))
			
	#cv2.putText(image, str(prediction[0]), (5, 20),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.putText(image, str(predicted_class), (5, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	
	cv2.imshow("OPI", image)
	cv2.waitKey(0)
