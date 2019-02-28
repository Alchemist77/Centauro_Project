# Link https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import sys

datagen = ImageDataGenerator(
     #   rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

dir_name = './data/2classes_carton_plastic'
print dir_name
for folder in os.listdir(dir_name):
    print folder 
    #if folder != 'nothing':
        #continue
    print os.path.join(dir_name, folder)
    if os.path.isdir(os.path.join(dir_name, folder)):     
        print "sono qui"
        for imageName in sorted(os.listdir(os.path.join(dir_name, folder))):
            print imageName
            img = load_img(dir_name + os.sep + folder + os.sep + imageName)  # this is a PIL image
	    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                 save_to_dir='./augmented_2classes_carton_plastic/'+folder, save_prefix=folder, save_format='jpeg'):
                   i += 1
                   if i > 30:
                        break  # otherwise the generator would loop indefinitely

