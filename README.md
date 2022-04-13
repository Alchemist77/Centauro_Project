# README #

*Author: Alessandro Manzi & Jaeseok Kim*

Example taken from  http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

## Download dataset ##
```
wget https://www.dropbox.com/s/adph9zgh9ilpg1b/augmented_2classes_carton_plastic.zip?dl=0
```
* change file name
```
mv augmented_2classes_carton_plastic.zip?dl=0 augmented_2classes_carton_plastic.zip
```

*Then, extract the zip file into *data_augmentation*
```
unzip augmented_2classes_carton_plastic.zip
```

## Training ##
* PC should be installed Keras with Tensorflow!
* Move into *LeNet Centauro* folder
* To train and save: 
```
python augmented_lenet_centauro.py --save-model 1 --weights output/lenet_centauro_weights.hdf5 
```
* To load a model:
```
python augmented_lenet_centauro.py --load-model 1 --weights output/lenet_centauro_weights.hdf5
```

---

### Some Utils used in the past to manage images ###
Now it is somewhat deprecated, because you can use the ImageGenerators from Keras

* to simply rename images in progressive number:
```
    convert '*.png' resized%02d.png
```
* to resize all the images:
```
    for file in *.png; do convert $file -resize 180x180! $file; done
    
```

# Reference #
Kim, Jaeseok, Olivia Nocentini, Marco Scafuro, Raffaele Limosani, Alessandro Manzi, Paolo Dario, Filippo Cavallo "An Innovative Automated Robotic System based on Deep Learning approach for Recycling Objects." In: 16th International Conference on Informatics in Control, Automation and Robotics (ICINCO 2019).

