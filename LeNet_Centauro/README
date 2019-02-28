# README #

*Author: Alessandro Manzi*

Example taken from  http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

## Training ##
* Move into your *virtualenv* environment. On the centauro PC, it is ```workon ros_keras_tf```
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
