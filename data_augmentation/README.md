# README #

*Author: Alessandro Manzi*

## Augment your dataset ##

* First, create your data collecting images. You can record a ROS bag and then easily extract frames automatically (http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data).

* Create a folder for each class and put in the relative images

* run ```python data_augmentation.py``` to apply augmentation (edit the dataset folder and augmentation options on the code)
* it will save the aumented dataset into the *augmented* folder 


