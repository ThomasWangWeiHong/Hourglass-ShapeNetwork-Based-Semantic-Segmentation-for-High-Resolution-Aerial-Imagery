# Hourglass-ShapeNetwork-Based-Semantic-Segmentation-for-High-Resolution-Aerial-Imagery
Python implementation of Convolutional Neural Network (CNN) discussed in paper

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the Hourglass - Shape Network (HSN) implemented in the paper 
'Hourglass-ShapeNetwork Based Semantic Segmentation for High Resolution Aerial Imagery' by Liu Y., Nguyen D.M., Deligiannis N., Ding W., 
Munteanu A. (2017)

The main differences between the implementations in the paper and the implementation in this repository is as follows:

- Adam optimizer is used instead of the Stochastic Gradient Descent, since it is an adaptive optimizer which might be more suited
for this type of semantic segmentation tasks
- Overlap inference is not used for the image semantic segmentation prediction, for ease of computation

Requirements:
- cv2
- glob
- json
- numpy
- keras (Tensorflow backend)
- rasterio
