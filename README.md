
# AIML Project - Energy Efficiency in AI- Quantization of ML models
## GROUP 6
### Nichitov

#Resource Link https://www.tensorflow.org/lite/performance/post_training_quantization


## Overview
Measuring the accuracy and computational cost of a quantized pre-trained model with three configurations:
- float32
- float16
- int8

## Prerequisites
### Model (suggested) - https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
In the experiments we used an already pretrained model: MobilenetSSD V2. It is a Machine Learning model for Fast Object Detection specifically optimized for mobile devices. The model takes as input an image (converted to array) computes the bounding box and category of an object from an input image. 
The inference of this model is to be done either directly on the mobile device or on Edge devices. 
### Model2 (Matlab) https://www.mathworks.com/help/vision/ref/yolov3objectdetector.html
### Data set - https://cocodataset.org/#download

The experiments are done using the COCO 2017 Validation Dataset (Common Objects in Context),
 which is a large-scale object detection dataset, very popular in various ML applications.