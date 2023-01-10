
# AIML Project - Energy Efficiency in AI- Quantization of ML models
## GROUP 6 - Nichitov


## Overview
Quantization is a method used to reduce the computation overhead of applications in the Artificial Intelligence domain . 
The scope of this exercise to gain a better understanding of Quantization In AI by experimenting with a pre-trained object 
detection model and to test the validity and tradeoffs of quantization, specifically considering our current situation with the energy market and need for energy efficiency. 
The experiments are done  in their entirety using Google Colab as the environment and Python as the programming language. 


## Prerequisites
### Model (suggested) - https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
In the experiments we used an already pretrained model: MobilenetSSD V2. It is a Machine Learning model for Fast Object Detection specifically optimized for mobile devices. The model takes as input an image (converted to array) computes the bounding box and category of an object from an input image. 
The inference of this model is to be done either directly on the mobile device or on Edge devices. 
### Model2 (Matlab) https://www.mathworks.com/help/vision/ref/yolov3objectdetector.html

### Data set - https://cocodataset.org/#download

The experiments are done using the COCO 2017 Validation Dataset (Common Objects in Context),
 which is a large-scale object detection dataset, very popular in various ML applications.

The overall scope is measuring the accuracy and computational cost of a quantized pre-trained model with three configurations:
- float32
- float16
- int8

## Evaluation metric: Mean Average Precision
The mAP score is calculated by taking the mean AP over all classes,  using the detection precision scores information.

#Useful Resources Links: 
https://www.tensorflow.org/lite/performance/post_training_quantization
https://www.tensorflow.org/lite/models/convert
https://voxel51.com/docs/fiftyone/user\_guide/index.html
https://colab.research.google.com/github/voxel51/fiftyone/blob/v0.9.1/docs/source/tutorials/
https://tfhub.dev/iree/lite-model/ssd\_mobilenet\_v2\_100/fp32/default/1
https://medium.com/axinc-ai/mobilenetssd-a-machine-learning-model-for-fast-object-detection-37352ce6da7d
https://voxel51.com/docs/fiftyone/user\_guide/evaluation.html\#supported-types