
# Energy Efficiency in AI- Quantization of ML models
## Tudor Nichitov


## Overview
Quantization is a method used to reduce the computation overhead of applications in the Artificial Intelligence domain . 
The scope of this exercise to gain a better understanding of Quantization In AI by experimenting with a pre-trained object 
detection model and to test the validity and tradeoffs of quantization, specifically considering our current situation with the energy market and need for energy efficiency. 
This Jupyter notebook provides a walk-through for quantizing an AI model and testing its performance using the TensorFlow framework. 

## Features:
### 1.Load an image into a numpy array.
### 2.Load COCO dataset label map.
### 3.Run inference on a sample image.
### 4.Set up interpreters for quantized models.
### 5.Convert a model to TensorFlow Lite format.
### 6.Quantize the TensorFlow Lite model to 8-bit and 16 bit precision.
### 7.Analyze the results -sizes and inference timesof the quantized models by running inference on multiple images using the quantized model

## Dependencies:

#### 1.TensorFlow: Deep learning framework.
#### 2.NumPy: To handle numerical operations.
#### 3.PIL (Python Imaging Library): To load and manipulate images.
#### 4.Matplotlib: For visualizing image outputs.
#### 5.TFLite: TensorFlow Lite for mobile and embedded devices.
#### 6.pycocotools: For COCO dataset related utilities.
#### 7.glob and os: For file-related operations.  

## Prerequisites
### Model (suggested) - https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2 (TFLITE)
In the experiments we used an already pretrained model: MobilenetSSD V2. It is a Machine Learning model for Fast Object Detection specifically optimized for mobile devices. The model takes as input an image (converted to array) computes the bounding box and category of an object from an input image. 
The inference of this model is to be done either directly on the mobile device or on Edge devices. 
### Model2 (Matlab) https://www.mathworks.com/help/vision/ref/yolov3objectdetector.html

### Data set - https://cocodataset.org/#download

The experiments are done using the COCO 2017 Validation Dataset (Common Objects in Context),
 which is a large-scale object detection dataset, very popular in various ML applications.

## How to Use:

1. **Loading Images**:
   - Use the function `load_image_into_numpy_array(path)` to load an image from the specified path into a numpy array.
   
2. **Label Map**:
   - The COCO dataset label map is preloaded. This map matches label IDs to their respective object names.

3. **Inference on Sample Image**:
   - Load the sample image from the provided directory.
   - Pass the loaded image through the model to obtain predictions.
   - Visualize the results using matplotlib.

4. **Setting up Interpreters for Quantized Models**:
   - Initialize the TensorFlow Lite interpreter with the path of the quantized model.
   - Adjust tensor shapes if necessary.
   - Allocate tensors for the interpreter.

5. **Quantization**:
   - Convert the TensorFlow model to TensorFlow Lite format.
   - Further quantize the TensorFlow Lite model to 8-bit precision for efficient deployment.
   
6. **Inference on Multiple Images**:
   - Loop through multiple images in the specified directory.
   - Run inference using the quantized model on each image.
   - Compute the mean average precision (MAP) over the processed images.
   
7. **Save Models**:
   - Save the converted and quantized models to disk for future use or deployment.

8. **Evaluation**:
   - Additional code snippets are provided for further evaluation and testing of quantized models.


#Useful Resources Links: 
https://www.tensorflow.org/lite/performance/post_training_quantization
https://www.tensorflow.org/lite/models/convert
https://voxel51.com/docs/fiftyone/user\_guide/index.html
https://colab.research.google.com/github/voxel51/fiftyone/blob/v0.9.1/docs/source/tutorials/
https://tfhub.dev/iree/lite-model/ssd\_mobilenet\_v2\_100/fp32/default/1
https://medium.com/axinc-ai/mobilenetssd-a-machine-learning-model-for-fast-object-detection-37352ce6da7d
https://voxel51.com/docs/fiftyone/user\_guide/evaluation.html\#supported-types