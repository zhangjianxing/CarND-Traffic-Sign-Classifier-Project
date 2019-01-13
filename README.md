# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[all_type_of_signs]: ./output/all_type_of_signs.jpg "Visualization"
[distribution_bar]: ./output/distribution_bar.jpg "Visualization"
[preprocessing_example]: ./output/preprocessing_example.jpg "Visualization"

[new_imgs_prediction]: ./output/new_imgs_prediction.jpg "Visualization"
[new_imgs]: ./output/new_imgs.jpg "Visualization"
[error_rate]: output/error_rate.jpg

[hidden_layer]: ./output/hidden_layer.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training img is 34799 (augmented to 63568)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

Here is all label and its sign
![alt text][all_type_of_signs]


Since there are a lot labels with low number of training images. 
I augment the training set by apply rotation, and shift on the image as “additional training set”. 
And would perform this operation until the class label has at least 1000 images.

It is a bar chart showing how the data distributed across different labels

![alt text][distribution_bar]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data. 

Firstly, I decided resize img to (32, 32), 
because reduce the size to input image makes neural network runs more efficient
 
Secondly, I normalize the image. e.g. (img / np.max(img)).
Because nerual network works better when input features are in range from 0 to 1

I used cvtColor, equalizeHist, CLAHE technique to pre-process images, so that it will reduce noise.
And normalize the image to [-0.5, 0.5] for each pixel, thus easier for the neural network to train.
The output is a gray image
```python
import numpy as np
import cv2

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def pre_process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = CLAHE.apply(img)
    img = cv2.resize(img, (32, 32))
    img = img/np.max(img) - 0.5
    return img
```

Here is an example of a traffic sign image before and after pre-processing.


![alt text][preprocessing_example]

#### 2. Final model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 GRAY image   							|
| Resize         		| output 32x32x1       							|  
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 				    |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 12x12x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x32				    |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 4x4x64  	|
| RELU					|												|
| Drop out      		| prob = 0.5                    				|
| Fully connected		| inputs 1024, outputs 512        				|
| RELU					|												|
| Drop out      		| prob = 0.5                    				|
| Fully connected		| inputs 512, outputs 256         				|
| RELU					|												|
| Drop out      		| prob = 0.5                    				|
| Fully connected		| inputs 256, outputs 128         				|
| RELU					|												|
| Drop out      		| prob = 0.5                    				|
| Fully connected		| inputs 128, outputs 64         				|
| RELU					|												|
| Fully connected		| inputs 64, outputs 43         				|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer because in practice, this optimizer works very well.

I choose BATCH_SIZE = 256, and train for learning_rate in [0.001, 0.0001], 
each learning_rate train 20 epochs.
Because larger learning rate trains faster and smaller learning rate 
help the model to be more accurate


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model performs very well, results were:
* Training set Accuracy = 1.000
* Validation set Accuracy = 0.957
* Test set Accuracy = 0.938

the learning curve (epochs v.s. error_rate):
![alt_text][error_rate]

If an iterative approach was chosen:
* The first architecture choose is LeNet, because it is easy to write. 
* But LeNet uses too few arguments, and can not reach high accuracy
* the architecture has more cov layers and more fully connected layer. 
And add drop to reduce over-fitting

* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? 
Typical adjustments could include choosing a different model architecture, 
adding or taking away layers (pooling, dropout, convolution, etc), 
using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![alt_text][new_imgs]

The 1st, 2nd, 5th image might be difficult to classify because they have noisy background. 
And their shape is not close to square. So that we need to resize it and might cause our neural network to predict.

I choose these traffic-signs because they have shape (tri-angle or circle), and they have different color.

The first one might hard to detect because the traffic sign is at right of the image and the whole image is not square.

The second one might hard to detect because that traffic looks like 'while animal crossing' traffic sign, and has watermark, and has additional traffic-sign on the image

The third one might hard to detect because the background is blue, and we might mis-classify it to a blue traffic-sign.
 
The fourth one might hard to detect because there is Picture watermark over the traffic sign.

The fifth one might hard to detect because it has noisy background.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

The second image is kind of hard, because that is too blur/

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield           		| Yield        									| 
| Children crossing     | Speed limit (50km/h)   						|
| Road work				| Road work								    	|
| Keep right	      	| Keep right					 		    	|
| No entry			    | No entry      						    	|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
This compares favorably to the accuracy on the test set.

#### 3. How certain the model is when predicting on each of the five new images 

![alt text][new_imgs_prediction]


Show how certain the model is when predicting, 
by looking a the softmax probabilities for each prediction.

Only the second one, which is wrong predicted, has confidence 70+%. 
All the other image are above 99% confident

1.jpeg: actual label = Yield(13)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|99.7% 		| Yield(13) 		|
|0.1% 		| Priority road(12) 		|
|0.1% 		| Stop(14) 		|
|0.0% 		| Speed limit (60km/h)(3) 		|
|0.0% 		| Ahead only(35) 		|

2.jpeg: actual label = Children crossing(28)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|73.7% 		| Speed limit (50km/h)(2) 		|
|12.1% 		| Beware of ice/snow(30) 		|
|5.1% 		| Traffic signals(26) 		|
|4.8% 		| Right-of-way at the next intersection(11) 		|
|1.0% 		| Speed limit (20km/h)(0) 		|

3.jpeg: actual label = Road work(25)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|100.0% 		| Bicycles crossing(29) 		|
|0.0% 		| Bumpy road(22) 		|
|0.0% 		| Road work(25) 		|
|0.0% 		| Children crossing(28) 		|
|0.0% 		| Slippery road(23) 		|

4.jpeg: actual label = Keep right(38)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|100.0% 		| Keep right(38) 		|
|0.0% 		| General caution(18) 		|
|0.0% 		| Speed limit (20km/h)(0) 		|
|0.0% 		| Turn left ahead(34) 		|
|0.0% 		| Speed limit (30km/h)(1) 		|

5.jpeg: actual label = No entry(17)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|100.0% 		| No entry(17) 		|
|0.0% 		| Stop(14) 		|
|0.0% 		| No passing(9) 		|
|0.0% 		| End of all speed and passing limits(32) 		|
|0.0% 		| End of no passing(41) 		|


### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. What characteristics did the neural network use to make classifications?
The main character is something link a circle or triangle. Which is the shape of the traffic sign

![alt_text][hidden_layer]

