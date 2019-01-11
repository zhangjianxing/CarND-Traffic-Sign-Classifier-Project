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
[new1]: ./new_images/1.jpeg "new image"
[new2]: ./new_images/2.jpeg "new image"
[new3]: ./new_images/3.jpeg "new image"
[new4]: ./new_images/4.jpeg "new image"
[new5]: ./new_images/5.jpeg "new image"

[hidden_layer]: ./output/hidden_layer.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

Here is all label and its sign
![alt text][all_type_of_signs]

It is a bar chart showing how the data distributed across diffrent labels

![alt text][distribution_bar]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data. 

Firstly, I decided resize img to (32, 32), 
because reduce the size to input image makes neural network runs more efficient
 
Secondly, I normalize the image. e.g. (img / np.max(img)).
Because nerual network works better when input features are in range from 0 to 1

Here is an example of a traffic sign image before and after pre-processing.

![alt text][preprocessing_example]

#### 2. Final model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
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

My final model results were:
* Training set Accuracy = 1.000
* Validation set Accuracy = 0.957
* Test set Accuracy = 0.938

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

![alt text][new1] ![alt text][new2] ![alt text][new3] 
![alt text][new4] ![alt text][new5]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

The second image is kind of hard, because that is too blur/

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield           		| Yield        									| 
| Children crossing     | Speed limit (30km/h)   						|
| Road work				| Road work								    	|
| Keep right	      	| Keep right					 		    	|
| No entry			    | No entry      						    	|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
This compares favorably to the accuracy on the test set.

#### 3. How certain the model is when predicting on each of the five new images 

Show how certian the model is when predicting, 
by looking at the softmax probabilities for each prediction.

Only the second one, which is wrong predicted, has confidence 70%. 
All the other image are above 99% confident

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99                  | Yield           	                            | 
| 0.70                  | Speed limit (30km/h)                          |
| 1.00                  | Road work			                            |
| 0.99                  | Keep right	                                |
| 1.00                  | No entry			                            |


For the second image ... 

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. What characteristics did the neural network use to make classifications?
The main character is something link a circle or triangle. Which is the shape of the traffic sign

![alt_text][hidden_layer]

