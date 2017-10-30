# Traffic Sign Recognition

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Sign_Classes.png "Sign Classes"
[image2]: ./examples/Train_Examples.png "Training Bar Chart"
[image3]: ./examples/Test_Examples.png "Test Bar Chart"
[image4]: ./examples/Validation_Examples.png "Validation Bar Char"
[image5]: ./examples/Original_Img.png "Original Image"
[image6]: ./examples/Jittered_Dataset.png "Jittered Dataset"
[image7]: ./examples/LeNet-5.png "LeNet. Source: Yann Lecun."
[image8]: ./Test_Images/1x.png "Traffic Sign 1"
[image9]: ./Test_Images/2x.png "Traffic Sign 2"
[image10]: ./Test_Images/3x.png "Traffic Sign 3"
[image11]: ./Test_Images/4x.png "Traffic Sign 4"
[image12]: ./Test_Images/5x.png "Traffic Sign 5"
[image13]: ./examples/Prediction_Result1.png "Prediction Result 1"
[image17]: ./examples/GoogLeNet_Inception.png "Inception"
[image18]: ./Test_Images/6x.png "Traffic Sign 6"
[image19]: ./Test_Images/7x.png "Traffic Sign 7"
[image20]: ./Test_Images/8x.png "Traffic Sign 8"
[image21]: ./Test_Images/9x.png "Traffic Sign 9"
[image22]: ./Test_Images/10x.png "Traffic Sign 10"
[image23]: ./examples/Prediction_Result2.png "Prediction Result 2"
[image24]: ./examples/Softmax_1.png "Softmax Result 1"
[image25]: ./examples/Softmax_2.png "Softmax Result 1"
[image26]: ./examples/Softmax_3.png "Softmax Result 1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/vikasmalik22/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

I used AWS EC2 GPUs g2.2xlarge instances to run this project because it was way faster than to train it on GPU on my PC. I also tried it on my PC whcih has Nvidia GEForce 960M but the process takes too much time and it's difficult to tune the model again and again since the running time on PC was anywhere 5-6 horus. And running it on AWS was 1-2 hours. 

## Start the Project
1. Download the dataset. This is a pickled dataset in which we've already resized the images to 32x32. [Available here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
2. Clone the project and start the notebook.
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
3. Launch the Jupyter notebook: jupyter notebook Traffic_Sign_Classifier.ipynb
4. Follow the instructions in the notebook

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 
34799
* The size of the validation set is ? 
4410
* The size of test set is ?
12630
* The shape of a traffic sign image is ?
(32, 32, 3)
* The number of unique classes/labels in the data set is ?
43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
The image below shows all the 43 different Traffic Signs extracted from the dataset with their correct labels. 

![alt text][image1]

The three bar charts below shows the dataset distribution between Test, Training and Validation examples.

![alt text][image2]

![alt text][image3]

![alt text][image4]

The above plots shows us the amount of different datasets we have. And we use this dataset only to train and validate our Model the results are very bad. Since, this data is not enough to properly train our CNN model we need to generate more data using preprocessing the existing data.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
From the above plots it is clear that we have some data sets which are present more than some of the others. For example, in a traffic sign identification task, there may be more stop signs than speed limit signs. Therefore, in these cases, we need to make sure that the trained model is not biased towards the class that has more data. As an example, consider a data set where there are 5 speed limit signs and 20 stop signs. If the model predicts all signs to be stop signs, its accuracy is 80%. Further, f1-score of such a model is 0.88. Therefore, the model has high tendency to be biased toward the 'stop' sign class. In such cases, additional data needs to be generated to make the size of data sets is similar.

One way to collect more data is to take the picture of the same sign from different angles. This can be done easily in openCV by applying affine transformations, such as rotations, translations and shearing. Affine transformations are transformations where the parallel lines before transformation remain parallel after transformation.

As a first step, I decided to convert the existing images which are available in 32x32 pixel format to 26x26 because the side pixels are extra and do not contribute to much information about the sign. 

Images do not have proper contrast, sharpness and differnt viewing angles representation, so I applied operations like gaussian blurness, histogram equalization, rotation, transformation and affine transformation.

To create the PreProcessed/Jittered Dataset, I used the function Create_Jittered_Dataset.

Color of the images were not changed to grayscale because the colors can be the main distinguishing factor between some signs. When I tried to run/train the CNN with grayscale images my accuracy was never going beyond 0.8 and was giving quite bad results. This is opposite to theory described by Pierre Sermanet and Yann LeCun in their paper Traffic Sign Recognition with Multi-Scale Convolutional Networks from [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

Here is an example of a traffic sign image before and after Jittered Dataset.

![alt text][image5]

![alt text][image6]

After, preprocessing the training and validation data I combined them and then split them in 0.8 and 0.2. 

Following is the distribution obtained after preprocessing.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 26x26x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 26x26x3		|
| RELU					|												|
| Convolution 5x5		| 1x1 stride, same padding, outputs 26x26x64	|
| RELU					|												|
| Inception Module		| Output 26x26x256								|
| Max pooling	      	| 2x2 stride,  outputs 13x13x256				|
| Inception Module		| Output 13x13x512								|
| Max pooling	      	| 3x3 stride,  outputs 6x6x512					|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 6x6x256		|
| Flatten				| output 9216									|
| Fully connected		| output 512        							|
| Dropout				| 0.5											|
| Fully connected		| output 43		       							|

![alt text][image17]
[Source](https://arxiv.org/pdf/1409.4842.pdf)
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following values
EPOCHS = 15
BATCH_SIZE = 128
Optimizer = AdamOptimizer

Hyperparameters 
mu = 0.0 #mean
sigma = 0.1 #standard deviation
base_rate = 0.0005 #Base learning rate
dropout = 0.5 #dropout rate

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

The approach was mainly trial and error based to get a final solution and achieving accuracy above 0.93 atleast. 

###### What was the first architecture that was tried and why was it chosen?
I first tried using the [LeNet-5. Source: Yann Lecun.](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). Since, it was taught in the programme and had good resuts with the MNIST dataset. 

![alt text][image7]

###### What were some problems with the initial architecture?
With the LeNet-5 Model, I was not able to achieve the accuracy above 0.8 even after fine tuning many parameters and hyperparameters. 

##### How was the architecture adjusted and why was it adjusted?
After not able to achieve the desired results I completely changed my model to GoogLeNet which uses inception module implementation.

##### Which parameters were tuned? How were they adjusted and why?
Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned. For Epoch, I started with the bigger number but since the model accuracy didn't improve after certain epochs I decided to reduce it. The batch size was not changed much I just changed it within the range of 100 - 128. The learning rate I chose initally was .001 which is the standard starting rate generally used, but as I was not getting better accuracy I tuned it to 0.0005 and it helped me in improving the accuracy of the data. The dropout probability I left mainly unchanged and kept it as standard value of 0.5. Dropout lets the Network to never rely on any given activation to be present because they might be terminated at any given time. So it is foreced to learn a redundant representation of everything to make sure that at least some of the information remains. It makes the network more robust & prevents overfitting.
The most important factor was generating and preprocessing or creating jittered dataset randomly which helped in achieving much better results.

##### What are some of the important design choices and why were they chosen? 
There were few important design choices and they were chosen based on the trial and error based approach and also what many blogs and reseacrch papers recommended.
1. Fine Tuning and generating the extra dataset from the existing one. So that the model can train and learn well.
2. The CNN architecture itself. In this case GoogLeNet with inception features.
Maxpooling -> the benefit of the max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining values.
3. Playing around with the parameters and hyperparameters. 

##### If a well known architecture was chosen:
###### What architecture was chosen?
I choose the GoogLeNet architecture. 

###### Why did you believe it would be relevant to the traffic sign application?
Since, GoogLeNet model has proven results based on the research I did on the internet after reading many blogs and papers. 
Also, the inception implementation gives it an edge against other architecture models. The main idea of the Inception architecture is to consider
how an optimal local sparse structure of a convolutional vision network can be approximated and covered by readily available dense components. I implemented the inception module with dimensionality reduction architecture because this is computationaly less expensive than naive inception module. 

So I decided to choose this and was kind of confident it will give better results in terms of accuracy for Traffic Sign Classification. 

###### How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    
    My final model results are:
    
    * training set accuracy of ? 
    99.40
    * validation set accuracy of ?
    99.10
    * test set accuracy of ?
    94.6

### Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web and tested against my CNN.

![alt text][image8] ![alt text][image9] ![alt text][image10]  ![alt text][image11] ![alt text][image12] ![alt text][image18] ![alt text][image19] ![alt text][image20] ![alt text][image21] ![alt text][image22] 
_ _ _

The speed signs like 30 and 60 Km/h might be difficult to predict since all the signs have an outer red ring and if the numbers between the white area are not properly processed might result in predicting wrong result. 
![alt text][image9] ![alt text][image11]

_ _ _

Same looks with the sign of No vehicles which is very similar to speed signs except the inner white circular region doesn't contain any number.
![alt text][image20]
_ _ _

Bumpy road sign contains an outer red triangle and an inner pattern which is similar to other signals like biclycle crossing or wild animals crossing. Since if this inner pattern is not very clear in the images it might be difficult to predict accurately. 
![alt text][image19]
_ _ _

Same is the case with Turn left ahead sign and Go straight or left. These signs have a blue circular region and contains white colored arrows inside. This might make them difficult to distinguish between their characterisitcs. 
![alt text][image18] ![alt text][image22]

### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

![alt text][image13] ![alt text][image23]

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This shows the model is able to predict the real world data accurately. While the Test Accuracy was 0.946 but the model was able to guess all the 10 traffic signs correctly. This shows that model works well to guess the real world images. 

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.
Following are the softmax probabilities for each sign prediction:

![alt text][image24] ![alt text][image25] ![alt text][image26]

For all the traffic sign images choosen somehow the model had the accuracy of 100% or the probabilty of 1. 

From the above results of the 10 images it looks the system is quite reliable on prediciting the real world data.

As discussed before the bumpy road sign probability is very close to the sign bicycles crossing. Other traffic signs didn't have probabilities from other signs which shows that network was trained very precisely for all others signs.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



