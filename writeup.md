**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.jpg "Center driving"
[image2]: ./examples/image2.png "Right to center"
[image3]: ./examples/image3.png "Right to center"
[image4]: ./examples/image4.png "Right to center"
[image5]: ./examples/image5.png "Track 2"

## Rubric Points
---
### Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model and loss visualization
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* (main.py with previous attempts, using image flipping, but without generators)

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_nvidia.h5
```

3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
There are several parameter (bools) which can be set at the beginning of the file, like using side cameras, displaying loaded images, which data set or which model to use.

### Model Architecture and Training Strategy

1. An appropriate model architecture has been employed

My (main) model (nvidia architecture) consists of a convolution neural network including:
* 3 convolution layers with 5x5 filters, subsamples of (2,2) and depths from 24 to 48
* 2 convolution layers with 3x3 filters and depth of 64
* 1 flatten layer
* 4 dense layers (sizes: 100, 50, 10, 1)

The model includes RELU layers to introduce nonlinearity (for all convolution layers).
The data is normalized and mean shifted to 0 using a Keras lambda layer (code line 115).
To remove irrelevant regions of the image (top and bottom parts) a cropping layer is used (code line 117).

2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 31). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model parameter tuning

The model used an adam optimizer (loss "MSE"), so the learning rate was not tuned manually (model.py line 139).

4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road etc.
Following scenarios have been recorded:
* driving around track 1

* driving around track 1 counterclockwise 
* driving from side of the road to its middle (several times)
    
* short driving on track 2

There is a parameter to use data from the side cameras ("useSideCameras", line 10), the appropriate angles are calculated with the correction value of 0.25 (see line 71).
For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

1. Solution Design Approach

First I implemented a very simple model und get everything running. Then I concentrated on 2 different approaches for the model architecture:
* LeNet (modified from the TSR project)
* NVIDIA (similar to here https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

Both approaches were implemented and tested. You can switch between both using the "nvidiaArchitecture" variable in line 6 of the code.

After adding each layer to the model I trained it and tested on the track to see the results. 

2. Final Model Architecture

Best results were gained with the NVIDIA-like approach, see line 119ff.

3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving (clockwise and counterclockwise). Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to do it on its own. These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Additionally I recorded some curves from track 2:
![alt text][image5]

After the collection process, I had 10.815 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
