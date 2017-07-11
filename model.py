import csv

# if True -> use own recorded data, otherwise udacity data
ownDataSet = True
# if True -> use NVIDIA architecture, otherwise old traffic-sign-LeNet modification
nvidiaArchitecture = True


if ownDataSet == True:
    pathToCSV = './data/driving_log.csv'
else:
    pathToCSV = './data_udacity/driving_log.csv'

'''
##########################################
####### LOADING DATA FROM CSV FILE #######
##########################################
'''
lines = [] # lines of the csv file
with open(pathToCSV) as csvfile:
    csvData = csv.reader(csvfile)
    for line in csvData:
        lines.append(line)

from sklearn.model_selection import train_test_split
from random import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
import sklearn

'''
------------------------------
---- GENERATOR DEFINITION ----
------------------------------
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if ownDataSet == True:
                    name = './data/IMG/' + batch_sample[0].split('\\')[-1]
                else:
                    name = './data_udacity/IMG/' + batch_sample[0].split('/')[-1]

                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

'''
#########################################
####### CREATING AND TRAIN MODEL #######
#########################################
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
# normalizing images and shifting mean to 0
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# cropping images (removing irrelevant top and bottom of the images)
model.add(Cropping2D(cropping=((75, 20),(0, 0))))
if nvidiaArchitecture == True:
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
else:
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

if nvidiaArchitecture == True:
    model.save('model_nvidia.h5')
else:
    model.save('model.h5')