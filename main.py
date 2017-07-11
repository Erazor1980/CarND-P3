import csv
import cv2
import numpy as np

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

images = []
steer_angles = []

print("Loading data... ")
for line in lines[1:]:
    # get image paths, load images with openCV and add them to the list
    if ownDataSet == True:
        img_path = line[0].split('\\')[-1]
        img_center = cv2.imread("./data/IMG/" + img_path)
        img_path_left = line[1].split('\\')[-1]
        img_left = cv2.imread("./data/IMG/" + img_path_left)
        img_path_right = line[2].split('\\')[-1]
        img_right = cv2.imread("./data/IMG/" + img_path_right)
    else:
        img_path = line[0].split('/')[-1]
        img_center = cv2.imread("./data_udacity/IMG/" + img_path)
        img_path_left = line[1].split('/')[-1]
        img_left = cv2.imread("./data_udacity/IMG/" + img_path_left)
        img_path_right = line[2].split('/')[-1]
        img_right = cv2.imread("./data_udacity/IMG/" + img_path_right)
    #cv2.imshow('bla', image)
    #cv2.waitKey(0)
    #images.append(image)
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)

    # get steering angle and add it to list
    # create adjusted steering measurements for the side camera images
    steer_angle = (float)(line[3])
    correction = 0.25
    steer_angle_left = steer_angle + correction
    steer_angle_right = steer_angle - correction
    #steer_angles.append(steer_angle)
    steer_angles.append(steer_angle)
    steer_angles.append(steer_angle_left)
    steer_angles.append(steer_angle_right)

print("done.\nOriginal Dataset:\n\tNumber images:", len(images), "\n\tNumber steering angles:", len(steer_angles))

'''
##################################
####### AUGMENTING DATASET #######
##################################
'''
aug_images = []
aug_steer_angles = []

for image, angle in zip( images, steer_angles ):
    aug_images.append(image)
    aug_steer_angles.append(float(angle))
    # flipping images horizontally
    aug_images.append(cv2.flip(image, 1))
    aug_steer_angles.append(float(angle) *-1.0)

X_train = np.array(aug_images)
y_train = np.array(aug_steer_angles)

print("\nAugmented Dataset:\n\tNumber images:", len(aug_images), "\n\tNumber steering angles:", len(aug_steer_angles))

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
model.fit(X_train, y_train, validation_split=0.25, shuffle=True, nb_epoch=3, batch_size=64)

if nvidiaArchitecture == True:
    model.save('model_nvidia.h5')
else:
    model.save('model.h5')


'''
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''