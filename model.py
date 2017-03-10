import csv
import cv2 
import matplotlib.image as mpimg
import numpy as np
import sklearn

samples = []

# going through csv file in the udacity dataset folder
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

# do data shuffling
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # below to store the centre,left and right images in the images folder
            images =[]
                # below to store steering angles
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[0]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = "./data/IMG/" + filename
                    image = cv2.imread(local_path)
                    assert image is not None
                    images.append(image)
                correction  = 0.25d
                measurement = float(batch_sample[3])      # cast to floa
                measurements.append(measurement)
                # below is 2nd image
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)
            X_train = np.array(images)
            # print(len(X_train))
            y_train = np.array(measurements)
            # print(len(y_train))
            mirror = np.ndarray(shape=(X_train.shape))
            count = 0
            for i in range(len(X_train)):
                mirror[count] = cv2.flip(X_train[i], 1)
                count += 1
            mirror_angles = y_train * -1

            X_train = np.concatenate((X_train, mirror), axis=0)
            y_train = np.concatenate((y_train, mirror_angles),axis=0)

            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=20)
validation_generator = generator(validation_samples, batch_size=20)


import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


# using Nvidia's End to End Learning Approach
model = Sequential()
#normalizing data in the model itself.
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
# crop the image to remove portion of top and bottom of the training images
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))     
model.add(Dropout(0.2))
model.add(Dense(50)) 
model.add(Dropout(0.2))
model.add(Dense(10)) 
model.add(Dense(1)) 
# keras model compile, choose optimizer and loss func
model.compile(optimizer='adam',loss='mse')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples),validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=25)
model.save('model.h5')