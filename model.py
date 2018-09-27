import cv2
import csv
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import sklearn

#Generate the images and measurments for training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                #Get the original image
                originalImage = cv2.imread(imagePath)
                
                #Adjust the image
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # Cut the image so that only the road is visible
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

#Returns the lines from the drivin log csv file inside a given data folder.
def getLines(folderName, skipHeader=False):
    lines = []
    #Open the csv file and read all the lines from it
    with open(folderName + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

# Create the preprocessing layers
def createPreProcessingLayers():
    model = Sequential()
    
    #Crop the images
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    # Normalize the images by dividing them by 255 so the range would be between 0 and 1
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    return model

#find all the images
def findImages(directory):
    cTotal = []
    lTotal = []
    rTotal = []
    mTotal = []
    
    #Get the lines from the csv file inside of this directory
    lines = getLines(directory)
    center = []
    left = []
    right = []
    measurements = []
    #Iterate through the lines and get the center, left and right images with the measurement
    for line in lines:
        measurements.append(float(line[3]))
        center.append(directory + '/' + line[0].strip().split('\\')[-1])
        left.append(directory + '/' + line[1].strip().split('\\')[-1])
        right.append(directory + '/' + line[2].strip().split('\\')[-1])

    cTotal.extend(center)
    lTotal.extend(left)
    rTotal.extend(right)
    mTotal.extend(measurements)

    return (cTotal, lTotal, rTotal, mTotal)

# Create the NVIDIA Model
def createNVidiaModel():
    model = createPreProcessingLayers()
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    
    #Add two dropout layers to avoid overfitting
    model.add(Dropout(0.5))
    model.add(Dropout(0.5))
    
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#Combine the images
def combineImages(center, left, right, measurement, correction):
    imgPaths = []
    imgPaths.extend(center)
    imgPaths.extend(left)
    imgPaths.extend(right)
    
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    
    return (imgPaths, measurements)

# Reading images locations.
centerPaths, leftPaths, rightPaths, measurements = findImages('data')

imgPaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format( len(imgPaths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(imgPaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create the NVIDIA Model
model = createNVidiaModel()

# Compile and train the model
model.compile(loss='mse', optimizer='adam')
batch_size = 32
history_object = model.fit_generator(train_generator, epochs=3, steps_per_epoch= \
                 int(len(train_samples) / batch_size), validation_data=validation_generator, \
                 verbose=1, validation_steps=int(len(validation_samples) / batch_size))

#Save the model and show some stats
model.save('model_new.h5')
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])