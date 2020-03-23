import os
import csv
import cv2

import numpy as np
from numpy import ceil
from numpy.random import shuffle


from keras.models import Sequential, Model
from keras.layers import Cropping2D, Conv2D, Flatten, Dense, Lambda

import sklearn
from sklearn.model_selection import train_test_split

def input_batch_generator(samples, datadir, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = datadir + '/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def load_csv(filepath):
    samples = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        for line in reader:
            if count > 0: # Ignore first row (column header)
                samples.append(line)
            count += 1
            
    print("Loaded {} samples from {}".format(len(samples), filepath))
    return samples


def create_train_and_validation_samples(filepath, test_size=0.2):
    samples = load_csv(filepath)
    train_samples, validation_samples = train_test_split(samples, test_size=test_size)
    print("Split samples into {} training and {} validation samples".format(len(train_samples), len(validation_samples)))
    return train_samples, validation_samples
            
def create_nvidia_e2e_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def define_training_and_validation_data(model, datadir, batch_size, epochs=5):
    train_samples, validation_samples = create_train_and_validation_samples(datadir + '/driving_log.csv')    
    train_generator = input_batch_generator(train_samples, datadir, batch_size=batch_size)
    validation_generator = input_batch_generator(validation_samples, datadir, batch_size=batch_size)
    history_object = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_samples)/batch_size),
                        epochs=epochs, verbose=1)
    
    return model, history_object
    
print("Hello")

model, history_object = define_training_and_validation_data(create_nvidia_e2e_model(), '/home/workspace/CarND-Behavioral-Cloning-P3/training/data', 32)
model.save('model.h5')


