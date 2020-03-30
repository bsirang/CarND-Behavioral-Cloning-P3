# **Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

### Model Architecture and Training Strategy

#### 1. Nvidia Architecture

The model used was derived from Nvidia's paper on deep learning for self-driving cars, which was a convolutional neural network with several convolutional layers followed by several fully connected layers.

This architecture was chosen because it was demonstrated to perform well for this particular end-to-end learning model in which steering wheel angles are inferred directly from dashboard cameras.

More information here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

#### 2. Preventing Overfitting
In order to prevent overfitting a handful of techniques were used.

1. **Dropout layers:** There were a couple dropout layers implemented after the convolutions. For both layers 20% dropout rate was chosen. Initially I tried placing the dropout layers in the middle of the convolution layers, but in the end seemed to get better performance by placing them in front of the first couple fully connected layers. More time could be spent understanding where to put dropout layers to maximize performance.

2. **Left/Right Mirroring:** In general, a neural net model is susceptible to overfitting if the training set is too small and/or doesn't capture enough of a variety of scenarios, which causes the network to bias itself to patterns that are not generic enough. As for self-driving, the images can be mirrored with respect to their left and right side. This doubles the training data and doesn't allow the network to bias itself towards left or right turns.

3. **Multiple Cameras** Another technique to increase the amount of training data is to use the images from all three cameras. While the center camera is considered the source of truth when it comes to steering angles, the left and right cameras can also be used with "corrected" steering angles. While the correction offset can probably be derived theoretically based on the characteristics of the camera, for the sake of this exercise it was chosen empirically by trial-and-error.

4. **Driving both Directions:** despite only having two tracks to work with in the simulator, more unique data can be generated simply by driving the track in the opposite direction.


#### 3. Model parameter tuning

The Adam optimizer was used with default parameters, so no model parameter tuning was done.

#### 4. Appropriate training data

The following types of training data was used:

* Data from driving around both tracks in the default direction, while keeping the car centered as much as possible.
* Same as above except from driving the tracks in reverse
* Left and right camera images, with angle correction offsets
* Left/right mirrored images of the center camera

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was to adopt Nvidia's end-to-end neural net architecture for self-driving cars given that it has been demonstrated to perform well on this type of deep learning problem.

The architecture was implemented in Keras with the addition of dropout layers to mitigate overfitting and a cropping layer to crop the sky and the hood of the vehicle.

It took several iterations to get a well performing model. The primary challenge was collecting good training data.

The secondary challenge was deriving the augmented data appropriately. It was straight forward to mirror the center camera images, but it took some more work to use the left and right camera images effectively. This is because it took some effort to determine reasonable steering angle correction offsets.

The third challenge was finding the right hyperparameters:
* Number of epochs
* Dropout layer positioning and drop rates
* Left and right camera steering angle correction offset

All of these parameters were chosen based on a trial-and-error approach.

#### 2. Final Model Architecture

The final model architecture is implemented in Keras as follows:

```python
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)))) # Trim images to see only section of road
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

#### 3. Creation of the Training Set & Training Process

##### Data Set
The initial attempt involved training data from driving both tracks while centering the vehicle. Enough practice driving was done beforehand to ensure smooth turning around all corners. The model performed poorly with this data. The moment the vehicle veered too far way from center, the model couldn't cope and caused the vehicle to run off the road.

The next attempt included both tracks being driven in reverse in an attempt to provide more diverse data. The model still didn't perform well primarily due to the reasons mentioned above.

After these attempts, more data was collected. This time the focus was on recovery data. The idea was to collect data from scenarios in which the vehicle was at the edge of the road and had to make a correction to stay on the road. This helped the performance a lot.

At this point I was close but I still had some situations where the vehicle would drive off the road. To combat this I collected more recovery scenarios, with a particular focus on the areas of the track that the algorithm struggled with the most. This was the parts of the track where there was a break in the road barrier with a dirt turnoff.

In the end, I managed to train an algorithm that would keep the car on the road, however there were still a close call towards the end of the track. I'm confident that more training data would further improve the performance.

##### Training Process
The collected data was shuffled and then split up into training and validation data sets using an 80%/20% split. This amounted to 18472 training and 4618 validation samples.

I settled on 11 epochs as that's when I observed the loss start to taper off.

Here's the results of the final training:
```
Epoch 1/11
653/653 [==============================] - 28s 43ms/step - loss: 0.0855 - val_loss: 0.0595
Epoch 2/11
653/653 [==============================] - 26s 40ms/step - loss: 0.0666 - val_loss: 0.0507
Epoch 3/11
653/653 [==============================] - 26s 40ms/step - loss: 0.0578 - val_loss: 0.0449
Epoch 4/11
653/653 [==============================] - 26s 39ms/step - loss: 0.0520 - val_loss: 0.0511
Epoch 5/11
653/653 [==============================] - 26s 39ms/step - loss: 0.0521 - val_loss: 0.0463
Epoch 6/11
653/653 [==============================] - 26s 39ms/step - loss: 0.0496 - val_loss: 0.0457
Epoch 7/11
653/653 [==============================] - 26s 39ms/step - loss: 0.0484 - val_loss: 0.0398
Epoch 8/11
653/653 [==============================] - 26s 40ms/step - loss: 0.0458 - val_loss: 0.0386
Epoch 9/11
653/653 [==============================] - 26s 40ms/step - loss: 0.0468 - val_loss: 0.0442
Epoch 10/11
653/653 [==============================] - 26s 40ms/step - loss: 0.0446 - val_loss: 0.0492
Epoch 11/11
653/653 [==============================] - 26s 40ms/step - loss: 0.0455 - val_loss: 0.0364
```
