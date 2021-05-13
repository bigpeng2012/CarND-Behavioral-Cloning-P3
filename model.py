import os
import csv
import cv2
import numpy as np
import sklearn

#   driving_log.csv: 
#   center,left,right,steering,throttle,brake,speed
#   IMG/center_2016_12_01_13_30_48_287.jpg, IMG/left_2016_12_01_13_30_48_287.jpg, IMG/right_2016_12_01_13_30_48_287.jpg, 0, 0, 0, 22.14829
#   IMG/center_2016_12_01_13_30_48_404.jpg, IMG/left_2016_12_01_13_30_48_404.jpg, IMG/right_2016_12_01_13_30_48_404.jpg, 0, 0, 0, 21.87963

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  header = next(reader)
  for line in reader:
    lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: #always on
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                
                mod_center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                mod_left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                mod_right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3])+0.2
                right_angle = float(batch_sample[3])-0.2
                
                images.append(mod_center_image)
                angles.append(center_angle)
                
                images.append(mod_left_image)
                angles.append(left_angle)
                
                images.append(mod_right_image)
                angles.append(right_angle)
            
                #flipping
                images.append(cv2.flip(mod_center_image,1))
                angles.append(center_angle*-1)
                
                images.append(cv2.flip(mod_left_image,1))
                angles.append(left_angle*-1)
                
                images.append(cv2.flip(mod_right_image,1))
                angles.append(right_angle*-1)
       
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)

validation_generator = generator(validation_samples, batch_size=32)



from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()

model.add(Lambda(lambda x:(x/255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

history_object = model.fit_generator(train_generator,samples_per_epoch=len(train_samples),validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5,verbose =1)

model.save('model.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


