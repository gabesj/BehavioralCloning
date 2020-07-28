import os
import csv
import math
import random
from scipy import ndimage #allows to import image as RGB instead of CV2's BGR
import numpy as np #because keras needs images as numpy arrays
from matplotlib import pyplot as plt
from PIL import Image
import cv2

# This function gathers sets of images and corresponding steering angles.  It includes some additional data augmentation.
def GetData(directory, start_index, end_index):
    # Read in the csv file from the recorded data and save it
    lines = []
    filepath = directory + 'driving_log.csv'
    with open(filepath) as csvfile:
          reader = csv.reader(csvfile)
          line_num = 0  #the first line in the provided csv file is the column title headers
          for line in reader:
            line_num += 1
            if ((line_num > 1) and (line_num > start_index) and (line_num <= end_index)):  
                lines.append(line)

    #Add center images and steering measurements to collective arrays. Also add left and right images with compensated steering.
    image_paths=[]
    angle_values=[]
    for i in range(len(lines)):
        correction_offset = 0.20
        center_source_path = lines[i][0] #path to center image  
        center_filename = center_source_path.split('/')[-1] #get the filename from the path
        center_current_path = directory + 'IMG/' + center_filename
        left_source_path = lines[i][1] #path to left image  
        left_filename = left_source_path.split('/')[-1] #get the filename from the path
        left_current_path = directory + 'IMG/' + left_filename
        right_source_path = lines[i][2] #path to right image  
        right_filename = right_source_path.split('/')[-1] #get the filename from the path
        right_current_path = directory + 'IMG/' + right_filename
        image_paths.append(center_current_path)
        image_paths.append(left_current_path)
        image_paths.append(right_current_path)
        angle_values.append(float(lines[i][3]))
        angle_values.append(float(lines[i][3])+correction_offset)
        angle_values.append(float(lines[i][3])-correction_offset)
        
    # Augment the training data set by flipping each image along a vertical axis and saving it as a new file.  Then update the image path and angle value arrays to include these new images.  Since the image is flipped, the steering angle will be inverted.
    for i in range(len(image_paths)):
        flipped_filename = image_paths[i].split('/')[-1]
        flipped_new_path = '/opt/FlippedImages/flipped_' + flipped_filename
        if (True != os.path.isfile(flipped_new_path)):
            image = ndimage.imread(image_paths[i])
            flipped_image = Image.fromarray(np.fliplr(image))
            flipped_image.save(flipped_new_path)
        image_paths.append(flipped_new_path)
        flipped_angle = -angle_values[i]
        angle_values.append(flipped_angle)

    return image_paths,angle_values    

total_image_paths=[]
total_angle_values=[]
images,angles = GetData('/opt/carnd_p3/data/', 1, 100000)

for i in range(len(images)):
    total_image_paths.append(images[i])
    total_angle_values.append(angles[i])

# Split the data into training and testing sets. A shuffle operation is included in train_test_split.
from sklearn.model_selection import train_test_split
train_image_paths, validation_image_paths, train_angle_values, validation_angle_values = train_test_split(total_image_paths, total_angle_values, test_size=0.2, random_state=5)                
                
# Sanity Check
assert (len(train_image_paths)==len(train_angle_values)), 'Quantity of training images and angles do not match!'               
assert (len(validation_image_paths)==len(validation_angle_values)), 'Quantity of validation images and angles do not match!'                
training_data = list(zip(train_image_paths, train_angle_values))  
validation_data = list(zip(validation_image_paths, validation_angle_values))  
                
import sklearn

#Hyperparameters
batch_size = 150
epochs = 25

def generator(data, bs):
    num_samples = len(data)
    while 1:
        for offset in range(0, num_samples, bs):
            batch_samples = data[offset:offset+bs]
            images = []
            angles = []
            for i in range(len(batch_samples)):
                read_image = ndimage.imread(batch_samples[i][0])
                images.append(read_image)
                read_angle = float(batch_samples[i][1])
                angles.append(read_angle)
            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)

train_generator = generator(training_data, batch_size)
validation_generator = generator(validation_data, batch_size)

################################# Nvidia Model ####################################################
in_img_shape = (160,320,3) 

def resize(org_img):
    # Add this line to fix the load_model fail issue when using drive.py because tf is not defined there
    import tensorflow as tf
    return tf.image.resize_area(org_img, size=(66, 200)) #size Nvidia used

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, MaxPool1D
from keras.layers.convolutional import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)),input_shape=in_img_shape))
model.add(Lambda(resize))
model.add(Lambda(lambda x: x/255.000 - 0.500))
model.add(Conv2D(24,(5,5), activation='elu', subsample=(2, 2)))
model.add(Conv2D(36,(5,5), activation='elu', subsample=(2, 2)))
model.add(Conv2D(48,(5,5), activation='elu', subsample=(2, 2)))
model.add(Dropout(0.75))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
    
#try out different optimizers
from keras.optimizers import SGD, Adam
sgd = SGD(lr=0.0001)
adam = Adam(lr=0.0001)

# Compile the model
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(filepath='./checkpoint', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
    
print('train set length', len(training_data))
print('validation set length', len(validation_data))

#Train the model
model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_image_paths)/batch_size), 
                    epochs=epochs, verbose=1, 
                    validation_data=validation_generator,  
                    validation_steps=math.ceil(len(validation_image_paths)/batch_size), 
                    callbacks=[stopper])

model.save('model.h5')
 