import os
import random
import pandas as pd
import sys
import numpy as np
from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import cv2

#gives the root directory
root_dir = os.path.abspath('.')

#gives the data directory
data_dir = root_dir + '/Data' 

#check if already trained model exists......
model_path = root_dir + '/my_model.h5'



if os.path.isfile(model_path):
	response = raw_input("An Already Trained model exists , do you want to use it? [y/n] : ")
	if response == 'y':
		#load the model
		model = load_model('my_model.h5')
		
		#Test an image input by user
		test_path = raw_input("Enter the path of image to test : ")
		temp = []
		img = imread(test_path)
		or_img = cv2.imread(test_path)
    		img = imresize(img, (32, 32))
    		temp.append(img.astype('float32'))
    		test_x = np.stack(temp)
    		pred = model.predict_classes(test_x)
    		print(pred[0])
    		plt.imshow(np.uint8(or_img))
    		
    		if pred[0] == 0:
    			print('MIDDLE')
    			cv2.imshow('MIDDLE',or_img)
    		elif pred[0] == 1:
    			print('OLD')
    			cv2.imshow('OLD',or_img)
    		else:
    			print('YOUNG')
    			cv2.imshow('YOUNG',or_img)

		
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
	else:
		#Loading the training and testing data that is stored in csv(having image name and 	corresponding class#(YOUNG,MIDDLE,OLD)
		train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
		test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

		#Display a random Image
		i = random.choice(train.index)
		img_name = train.ID[i]
		img = imread(os.path.join(data_dir, 'Train', img_name))


		#Display the corresponding class of the random image
		plt.imshow(np.uint8(img))
		plt.show()
		print("Age:"+ train.Class[i])


		#Train Data
		temp = []
		for img_name in train.ID:
    			img_path = os.path.join(data_dir, 'Train', img_name)
    			img = imread(img_path)
    			img = imresize(img, (32, 32))
    			img = img.astype('float32')
    			temp.append(img)

		train_x = np.stack(temp)


		#Test Data
		temp = []
		for img_name in test.ID:
    			img_path = os.path.join(data_dir, 'Test', img_name)
    			img = imread(img_path)
    			img = imresize(img, (32, 32))
    			temp.append(img.astype('float32'))

		test_x = np.stack(temp)


		#Rescale image from [0,255[ to [0,1] for normalization
		train_x = train_x / 255.
		test_x = test_x / 255.


		#Transform non-numerical labels to numerical labels
		lb = LabelEncoder()
		train_y = lb.fit_transform(train.Class)
		print(train_y)
		train_y = keras.utils.np_utils.to_categorical(train_y)
		print(train_y)



		input_num_units = (32, 32, 3)
		#hidden_num_units = 500
		#output_num_units = 3

		epochs = 30
		batch_size = 64

		model = Sequential()
		model.add(Conv2D(50, (5, 5), input_shape=(input_num_units), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(40, (5, 5), input_shape=(input_num_units), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Conv2D(30, (5, 5), input_shape=(input_num_units), activation='relu'))
		#model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(15, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(3, activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1,validation_split = 0.2)

		#save the model

		model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
		del model  # deletes the existing model

		# returns a compiled model
		# identical to the previous one
		model = load_model('my_model.h5')

		#prediction in numerical o/p's
		pred = model.predict_classes(test_x)
		#precition to class mapping
		pred = lb.inverse_transform(pred)

		#save the o/p on test classification
		test['Class'] = pred
		test.to_csv('sub02.csv', index=False)
