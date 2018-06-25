import numpy as np
import pandas as pd
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import pdb


def flat_to_one_hot(labels):
	num_classes = np.unique(labels).shape[0]
	num_labels = labels.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels,num_classes))
	labels_one_hot.flat[index_offset + labels.ravel()] = 1
	return labels_one_hot

def get_csv_traindata(valid_size=2000):

	data = pd.read_csv('Data/train.csv')
	imgs = data.iloc[:,1:].values
	labels = data['label'].values.ravel()
	# ravel() is used to return the 1-D array of the input
	# pdb.set_trace()
	# Converting the images from uint8 to double:
	imgs = np.multiply(imgs,1.0/255.0)
	# Converting the labels to one hot encoding:
	# labels = flat_to_one_hot(labels)

	# Spliting the data into validation and training data:    
	validation_imgs = imgs[:valid_size]
	validation_labels = labels[:valid_size]
	train_imgs = imgs[valid_size:]
	train_labels = labels[valid_size:]

	# Converting the images from flat to matrix form:
	train_imgs = train_imgs.reshape(train_imgs.shape[0],1,28,28)
	print(train_imgs.shape)
	validation_imgs = validation_imgs.reshape(validation_imgs.shape[0],1,28,28)
	print(validation_imgs.shape)
	# Return the data:
	return (train_imgs,train_labels),(validation_imgs,validation_labels)


if __name__ == '__main__':
	(X_train,y_train),(X_val,y_val) = get_csv_traindata()

	# Constructing the model which is in the form of a sequential model(feature of keras)
	model = Sequential()
	model.add(Conv2D(nb_filter=32,nb_row=5,nb_col=5,border_mode='same',input_shape=(1,28,28)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

	model.add(Conv2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same',input_shape=(32,14,14)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	if os.path.exists('./model_weights.h5'):
	    model.load_weights('model_weights.h5')

	# Training the model established above:
	model.fit(X_train,y_train,batch_size=50,nb_epoch=5,verbose=1,validation_data=(X_val,y_val))
	score = model.evaluate(X_val,y_val,verbose=0)
	print('Validation score:', score[0])
	print('Validation accuracy:', score[1])

	# Saving the model:
	json_string = model.to_json()
	open('trained_model.json','w').write(json_string)

	# Saving the weights
	model.save_weights('weighted_model.h5',overwrite=True)
