import os
import sys
import numpy as np
import pandas as pd

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


class path_define():
    train_path = sys.argv[1]
    model_path = "model/model.h5"
    
    
def Load_training_data(train_data_path):
    train_data = pd.read_csv(train_data_path, sep=',', header=0)
    
    X_train = train_data['feature']
    X_train = X_train.str.split(' ', expand=True)
    X_valid = X_train[26709:]
    X_train = X_train[:26709]
    X_valid = np.array(X_valid.values)
    X_train = np.array(X_train.values)
    X_valid = np.reshape(X_valid, (2000, 48, 48, 1))
    X_train = np.reshape(X_train, (26709, 48, 48, 1))
    
    
    Y_train = train_data['label'].values
    Y_valid = Y_train[26709:]
    Y_train = Y_train[:26709]
    Y_valid = np.array(Y_valid)
    Y_train = np.array(Y_train)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes=7)
    Y_train = np_utils.to_categorical(Y_train, num_classes=7)
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_valid.shape)
    print(Y_valid.shape)
    return X_train, Y_train, X_valid, Y_valid
    
    


if __name__ == '__main__':
    path = path_define()
    X_train, Y_train, X_valid, Y_valid = Load_training_data(path.train_path)
    
    # build model  
    
    input_img = Input(shape=(48, 48, 1))

    block1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    block1 = Conv2D(64, (3, 3), padding='same', activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(block1)

    block2 = Conv2D(256, (3, 3), activation='relu', padding='same')(block1)
    block2 = Conv2D(256, (3, 3), activation='relu', padding='same')(block2)
    block2 = MaxPooling2D(pool_size=(3, 3))(block2)

    block3 = Conv2D(512, (3, 3), activation='relu', padding='same')(block2)
    block3 = Conv2D(512, (3, 3), activation='relu', padding='same')(block3)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)

    block4 = Conv2D(256, (3, 3), activation='relu', padding='same')(block3)
    block4 = Conv2D(256, (3, 3), activation='relu', padding='same')(block4)
    block4 = MaxPooling2D(pool_size=(3, 3))(block4)

    block5 = Flatten()(block4)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)
    
    
    """
    input_img = Input(shape=(48, 48, 1))

    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    
    
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)
    """
    
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    plot_model(model, to_file='model.png')
    model.summary()
    
    #--------------------------------------------------
    
    # Start training
    
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    validgen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    
    batch_size = 64
    
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=(len(X_train)*3), epochs=50,
                        validation_data=(X_valid, Y_valid))
    
    
    """
    model.fit(  X_train, 
                Y_train, 
                batch_size=128, 
                epochs=60,
                validation_split = 0.2)
    """
    model.save(path.model_path)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
