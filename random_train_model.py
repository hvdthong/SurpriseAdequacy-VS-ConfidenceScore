import argparse

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.optimizers import SGD

from generate_sample_data import load_random_sample_data
import os 

CLIP_MIN = -0.5
CLIP_MAX = 0.5
weight_decay = 0.0005

def train(args, ntime, data):
    train, test = data
    x_train, y_train = train
    x_test, y_test = test

    if args.d == "mnist":
        layers = [
            Conv2D(64, (3, 3), padding="valid", input_shape=(28, 28, 1)),
            Activation("relu"),
            Conv2D(64, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
        ]

    elif args.d == "cifar":
        layers = [
            Conv2D(64, (3, 3), padding="same", input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.3),

            Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),            
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),

            Flatten(),         
            Dense(512, kernel_regularizer=l2(0.0005)),
            Activation('relu'),
            BatchNormalization(),

            Dropout(0.5),
            Dense(10),
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))

    if args.d == 'cifar':
        print(model.summary())
        opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
        model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )
    elif args.d == 'mnist':
        print(model.summary())    
        model.compile(
            loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
        )
    else:
        print('Input the wrong dataset! Please retype again')
        exit()

    # checkpoint
    path = './random_sample_model/%s/%i/' % (args.d, ntime)

    if not os.path.exists(path):
        os.makedirs(path)

    filepath= path + "model_-{epoch}-.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    if args.d == 'mnist':
        epochs = 100
    if args.d == 'cifar':
        epochs = 250

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--r", "-r", help="How many times we want to sample dataset", type=int, default=100)
    parser.add_argument("--s", "-s", help="Start times of random sampling", type=int, default=0)
    parser.add_argument("--e", "-e", help="End times of random sampling", type=int, default=100)
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    print(args)
    
    save_path = '../2020_FSE_Empirical/%s' % args.d
    for t in range(args.r):
        data_train, data_test = load_random_sample_data(save_path=save_path, ntime=t)        
        x_train, y_train = data_train
        x_test, y_test = data_test

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        data = (data_train, data_test)
        train(args, t, data)