#!/usr/bin/env python3

# Copyright © 2020  Hethsron Jedaël BOUEYA and Yassine BENOMAR

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""!
    @file       train.py
    @brief      Basic Processing Algorithm to perform model training
    @details    
    
    @author     BOUEYA Hethsron Jedaël <hethsron-jedael.boueya@uha.fr>
                BENOMAR Yassine <yassine.benomar@uha.fr>
    
    @version    0.0.1
    @date       October 23th, 2020
    
    @note       For this program, we recommand to use the existing virtual
                environment that allows you to avoid installing Python
                packages globally which could break system tools or other projects 
    
    @pre        Before you can start installing or using packages in the
                existing virtual environment, you'll need to activate it.
                Activating this virtual environment will put the virtual
                environment-specifi python and pip executables into your
                shell's PATH.
                    -   On macOS and Linux, run :
                            source env/bin/activate
                    -   On Windows, run :
                            .\env\Scripts\activate
    @post       If you want to switch projects or otherwise leave this virtual
                environment, simply run :
                            deactivate
    @bug        No known bug to date
    @warning    Misuse could cause program crash
    @attention
    @remark
    @copyright  GPLv3+ : GNU GPL version 3 or later
                Licencied Material - Property of Stimul’Activ®
                © 2020 ENSISA (UHA) - All rights reserved.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from .access import os, re
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class training(object):
    """!
        @class      training
        @brief      Define useful static methods which implements various
                    techniques such as :
                        - Training process
    """

    @staticmethod
    def __load_faces__(directory = str()):
        """!
            @fn             __load_faces__
            @brief          Load images and extract faces for all images in a directory

            @param[in]      directory           Input directory of dataset
            @return                             Faces vectors      
        """

        # Initialize the list
        faces = list()

        # Enumerate files
        for filename in os.listdir(directory):
            # Skip any files that may be unauthorized
            if not re.compile('^(.*jpg)|(.*jpeg)|(.*png)$').match(filename):
                # Go to the next step
                continue

            # Create the absolute path of the file
            pathname = directory + filename

            # Load an image from a given file
            image = Image.open(pathname)
            
            # Convert image to RGB, if needed
            image = image.convert('RGB')

            # Resize image to tensor size
            image = image.resize((224, 224))

            # Convert RGB image to numpy array
            face = np.asfarray(image)

            # Check if face array is not None
            if face is not None:
                # Store face
                faces.append(face)

        # Return extracted faces
        return faces
    
    @staticmethod
    def __load_datasets__(directory = str()):
        """!
            @fn             __load_datasets__
            @brief          Load dataset that contains one subdir for each class that in turn contains images
                            and return faces and labels vectors

            @param[in]      directory           Input directory of dataset
            @return                             Faces and labels vectors
        """

        # Check if directory exists
        if not os.path.isdir(directory):
             assert False, 'Unknow directory : {}'.format(directory)

        # Initialize entries
        X, y = list(), list()

        # Enumerate forlders, on per class
        for subdir in os.listdir(directory):
            # Define absolute path
            path = directory + subdir + '/'

            # Skip any files that might be in the dir
            if not os.path.isdir(path):
                # Go to the next step
                continue

            # Load all faces in the subdirectory
            faces = training.__load_faces__(path)

            # Create labels
            labels = [subdir for _ in range(len(faces))]

            # Store faces and labels
            X.extend(faces)
            y.extend(labels)
        
        # Return faces and labels
        return np.asarray(X), np.asarray(y)

    @staticmethod
    def __save_datasets__():
        """!
            @fn             __save_datasets__
            @brief          Save datasets that contains faces and labels for train and validation
        """

        # Load dataset
        X, y = training.__load_datasets__('datasets/')

        # Save arrays to one file in compressed format
        np.savez_compressed('res/mwoo.npz', X, y)

    @staticmethod
    def load_arrays():
        """!
            @fn             load_arrays
            @brief          Load arrays from compressed format and return X, y vectors for train and validation

            @return         (X, y)
        """

        # Check if compressed file exists
        if not os.path.isfile('res/mwoo.npz'):
            # Save datasets
            training.__save_datasets__()

        # Load the face data
        data = np.load('res/mwoo.npz', allow_pickle = True)

        # Return vectors
        return data['arr_0'], data['arr_1']

    
    @staticmethod
    def __normalize__(X_train = None, X_test = None):
        """!
            @fn             __normalize__
            @brief          Normalize input vectors to unit norm and return normalized vectors

            @param[in]      X_train         Training Data Set
            @param[in]      X_test          Validation Data Set
            @return                         Normalized vectors of training set and validation set
        """

        # Return normalized vectors of training set and validation set
        return X_train / 255, X_test / 255

    @staticmethod
    def __transform_labels__(y_train = None, y_test = None):
        """!
            @fn             __transform_labels__
            @brief          Transform non-binary classes to a binary representation

            @param[in]      y_train             Input train label
            @param[in]      y_test              Input test label
            @return                             (y_train, y_test)

            For example, if we have a list of 6 persons each can have one of 3 classes
            Input : [
                        1, 
                        3,
                        3,
                        2,
                        1,
                        2
                    ]
                    
            Output : [
                        [1,0,0], # class 1 
                        [0,0,1], # class 3
                        [0,0,1], # class 3
                        [0,1,0], # class 2
                        [1,0,0], # class 1
                        [0,1,0]  # class 2
                    ]
        """
        # Concatenate labels
        y = np.concatenate((y_train, y_test), axis = 0)

        # Initialize LabelEncoder object
        encoder = LabelEncoder()

        # Make transformation [1,3,3,2,1,2] --> [0,2,2,1,0,1] 
        y = encoder.fit_transform(y)

        # Initialize OneHotEncoder
        encoder = OneHotEncoder()

        # Make transformation
        y = encoder.fit_transform(y.reshape(-1, 1))

        # Resplit train and test label
        Y_train = y[0:len(y_train)]
        Y_test = y[len(y_train):]

        # Return labels
        return Y_train.toarray(), Y_test.toarray()

    @staticmethod
    def __add_layers__(base_model = None, num_classes = int()):
        """!
            @fn             __add_layers__
            @brief          Add multiple layers in the base model in place of fully-connected layer

            @param[in]      base_model      Base model
            @param[in]      num_classes     Total number of classes in the model
            @return                         output layers
        """

        # Add a global spatial average pooling layer
        x = base_model.output
        x = Flatten()(x)

        # Let's add a logistic layer -- let's say we have units classes
        outputs = Dense(units = num_classes, activation = 'softmax')(x)

        # Return predictions
        return outputs

    @staticmethod
    def __create_model__(height = 224, width = 224, depth = 3, num_classes = int()):
        """!
            @fn             __create_model__
            @brief          create the base pre-trained model

            @param[in]      height          Height of the shape
            @param[in]      width           Width of the shape
            @param[in]      depth           Depth of the shape
            @param[in]      num_classes     Total number of classes in the model
            @return                         Model to use
        """

        # Define input shape of image
        input_shape = (height, width, depth)

        # Load VGG-16 pre-trained on ImageNet and without the fully-connected layers
        base_model = VGG16(include_top = False, 
                weights = 'imagenet', 
                input_tensor = None,
                input_shape = input_shape,
                classes = num_classes
        )

        # Only the new classifier is trained and the other layers are not re-trained.
        for layer in base_model.layers:
            layer.trainable = False

        # Add layers
        outputs = training.__add_layers__(base_model = base_model, num_classes = num_classes)

        # Groups layers into an object with training and inference features
        model = Model(inputs = base_model.input, outputs = outputs)

        # Return model
        return model

    @staticmethod
    def __generate_batches__():
        """!
            @fn             __generate_batches__
            @brief          Generate and return batches of tensor image data with real-time data augmentation

            @return                         (train_set, test_set)
        """

        # Define training data image generator
        train_gen = ImageDataGenerator(
                        rescale = 1./255,
                        featurewise_center = True,
                        featurewise_std_normalization = True,
                        rotation_range = 20,
                        width_shift_range = 0.2,
                        height_shift_range = 0.2,
                        brightness_range = (0.0, 1.0),
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        horizontal_flip = True
        )

        # Define validation data image generator
        test_gen = ImageDataGenerator(
                        rescale = 1./255
        )

        # Create set of training data
        train_set = train_gen.flow_from_directory(
                        directory = 'datasets',
                        target_size = (224, 224),
                        batch_size = 32,
                        class_mode = 'categorical'
        )

        # Create set of validation data
        test_set = test_gen.flow_from_directory(
                        directory = 'datasets',
                        target_size = (224, 224),
                        batch_size = 32,
                        class_mode = 'categorical'
        )

        # Return batches set
        return train_set, test_set

    @staticmethod
    def make(epochs = 7):
        """!
            @fn             make
            @brief          Perform training process

            @param[in]      epochs          Number of iterations
        """

        # Load data
        X, y = training.load_arrays()

        # Displaying message
        print('[+] Initiating The Training Process ...')
        print('[+] It Will Take A Few Seconds. Wait  ...')
        print('[+] List of classes is :', np.unique(y))

        # Define number of classes
        num_classes = len(np.unique(y))
        print('[+] Number of classes is : ', num_classes)

        # Define size of batch
        batch_size = len(X)
        print('[+] Size of batches is : ', batch_size)

        # Generate batches
        train_set, test_set = training.__generate_batches__()

        # Create model
        model = training.__create_model__(num_classes = len(glob('datasets/*')))

        # Print string summary of the network
        print(model.summary())
        
        # Compile the model
        model.compile(
                        loss = 'categorical_crossentropy',
                        optimizer = 'adam',
                        metrics = ['accuracy']
        )

        # Fit the model with batch gradient descent
        H = model.fit(
                        train_set,
                        epochs = epochs,
                        validation_data = test_set,
                        steps_per_epoch = len(train_set),
                        validation_steps = len(test_set)
        )

        # Save trained model
        model.save('models/mwoo.h5')

        # Evaluate the model on the test data using `evaluate`
        print('[+] Evaluate on test data')
        results = model.evaluate(test_set, batch_size = batch_size)
        print('[+] val loss, val acc : ', results)

        # Plot of Model Loss on Training and Validation Datasets
        plt.plot(H.history['loss'], color = 'blue', label = 'train loss')
        plt.plot(H.history['val_loss'], color = 'red', label = 'val loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('res/epoch-loss.pdf')
        plt.show()
        plt.close()

        # Plot of Model accuracy on Training and Validation Datasets
        plt.plot(H.history['accuracy'], color = 'blue', label = 'train acc')
        plt.plot(H.history['val_accuracy'], color = 'red', label = 'val acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('res/epoch-accuracy.pdf')
        plt.show()
        plt.close()
