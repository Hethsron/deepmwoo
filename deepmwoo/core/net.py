#!/usr/bin/env python3

# Copyright © 2020  Hethsron Jedaël BOUEYA

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
    @file       net.py
    @brief      Basic Processing Hub For Facial Recognition Algorithm Using Deep Learning
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

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from .access import re, os
from .log import maker

class net(object):
    """!
        @class      net
        @brief      Define useful static methods which implements various
                    techniques such as :
                        - Face extraction that extract a single face from given photography
                        - Data rescaling that improve the quality of dataset
    """

    @staticmethod
    def __extract_face__(filename, required_size = (224, 224)):
        """
            @fn         __extract_face__
            @brief      Extract a single face from a given photograph

            @param[in]      filename            Input file that contains a face to extract
            @param[in]      required_size       Model size of image
            @return                             Single face extracted
        """
        # Load an image from a given file
        image = Image.open(filename)

        # Convert the image to RGB, if needed
        image = image.convert('RGB')

        # Convert RGB image to numpy array
        pixels = np.asfarray(image)

        # Create the detector, using default weights
        detector = MTCNN()

        # Detect faces in the image
        faces = detector.detect_faces(pixels)

        # Extract the bounding box from the first face
        x, y, width, height = faces[0]['box']

        # Fix bug if needed
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + width, y1 + height

        # Define face array instance
        face = None

        # Avoid out of box exception
        try:
            # Extract the face
            face = pixels[y1:y2, x1:x2]
        except IndexError as err:
            print(err)

        # Define image
        image = None

        try:
            # Resize pixels to the model size
            image = Image.fromarray((face).astype(np.uint8))
            image = image.resize(required_size)
        except TypeError as err:
            print(err)

        # Return face array
        return np.asarray(image)

    @staticmethod
    def __load_faces__(directory = str()):
        """
            @fn         __load_faces__
            @brief      Load images and extract faces for all images in a directory

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
            path = directory + filename

            # Define face array instance
            face = None

            # Get face
            face = net.__extract_face__(path)

            # Check if face array is not None
            if face is not None:
                # Store face
                faces.append(face)

        # Return extracted faces
        return faces

    @staticmethod
    def __load_datasets__(directory = str()):
        """
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
            faces = net.__load_faces__(path)

            # Create labels
            labels = [subdir for _ in range(len(faces))]

            # Store faces and labels
            X.extend(faces)
            y.extend(labels)
        
        # Return faces and labels
        return np.asarray(X), np.asarray(y)

    @staticmethod
    def save_datasets():
        """
            @fn             save_datasets
            @brief          Save datasets that contains faces and labels for train and validation
        """
        # Load train dataset
        X_train, y_train = net.__load_datasets__('datasets/train/')

        # Load validation dataset
        X_test, y_test = net.__load_datasets__('datasets/validation/')

        # Save arrays to one file in compressed format
        np.savez_compressed('datasets/mwoo_faces.npz', X_train, y_train, X_test, y_test)

    @staticmethod
    def __load_data__():
        """
            @fn             __load_data__
            @brief          Load data from compressed format and return X_train, y_train, X_test and y_test vectors

            @return         (X_train, y_train, X_test, y_test)
        """
        # Check if compressed file exists
        if not os.path.isfile('datasets/mwoo_faces.npz'):
            # Save datasets
            net.save_datasets()

        # Load the face data
        data = np.load('datasets/mwoo_faces.npz')

        # Return vectors
        return data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    @staticmethod
    def __normalize_vectors__(X_train, X_test):
        """
            @fn             __normalize_vectors__
            @brief          Normalize input vectors to unit norm and return normalized vectors

            @param[in]      X_train             Input train vector
            @param[in]      X_test              Input validation vector
            @return                             (X_train, X_test)
        """
        # Return normalized vectors , divide by 255 so that everything is between 0 and 1 
        return X_train / 255, X_test / 255

    @staticmethod
    def __transform_labels__(y_train, y_test):
        """
            @fn             __transform_labels__
            @brief          Transform non-binary classes to a binary representation

            @param[in]      y_train             Input train label
            @param[in]      y_test              Input test label
            @return                             (y_train, y_test)

            For example, if we have a list of 6 flowers each can have one of 3 classes
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
    def __add_layers__(base_model, num_classes = int()):
        """
            @fn             __add_layers__
            @brief          Add multiple layers in the base model in place of fully-connected layer

            @param[in]      base_model      Base model
            @param[in]      num_classes     Total number of classes in the model
            @return                         Model to use
        """
        # Add a global spatial average pooling layer
        x = base_model.output
        x = Flatten()(x)

        # Let's add a logistic layer -- let's say we have units classes
        predictions = Dense(units = num_classes, activation = 'softmax')(x)

        # Return predictions
        return predictions

    @staticmethod
    def __create_model__(height = 224, width = 224, depth = 3, num_classes = int()):
        """
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
        predictions = net.__add_layers__(base_model = base_model, num_classes = num_classes)

        # Groups layers into an object with training and inference features
        model = Model(inputs = base_model.input, outputs = predictions)

        # Return model
        return model

    @staticmethod
    def __generate_batches__(x_train = None, y_train = None, y_train_binary = None, y_test = None, batch_size = int()):
        """
            @fn             __generate_batches__
            @brief          Generate and return batches of tensor image data with real-time data augmentation

            @param[in]      x_train         Train sample data
            @param[in]      y_train         Train labels
            @param[in]      y_train_binary  Train labels transformed
            @param[in]      y_test          validation labels
            @return                         (X_batch, Y_batch)
        """
        # Set train data image generator
        datagen = ImageDataGenerator(
                    rescale = 1./255,
                    featurewise_center = True,
                    featurewise_std_normalization = True,
                    rotation_range = 20,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    brightness_range = (0.0,1.0),
                    horizontal_flip = True
        )

        # Fits the train data generator to train sample data
        datagen.fit(x = x_train, augment = True)

        # Takes train sample data & label arrays, generates batches of augmented data
        X_train = list()
        Y_train = list()

        X_train.append(x_train)
        Y_train.append(y_train_binary)

        for x_batch, y_batch in datagen.flow(x = x_train, y = y_train, batch_size = batch_size):
            X_train.append(x_batch)
            y_batch_binary, _ = net.__transform_labels__(y_batch, y_test)
            Y_train.append(y_batch_binary)
            if len(X_train) >= 5:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        
        # Join sequence of arrays
        X_batch = np.concatenate(X_train)
        y_batch = np.concatenate(Y_train)

        # Returns batches of tensor images
        return X_batch, y_batch

    @staticmethod
    def computation(epochs = 100, learning_rate = 0.01, momentum = 0.9):
        """
            @fn             computation
            @brief          Compute transfer learning process from pre-trained weights

            @param[in]      epochs          Number of epochs
            @param[in]      learning_rate   A Tensor, floating point value
            @param[in]      momentum        Float hyperparameter >= 0 that accelerates gradient descent
        """
        # Load data
        X_train, y_train, X_test, y_test = net.__load_data__()

        # Normalize input vectors
        X_train, X_test = net.__normalize_vectors__(X_train, X_test)

        # Transform labels
        y_train_binary, y_test_binary = net.__transform_labels__(y_train, y_test)

        # Define number of classes
        num_classes = len(np.unique(y_train))

        # Define batch size
        batch_size = len(X_train)

        # Generate batches of tensor image data with real-time data augmentation
        X_batch, y_batch = net.__generate_batches__(
                            x_train = X_train, 
                            y_train = y_train,
                            y_train_binary = y_train_binary,
                            y_test = y_test,
                            batch_size = batch_size
        )

        # Create model
        model = net.__create_model__(num_classes = num_classes)

        # Prints a string summary of the network.
        print(model.summary())

        # Compile the new model
        model.compile(optimizer = SGD(learning_rate = learning_rate, momentum = momentum),
                                            loss = 'categorical_crossentropy',
                                            metrics = ['accuracy']
        )

        # Define callbacks
        callbacks = None

        # Fit the model with batch gradient descent
        H = model.fit(x = X_batch, 
                        y = y_batch, 
                        batch_size = batch_size,
                        epochs = epochs,
                        callbacks = callbacks,
                        verbose = True,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data = (X_test, y_test_binary)
        )

        # Save trained model
        model.save('models/mwoo_model.h5')

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model.evaluate(X_test, y_test_binary, batch_size = batch_size)
        print("val loss, val acc:", results)

        # Plot of Model Loss on Training and Validation Datasets
        plt.plot(H.history['loss'], color = 'blue', label = 'train loss')
        plt.plot(H.history['val_loss'], color = 'red', label = 'val loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('src/assets/pdf/epoch-loss.pdf')
        plt.show()
        plt.close()

        # Plot of Model accuracy on Training and Validation Datasets
        plt.plot(H.history['accuracy'], color = 'blue', label = 'train acc')
        plt.plot(H.history['val_accuracy'], color = 'red', label = 'val acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('src/assets/pdf/epoch-accuracy.pdf')
        plt.show()
        plt.close()