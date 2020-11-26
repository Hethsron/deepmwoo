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

from glob import glob
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from .access import re, os
from .shapes import cv2

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
        """!
            @fn     __extract_face__
            @brief  Extract face from an image and return face array instance resized to the
                    model size

            @param[in]      filename        String representing the image
            @param[in]      required_size   Model size of the image
        """
        # Load image from filename
        frame = cv2.imread(filename)

        # Create the detector, using default weights
        detector = MTCNN()

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Extract the bounding box from the first face
        x, y, width, height = faces[0]['box']

        # Define face array instance
        face_array = None

        # Extract the face
        try:
            face_array = frame[y:y + height, x:x + width]
        except IndexError as err:
            print(err)

        # Check if face array instance is not None
        if face_array is not None:
            # Convert face array instance to grayscale
            face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)

            # Resize face array instance to the model size
            face_array = cv2.resize(face_array, required_size, cv2.INTER_AREA)
            pass

        # Return the pixel to the model
        return face_array

    @staticmethod
    def __extract_face_2__(filename, required_size = (224, 224)):
        """!
            @fn     __extract_face__
            @brief  Extract face from an image and return face array instance resized to the
                    model size

            @param[in]      filename        String representing the image
            @param[in]      required_size   Model size of the image
        """
        # Load image from filname
        frame = cv2.imread(filename)

        # Define face array instance
        face_array = None

        # Load Haar Cascade file
        if os.path.isfile('models/haarcascade.xml'):
            faceCascade = cv2.CascadeClassifier('models/haarcascade.xml')

            # Detect faces in the frame
            faces = faceCascade.detectMultiScale(
                    frame,
                    scaleFactor = 1.3,
                    minNeighbors = 3,
                    minSize = required_size
            )

            # Extract the bounding box from the first face
            for (x, y, width, height) in faces:
                # Extract the face
                try:
                    face_array = frame[y:y + height, x:x + width]
                except IndexError as err:
                    print(err)

             # Check if face array instance is not None
            if face_array is not None:
                # Convert face array instance to grayscale
                face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)

                # Resize face array instance to the model size
                face_array = cv2.resize(face_array, required_size, cv2.INTER_AREA)
                pass

        # Return the pixel of the model
        return face_array

    @staticmethod
    def rescale_datasets(root_dir, required_size = (224, 224)):
        """!
            @fn     rescale_datasets
            @brief  Prepare good training and validation datasets for deep learning process.
                    This procedure improve the quality of a dataset by reducing dimensions 
                    and avoiding the situation when some of the values overweight others

            @param[in]      root_dir        String representing the directory of datasets
            @param[in]      required_size   Model size of the image
        """
        # Define filename list
        filenames = []

        # Define face array instance list
        face_arrays = []

        # Define dataset directories list
        root_dirs = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # Check if filename match with the required regex
                if re.compile('^(.*jpg)|(.*jpeg)|(.*png)$').match(file):
                    # Append dataset directory in the list
                    root_dirs.append(root_dir)

                    # Append filename in the list
                    filenames.append(root_dir + file)
        
        # Check if there are entries in datasets directories
        if not filenames and not root_dirs:
            print('[+] There are no entries in datasets directories')
            return

        for filename in filenames:
            # Extract face form an image
            face_array = net.__extract_face_2__(filename, required_size)
            
            # Check if face array is not None
            if face_array is not None:
                # Append extracted face array in the list
                face_arrays.append(face_array)

        # Check if there are extracted faces in the memory
        if not face_arrays:
            print('[+] There are any extracted faces in the memory')
            return

        for (i, image) in enumerate(face_arrays):
            # Save image to storage datasets
            cv2.imwrite('{}{}{}'.format(root_dirs[i], i + 1, '.jpg'), image)

        for filename in filenames:
            # Remove original filename
            os.remove(filename)

    @staticmethod
    def computation(epochs = 32, batch_size = 32, required_size = (224, 224)):
        """!
            @fn     computation
            @brief  Compute transfer learning process from pre-trained weights

            @param[in]      epochs          Number of epochs
            @param[in]      batch_size      Number of samples that will be propagated through the network.
            @param[in]      required_size   Model size of the image
        """
        # Check if datasets folders exist
        if os.path.isdir('datasets/train') and os.path.isdir('datasets/validation'):
            # Set relative path of train sets
            train_sets_dir = 'datasets/train'

            # Set relative path of validation sets
            validation_sets_dir = 'datasets/validation'

            # Set class number
            num_classes = len(glob('datasets/train/*'))

            # Set the transformation of each train image in the batch by a series of random translations, rotations, etc
            train_datagen = ImageDataGenerator(featurewise_center = False,
                                   featurewise_std_normalization = False,
                                   rotation_range = 30,
                                   width_shift_range = 0.4,
                                   height_shift_range = 0.4,
                                   shear_range = 0.15,
                                   fill_mode = 'nearest',
                                   zoom_range = 0.3,
                                   horizontal_flip = True,
                                   rescale = 1./255
            )

            # Set the transformation of each validation image
            validation_datagen = ImageDataGenerator(rescale=1./255)

            # Generate batches of tensor image data with real time data augmentation
            train_generator = train_datagen.flow_from_directory(train_sets_dir,
                                    target_size = required_size,
                                    batch_size = batch_size,
                                    class_mode = 'categorical',
                                    shuffle = True
            )

            # Generate batches of tensor image data with real time data augmentation
            validation_generator = validation_datagen.flow_from_directory(validation_sets_dir,
                                    target_size = required_size,
                                    batch_size = batch_size,
                                    class_mode = 'categorical',
                                    shuffle = True
            )

            # Load base model
            model = VGG16(input_shape = [224, 224] + [3],
                                    weights = 'imagenet',
                                    include_top = False
            )

            # Lock training of pretrained weights
            for layer in model.layers:
                layer.trainable = False

            # Reshape the input data into a format suitable for the convolutional layers
            x = Flatten()(model.output)

            # Group layers into an object with training and inference features
            model = Model(inputs = model.input, 
                                    outputs =  [
                                                Dense(units = num_classes, activation ='softmax')(x)
                                            ]
            )

            # Print a string summary of the network.
            model.summary()

            # Configures the model for training
            model.compile(optimizer = Adam(lr=0.001),
                                    loss = 'categorical_crossentropy',
                                    metrics = ['accuracy']
            )

            # Set callbacks
            callbacks = None

            # Check if pre-trained model exists
            if os.path.isfile('models/mwoo_model.h5'):
                # Define ModelCheckPoint callback
                checkpoint = ModelCheckpoint('models/mwoo_model.h5',
                                        monitor = 'val_loss',
                                        mode = 'min',
                                        save_best_only = True,
                                        verbose = 1
                )

                # Define EarlyStopping callback
                earlystop = EarlyStopping(monitor = 'val_loss',
                                        min_delta = 0,
                                        patience = 3,
                                        verbose = 1,
                                        restore_best_weights = True
                )

                # Define ReduceLROnPlateau callback
                reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                        factor = 0.2,
                                        patience = 3,
                                        verbose = 1,
                                        min_delta = 0.0001
                )

                # Update Callbacks
                callbacks = [earlystop, checkpoint, reduce_lr]

            # Trains the model on data generated batch-by-batch by a Python generator
            H = model.fit(train_generator,
                                    validation_data = validation_generator,
                                    epochs = epochs,
                                    callbacks = callbacks,
                                    steps_per_epoch = len(train_generator),
                                    validation_steps = len(validation_generator)
            )

            # Save trained model
            model.save('models/mwoo_model.h5')

            # Evaluate the model on a data generator
            loss, acc = model.evaluate(train_generator, steps = len(train_generator))

            print("The accuracy of train sets is : ", acc)
            print(loss, acc)

            # Evaluates the model on a data generator
            loss, acc = model.evaluate(validation_generator, steps = len(validation_generator))

            print("The accuracy of validation sets is :", acc)
            print(loss, acc)

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
        else:
            print('[+] There are any datasets')
            pass

