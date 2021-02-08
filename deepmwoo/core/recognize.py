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
    @file       recognize.py
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

import time
from .access import os
from .shapes import cv2
from .train import training, np, Image
from keras.models import load_model

class recognition(object):
    """!
        @class      recognition
        @brief      Define useful static methods which implements various
                    techniques such as :
                        - Face detection and tracking, for locating faces in images and video sequences
                        - Face recognition for identifying unknown people using stored database of known faces
    """

    @staticmethod
    def fromStream(video_source = None):
        """!
            @fn             fromStream
            @brief          Perform face recognition process

            @param[in]      video_source        Source video file to capture frame by frame
        """

        # Check if Haar cascade file exists
        if os.path.isfile('res/haarcascade_frontalface_default.xml') and os.path.isfile('res/haarcascade_profileface.xml'):
            # Check if training model file exists
            if os.path.isfile('models/mwoo.h5'):
                # Load pre-trained model
                model = load_model('models/mwoo.h5')

                # Load Front Haar Cascade Classifier
                front_detector = cv2.CascadeClassifier('res/haarcascade_frontalface_default.xml')

                # Load Profile Haar Cascade Classifier
                profile_detector = cv2.CascadeClassifier('res/haarcascade_profileface.xml')

                # Define frame time for FPS
                prev_frame_time = 0
                new_frame_time = 0

                # Loads labels from the datasets
                _, y = training.load_arrays()

                # Loads unique entrie of labels
                names = np.unique(y)

                # Create a VideoCapture object
                cap = cv2.VideoCapture(video_source)

                # Check if VideoCapture is opened
                if not cap.isOpened():
                    print('[-] Error opening video file.')
                else:
                    try:
                        # Define video resolution (width * height)
                        cap.set(3, 640)
                        cap.set(4, 480)

                        # Define min window size to be recognized as a face
                        minW = 0.1 * cap.get(3)
                        minH = 0.1 * cap.get(4)

                        # Displaying message
                        print('[+] Initiating The Recognition Process ...')
                        print('[+] Look At The Camera And Wait ...')

                        # Loop
                        while True:
                            # Read image
                            ret, img = cap.read()

                            # Calculate FPS
                            new_frame_time = time.time()
                            fps = 1 / (new_frame_time - prev_frame_time)
                            prev_frame_time = new_frame_time

                            # Convert FPS to string
                            fps = int(fps)
                            fps = str(fps)

                            # Put FPS in the frame
                            cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

                            # Detects objects of different sizes in the input image and return as a list of rectangles
                            faces = front_detector.detectMultiScale(
                                img,
                                scaleFactor = 1.2,
                                minNeighbors = 5,
                                minSize = (int(minW), int(minH))
                            )

                            # Check if list of rectangles is empty
                            if not len(faces):
                                faces = profile_detector.detectMultiScale(
                                    img,
                                    scaleFactor = 1.2,
                                    minNeighbors = 5,
                                    minSize = (int(minW), int(minH))
                                )

                            # Define face
                            face = None

                            for (x, y, w, h) in faces:
                                # Draw rectangle around detected face
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                # Get face for prediction
                                face = img[y:y + h, x:x + w]

                            # Check if detected face is numpy array
                            if type(face) is np.ndarray:
                                # Resize face
                                face = cv2.resize(face, (224, 224))

                                # Get array from RGB image
                                face = Image.fromarray(face, 'RGB')
                                face = np.array(face)

                                # Expands dimension of image
                                face_array = np.expand_dims(face, axis = 0)

                                # Get prediction from the model
                                pred = model.predict(face_array)
                                print(pred)
                            else:
                                cv2.putText(img, 'No Face Found', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            # Display frame
                            cv2.imshow('<-> MWOO <->', img)

                            # Check user interruptions
                            # Press 'ESC' for exiting video
                            k = cv2.waitKey(10) & 0xff
                            if k == 27:
                                break
                            
                            # Check if frame is not read correctly
                            if not ret:
                                print('[-] Cannot retrive frame - stream may have ended. Existing ...')
                    except:
                        print('[-] Video stream has ended caused by internal error')

                # Do a bit of cleanup
                print('[+] End Of Recognition Process')
                cap.release()
                cv2.destroyAllWindows()
            else:
                # Built-in assert statement to find errors
                assert False, 'Training Model File `mwoo.h5` Not Found In `model` Folder'
        else:
            # Built-in assert statement to find errors
            assert False, 'Cascade Haar Classifiers `haarcascade_frontalface_default.xml` and `haarcascade_profileface.xml` Not Found In `res` Folder'

    @staticmethod
    def fromImage(image_source = None):
        """!
            @fn             fromImage
            @brief          Perform face recognition process

            @param[in]      image_source        Source image file to read
        """

        # Check if Haar cascade file exists
        if os.path.isfile('res/haarcascade_frontalface_default.xml') and os.path.isfile('res/haarcascade_profileface.xml'):
            # Check if training model file exists
            if os.path.isfile('models/mwoo.h5'):
                # Load pre-trained model
                model = load_model('models/mwoo.h5')

                # Load Front Haar Cascade Classifier
                front_detector = cv2.CascadeClassifier('res/haarcascade_frontalface_default.xml')

                # Load Profile Haar Cascade Classifier
                profile_detector = cv2.CascadeClassifier('res/haarcascade_profileface.xml')

                # Define frame time for FPS
                prev_frame_time = 0
                new_frame_time = 0

                # Loads labels from the datasets
                _, y = training.load_arrays()

                # Loads unique entrie of labels
                names = np.unique(y)

                
            else:
                # Built-in assert statement to find errors
                assert False, 'Training Model File `mwoo.h5` Not Found In `model` Folder'
        else:
            # Built-in assert statement to find errors
            assert False, 'Cascade Haar Classifiers `haarcascade_frontalface_default.xml` and `haarcascade_profileface.xml` Not Found In `res` Folder'
