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

