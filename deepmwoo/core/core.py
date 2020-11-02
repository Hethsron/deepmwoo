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
    @file       core.py
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
from deepmwoo.core.shapes import cv4, cv2

class hub(object):
    """!
        @class      hub
        @brief      Define useful static methods which implements various
                    techniques such as :
                        - Face detection and tracking, for locating faces in images
                        and video sequences
                        - Face recognition for identifying unknown people using stored
                        database of known faces
    """

    @staticmethod
    def VideoTracking(video_source = None):
        """!
            @fn     VideoTracking
            @brief  Capture live stream and performs face detection, face tracking and
                    face recognition techniques for identifying unknown people

            @param[in]      video_source        Source video file to capture frame by frame  
        """
        # Initialize the MTCNN 
        detector = MTCNN()

        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_source)

        # Check whether VideoCapture is initialized or not
        while (cap.isOpened()):
            ret, frame = cap.read()

            # Check if frame is read correctly
            if ret is True:
                # Detect list of faces
                faces = detector.detect_faces(frame)

                count = 0
                for face in faces:
                    if face['box'] is not None:
                        x, y, width, height = face['box']
                        count += 1

                        try:
                            frame = cv4.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
                        except IndexError as err:
                            print(err)
                    else:
                        pass
                
                print(count)
                cv2.imshow("DeepMwoo",frame)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
            else:
                break
        
        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def ImageTracking(image_source = None):
        """!
            @fn     ImageTracking
            @brief  Read an image and performs face detection, face tracking and
                    face recognition techniques for identifying unknown people

            @param[in]      image_source        Source image file to read  
        """
        pass