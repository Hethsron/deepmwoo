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
    @file       log.py
    @brief      Basic Log Manager For Facial Recognition Algorithm Using Deep Learning
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

from .shapes import cv2
from hashlib import md5
import pyshine as ps
import pickle as pc

class maker(object):
    """!
        @class      maker
        @brief      Define useful static methods which implements various
                    techniques to display and save log of result
    """

    @staticmethod
    def legend(given_frame = None, score_people = None):
        """!
            @fn     legend
            @brief  Returns image with bounded box of legend

            @param[in]      given_frame     Given frame
            @param[in]      number_people   Score of people
            @return                         Image
        """
        # Store pixels values of image
        overlay = given_frame.copy()
        
        # Definition of text
        text  =  'People : ' + str(score_people)

        # Adding text with transparent rectangle
        overlay = ps.putBText(overlay, text, text_offset_x = int(overlay.shape[1] / 2), text_offset_y = int(overlay.shape[0] / 2), vspace = 10, hspace = 10, font_scale = 1.0, background_RGB = (0, 0, 0), text_RGB = (255, 255, 0))

        # Drawing line
        overlay = cv2.line(overlay, (int(overlay.shape[1] / 2) + int(overlay.shape[0] / 4), int(overlay.shape[0] / 2.0125)), (int(overlay.shape[0] / 2) + int(overlay.shape[1] / 5.35), int(overlay.shape[0] / 2.0125)), (0, 255, 255), 1)

        # Returns image
        return overlay

    @staticmethod
    def hasChanged(filename = str()):
        """!
            @fn     hasChanged
            @brief  Returns True if given filename have been changed, false otherwise

            @param[in]      filename        Given filename
            @return                         True if given filename have been changed, false otherwise
        """
        try:
            # Read the pickled representation of an object from the open file object given in the constructor
            l = pc.load(open(file = 'datasets/saved_checksum.p', mode = 'rb'))
            pass
        except IOError as err:
            l = []
            pass

        # Load the dictionnary
        database = dict(l)

        # Generate MD5 checksum
        checksum = md5(open(file = filename).read().encode('utf-8')).hexdigest()

        # Check if database contains a checksum
        if database.get(filename, None) != checksum:
            # Load checksum
            database[filename] = checksum

            # Write the pickled representation of the object
            pc.dump(database, open(file = 'datasets/saved_checksum.p', mode = 'wb'))

            # Returns True while filename have been changed
            return True
        else:
            # Returns False while filename have not been changed
            return False