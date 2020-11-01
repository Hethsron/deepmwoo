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
    @file       shapes.py
    @brief      Basic Geometric Shapes For Facial Recognition Algorithm
    @details    Declaration of basic methods used to draw shapes on given image
    
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
from enum import IntEnum, unique

@unique
class cv4(IntEnum):
    """!
        @enum       cv4
        @brief      Define useful static methods to draw geometric shapes on any image
    """

    RADIUS_AA = 1
    RADIUS_BB = 2
    RADIUS_CC = 3
    RADIUS_DD = 4
    RADIUS_EE = 5
    RADIUS_FF = 6
    RADIUS_GG = 7
    RADIUS_HH = 8
    RADIUS_II = 9
    RADIUS_JJ = 10

    @staticmethod
    def rectangle(src, top_left, bottom_right, color=255, thickness=1, line_type=cv2.LINE_AA, corner_radius = RADIUS_JJ):
        """!
            @fn     rectangle
            @brief  Draw a rectangle with rounded corners on any image.

            @param[in]      src             It is the image on which rectangle is to be drawn.
            @param[in]      top_left        It is the starting coordinates of rectangle. The
                                            coordinates are represented as tuple of two values
                                            i.e (X coordinate value, Y coordinate value).
            @param[in]      bottom_right    It is the ending coordinate of rectangle. The
                                            coordinates are represented as tuple of two values
                                            i.e (X coordinate value, Y coordinate value).
            @param[in]      color           It is the color of border line of rectangle to be
                                            drawn. For BGR, we pass a tuple. eg (255, 0, 0)
                                            for blue color.
            @param[in]      thickness       It is the thickness of the rectangle border line 
                                            in pixel (px). Thickness of -1px will fill the
                                            rectangle shape by the specified color.
            @param[in]      line_type       It denote the type of the line for drawing.
            @param[in]      corner_radius   It denote the rounded corner radius of rectangle.

            @return         
        """

        #  Define corners:
        #  p1 - p2
        #  |     |q
        #  p4 - p3

        p1 = top_left
        p2 = (bottom_right[0], top_left[1])
        p3 = bottom_right
        p4 = (top_left[0], bottom_right[1])

        # Draw straight lines
        cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
        cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
        cv2.line(src, (p4[0] + corner_radius, p4[1]), (p3[0] - corner_radius, p3[1]), color, abs(thickness), line_type)
        cv2.line(src, (p1[0], p1[1] + corner_radius), (p4[0], p4[1] - corner_radius), color, abs(thickness), line_type)

        # Draw arcs
        cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
        cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
        cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
        cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

        # Return an image
        return src