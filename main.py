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
    @file       main.py
    @brief      Facial Recognition Algorithm Using Deep Learning
    @details    Declaration of starting point of this program
    
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

import sys, glob
from art import tprint
from deepmwoo.core.access import argv, os, re
from deepmwoo.core.core import hub
from deepmwoo.core.net import net
from getopt import getopt, GetoptError

class mwoo(object):
    """!
        @class      mwoo
        @brief      Define useful static methods to run `deepmwoo`.
    """

    @staticmethod
    def __version__():
        """!
            @fn     __version__
            @brief  Display information about `deepmwoo` release.
        """
        tprint('DeepMwoo', font = 'bulbhead')
        print('Version 0.0.1')
        print('License GPLv3+ : GNU GPL version 3 or later')
        print('Licencied Material - Property of Stimul’Activ®')
        print('© 2020 ENSISA (UHA) - All rights reserved.')

    @staticmethod
    def __usage__():
        """!
            @fn     __usage__
            @brief  Display most of command line options that you can use
                    with `deepmwoo`.
        """
        if sys.platform in ('win32', 'win64'):
            pass
        else:
            os.system('clear')
            os.system('groff -Tascii -man deepmwoo.man')
        
    @staticmethod
    def main():
        """!
            @fn     main
            @brief  Parse and interpret options.
        """
        try:
            opts, args = getopt(sys.argv[1:], 'c:d:hi:m:uvr:', [ 'compile=', 'device=', 'help', 'image=', 'media=', 'url', 'version', 'rescale=' ])
        except GetoptError as err:
            print(err)

            # Unsucessful termination occurs when parsing command-line options
            sys.exit(2)

        for o, a in opts:
            if o in ('-c', '--compile'):
                # Check if variable is an integer
                if isinstance(int(a), int):
                    # Compute transfer learning process
                    net.computation(int(a))
                else:
                     # Built-in assert statement to find errors
                    assert False, 'Invalid argument'
            elif o in ('-d', '--device'):
                # Check if given argument is a valid device
                if argv.is_device(given_argv = a):
                    # Built-in tracking
                    hub.VideoTracking(video_source = int(re.findall(r'\d+', a)[0]))
                else:
                    # Built-in assert statement to find errors
                    assert False, 'Invalid argument'
            elif o in ('-h', '--help'):
                mwoo.__usage__()
            elif o in ('-i', '--image'):
                # Check if given argument is a valid readable image
                if argv.is_image(given_argv = a):
                    # Built-in tracking
                    hub.ImageTracking(image_source = a)
                else:
                    # Built-in assert statement to find errors
                    assert False, 'Invalid argument'
            elif o in ('-m', '--media'):
                # Check if given argument is a valid readable video
                if argv.is_video(given_argv = a):
                    # Built-in tracking
                    hub.VideoTracking(video_source = a)
                else:
                    # Built-in assert statement to find errors
                    assert False, 'Invalid argument'
            elif o in ('-u', '--url'):
                pass
            elif o in ('-v', '--version'):
                mwoo.__version__()
            elif o in ('-r', '--rescale'):
                # Check if given argument is a valid dataset directory
                if os.path.isdir('datasets/' + a):
                    root_dirs = glob.glob('datasets/' + a + '/*/')
                    for root_dir in root_dirs:
                        # Rescaling given dataset directory
                        net.rescale_datasets(root_dir = root_dir)
                else:
                    # Built-in assert statement to find errors
                    assert False, 'Invalid argument'
            else:
                # Built-in assert statement to find errors
                assert False, 'Unhandled option'

        # No problems occured (successful termination)
        sys.exit(0)

if __name__ == '__main__':
    mwoo.main()