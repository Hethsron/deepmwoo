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

import sys, os
from art import tprint
from getopt import getopt, GetoptError

class Main(object):
    """!
        @class      Main
        @brief      Define useful static method to run `deepmwoo`
    """

    @staticmethod
    def version():
        """!
            @fn     version
            @brief  Display information about `deepmwoo` release
        """
        tprint('DeepMwoo', font='bulbhead')
        print('Version 0.0.1')
        print('License GPLv3+ : GNU GPL version 3 or later')
        print('Licencied Material - Property of Stimul’Activ®')
        print('© 2020 ENSISA (UHA) - All rights reserved.')

    @staticmethod
    def usage():
        """!
            @fn     usage
            @brief  Display most of command line options that you can use
                    with `deepmwoo`
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
            opts, args = getopt(sys.argv[1:], 'dhipuv', ['device', 'help', 'image', 'player', 'url', 'version'])
        except GetoptError as err:
            print(err)

            # Unsucessful termination occurs when parsing command-line options
            sys.exit(2)

        for o, a in opts:
            if o in ('-d', '--device'):
                pass
            elif o in ('-h', '--help'):
                Main.usage()
            elif o in ('-i', '--image'):
                pass
            elif o in ('-p', '--player'):
                pass
            elif o in ('-u', '--url'):
                pass
            elif o in ('-v', '--version'):
                Main.version()
            else:
                assert False, 'Unhandled option'

        # No problems occured (successful termination)
        sys.exit(0)

if __name__ == '__main__':
    Main.main()