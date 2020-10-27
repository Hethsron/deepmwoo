#!/usr/bin/env python3

"""!
    @file       main.py
    @brief      Facial Recognition Algorithm Using Deep Learning
    @details    Declaration of starting point of this program
    
    @author     BOUEYA Hethsron Jedaël <hethsron-jedael.boueya@uha.fr>
                BENOMAR Yassine <yassine.benomar@uha.fr>
    
    @version    0.1
    @date       October, 23th 2020
    
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
                Licencied Material - Property of STIMULACTIV
                © 2020 ENSISA (UHA) - All rights reserved.
"""

import sys
from getopt import getopt, GetoptError

def main():
    """!
        @fn     main
        @brief  Parse and interpret options.
    """
    try:
        opts, args = getopt(sys.argv[1:], "ht", ["help", "train"])
    except GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            print("Help")
            sys.exit()
        elif o in ("-t", "--train"):
            print("Train")
            sys.exit()
        else:
            assert False, "Unhandled option"


if __name__ == '__main__':
    main()