#!/usr/bin/env python3
import re
import warnings
import numpy as np 

def ReadUserInput(name: str) -> set:
    '''
    This function reads the input `name` and generates a set of data that will be used by the code.

    Parameters
    ----------
    `name`: str
        file name that will be read and converted into a data set.
    
    Returns
    -------
    `IN`: set
        data set. 

    '''
    IN = {}

    infile = open(name, 'r')
    
    for line in infile:
      words = re.split('=| |\n|,|[|]',line)
      
      if not any(words[0] in s for s in ['\n', '%', ' ', '#']):
        
        words = list(filter(None, words))

        for i in range(0, len(words)):
            try:
                words[i] = float(words[i])
            except:
                words[i] = words[i]
      
        if len(words[1::1]) == 1 and isinstance(words[1], float):
            IN[words[0]] = [words[1]]
        elif len(words[1::1]) == 1 and isinstance(words[1], str):
            IN[words[0]] = words[1]
        else:
            IN[words[0]] = words[1::1]
    
    IN['Config_Path'] = name
    
    infile.close()

    return IN

def dataAllocation(
        x:      np.ndarray = None, 
        Nsuct:  int        = None, 
        Npress: int        = None
    ) -> list:
    '''
    This function reads the data from the optimization algorithm stored in the `x` array.

    Parameters
    ----------
    `x`: np.array
        array that stores the blade geometry properties.
    `Nsuct`: int 
        variable that defines the degree of freedom of the blade suction side.
    `Npress`: int 
        variable that defines the degree of freedom of the blade pressure side.
    
    Returns
    -------
    `staggerAngle`: float
        blade stagger angle.
    `metalInAngle`: float 
        blade metal inlet angle.
    `metalOutAngle`: float 
        blade metal outlet angle.
    `A0`: float 
        Kulfan parameter that defines the blad leading edge. It is the same for the pressure and suction side of the blade.
    `Asuct`: np.array
        array that stores the parameters that define the blade suction side.
    `Apress`: np.array
        array that stores the parameters that define the blade pressure side.
    `wedgeAngle`: float 
        trailing edge wedge angle. 
    '''

    if len(x) != (6 + Nsuct + Npress):
        warnings.warn('Warning on the data input size ({0:d})'.format(6 + Nsuct + Npress))

    # data allocation
    staggerAngle  = x[0]
    metalInAngle  = x[1]
    metalOutAngle = x[2]
    A0            = x[3]
    
    # profile properties allocation 
    Asuct  = np.zeros((Nsuct,))
    Apress = np.zeros((Npress,))
    
    # suction side allocation
    for ii in range(Nsuct):
        Asuct[ii] = x[4+ii]
    
    # pressure side allocation
    for ii in range(Npress):
        Apress[ii] = x[4+Nsuct+ii]

    # wedge angle 
    wedgeAngle = x[-2]

    # pitch 
    pitch = x[-1]

    return staggerAngle, metalInAngle, metalOutAngle, A0, Asuct, Apress, wedgeAngle, pitch
