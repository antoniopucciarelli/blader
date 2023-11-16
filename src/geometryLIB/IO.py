#!/usr/bin/env python3
import numpy              as     np 
from   geometryLIB.reader import ReadUserInput

def readData(fileName: str) -> list:
    '''
    This function reads the configuration data from `filename` and then generated the profile properties for the camberline, suction side surface and pressure side surface.
    
    Parameters
    ----------
    `filename`: str
        file name where profile data properties are stored. 

    Returns
    -------
    `staggerAngle`: float
        stagger angle. 
    `metalInAngle`: float
        inlet metal angle. 
    `metalOutAngle`: float 
        outlet metal angle.
    `pitch`: float 
        blade cascade pitch. 
    `LEradius`: float 
        leading edge radius.
    `TEradius`: float 
        traling ege radius. 
    `Nsuct`: int
        discretization parameter for the suction side. 
    `Asuction`: np.array
        array that stores all the parametrization variables used for the suction side.
    `Npress`: int 
        discretization paramter for the pressure side. 
    `Apressure`: np.array
        array that stores all the parametrization variables used for the pressure side.
    '''
    caseInput = ReadUserInput(fileName)

    pitch         = caseInput["Pitch"][0]
    staggerAngle  = caseInput["Stagger"][0]
    metalInAngle  = caseInput["MetalInAngle"][0]
    metalOutAngle = caseInput["MetalOutAngle"][0]  
    LEradius      = caseInput["LEradius"][0]    
    TEradius      = caseInput["TEradius"][0]       
    Nsuct         = np.int64(caseInput["Nsuct"][0])
    Npress        = np.int64(caseInput["Npress"][0])

    A0 = np.sqrt(2 * LEradius)

    Asuction = np.zeros((Nsuct+1,))
    Asuction[0] = A0 
    for ii in range(1, Nsuct+1):
        Asuction[ii] = caseInput["A{0:d}suct".format(ii)][0]
    
    Apressure = np.zeros((Npress+1,))
    Apressure[0] = A0
    for ii in range(1, Npress+1):
        Apressure[ii] = caseInput["A{0:d}press".format(ii)][0]

    return staggerAngle, metalInAngle, metalOutAngle, pitch, LEradius, TEradius, Nsuct, Asuction, Npress, Apressure

def readCoordinates(fileName: str) -> list:
    '''
    This function reads the coordinates from `filename`. 
    
    Parameters
    ----------
    `filename`: str 
        input file name where all the variables are stored. `filename` has to have `.total.airf` extension.
    
    Returns
    -------
    `SScoords`: list 
        array that stores all the suction side coordinates (`x`, `y`).
    `PScoords`: list 
        array that stores all the pressure side coordinates (`x`, `y`). 
    '''
    with open(fileName + '.total.airf', 'r') as f: 
        f.readline()
        lines = f.readlines()

        SScoords = [] 
        PScoords = [] 
        for line in lines:
            coords = line.split()
            try:
                SScoords.append(np.array([np.float64(coords[2]), np.float64(coords[3])]))
            except: 
                pass 
            try:
                PScoords.append(np.array([np.float64(coords[4]), np.float64(coords[5])]))
            except:
                pass 

    return SScoords, PScoords

def saveData(
        fileName:      str, 
        staggerAngle:  float, 
        metalInAngle:  float, 
        metalOutAngle: float, 
        pitch:         float, 
        LEradius:      float, 
        TEradius:      float, 
        Nsuction:      int, 
        Asuction:      float, 
        Npressure:     int, 
        Apressure:     float
    ) -> None:
    '''
    This function saves all the parametrization data into a a configuration file with `.cfg` extension.

    Parameters
    ----------
    `filename`: str
        file name where profile data properties are stored. 
    `staggerAngle`: float
        stagger angle. 
    `metalInAngle`: float
        inlet metal angle. 
    `metalOutAngle`: float 
        outlet metal angle.
    `pitch`: float 
        blade cascade pitch. 
    `LEradius`: float 
        leading edge radius.
    `TEradius`: float 
        traling ege radius. 
    `Nsuct`: int
        discretization parameter for the suction side. 
    `Asuction`: np.array
        array that stores all the parametrization variables used for the suction side.
    `Npress`: int 
        discretization paramter for the pressure side. 
    `Apressure`: np.array
        array that stores all the parametrization variables used for the pressure side.
    '''
    with open(fileName, 'w') as f:
        f.write('Pitch         = {0:f}\n'.format(pitch)        )
        f.write('Stagger       = {0:f}\n'.format(staggerAngle) )
        f.write('MetalInAngle  = {0:f}\n'.format(metalInAngle) )
        f.write('MetalOutAngle = {0:f}\n'.format(metalOutAngle))
        f.write('LEradius      = {0:f}\n'.format(LEradius)     )
        f.write('TEradius      = {0:f}\n'.format(TEradius)     )

        f.write('Nsuct         = {0:d}\n'.format(Nsuction)     )
        for ii in range(1, Nsuction+1):
            f.write('A{0:d}suct        = {1:f}\n'.format(ii, Asuction[ii]))
        
        f.write('Npress        = {0:d}\n'.format(Npressure)    )
        for ii in range(1, Npressure+1):
            f.write('A{0:d}press       = {1:f}\n'.format(ii, Apressure[ii]))

def saveAirfoil(
        fileName:     str, 
        camberline:   np.array, 
        suctionline:  np.array, 
        pressureline: np.array
    ) -> None:
    '''
    This function saves the airfoil coordinates in a `.total.airf` extension file.

    Paramters
    ---------
    `filename`: str
        file name where profile coordinates will be stored. 
    `cambeline`: kulfanLIB.camberline.camberline
        camberline kulfanLIB.camberline.camberline
    `suctionline`: kulfanLIB.profileLine.profileLine
        suction side kulfanLIB.profileLine.profileLine object.
    `pressureline`: kulfanLIB.profileLine.profileLine 
        pressure side kulfanLIB.profileLine.profileLine object.
    '''
    lenCamberline   = len(camberline.x)
    lenSuctionLine  = len(suctionline.X)
    lenPressureLine = len(pressureline.X)
    
    maxVal = max([lenCamberline, lenSuctionLine, lenPressureLine])

    with open(fileName + '.total.airf', 'w') as f:
        f.write('# {0:s} -- Xc = X/Cx | Yc = Y/Cx | Xss = X/Xc | Yss = Y/Xc | Xps = X/Xc | Yps = Y/Xc\n'.format(fileName))
        for ii in range(maxVal):    
            if ii >= lenCamberline:
                camberValx = ' '
                camberValy = ' '
            else:
                camberValx = camberline.x[ii]
                camberValy = camberline.y[ii]
            if ii >= lenSuctionLine:
                suctValx = ' '
                suctValy = ' '
            else:
                suctValx = suctionline.X[ii]
                suctValy = suctionline.Y[ii]
            if ii >= lenPressureLine:
                pressValx = ' '
                pressValy = ' '
            else:
                pressValx = pressureline.X[ii]
                pressValy = pressureline.Y[ii]

            f.write(f"{'{0} '.format(camberValx) : <30}")
            f.write(f"{'{0} '.format(camberValy) : <30}")
            f.write(f"{'{0} '.format(suctValx)   : <30}")
            f.write(f"{'{0} '.format(suctValy)   : <30}")
            f.write(f"{'{0} '.format(pressValx)  : <30}")
            f.write(f"{'{0} '.format(pressValy)}")
            f.write('\n')
