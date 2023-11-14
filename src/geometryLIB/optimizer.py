import numpy                      as     np 
import matplotlib.pyplot          as     plt 
from   scipy                      import misc
from   scipy                      import integrate
from   scipy                      import interpolate
from   scipy                      import optimize
from   scipy.optimize             import Bounds    
from   src.geometryLIB.blade      import Blade 
from   src.geometryLIB.camberline import Camberline

def bladeInOrigin(data: list | np.ndarray) -> np.ndarray:
    '''
    This function displaces the blade data into the origin.
    '''
    
    # getting minimum position 
    minPos = np.argmin(data[:, 0])
    minX   = data[minPos, 0]
    minXy  = data[minPos, 1]

    # updating data -> displacement into origin (0, 0)
    data[:,0] = data[:,0] - minX 
    data[:,1] = data[:,1] - minXy

    # converting data into numpy array object 
    data = np.array(data)

    return data

def bladeTEradius(data: list | np.ndarray, chord: float = 1) -> float:
    '''
    Blade traling edge computation and normalization with respect to the blade axial chord.
    '''

    # trailing edge computation 
    TEradius = np.linalg.norm(data[0,:] - data[-1,:]) / chord
    print('>>> TRAILING EDGE RADIUS = {0:.2E}'.format(TEradius))

    return TEradius

def bladeLEpos(upperLine: interpolate.interp1d, lowerLine: interpolate.interp1d, maxX: float = 0.1, nPoints: int = 30, plot: bool = True) -> tuple[int, float]:
    
    # setting up delta x 
    dx = 1E-3

    # setting up study vector
    x = np.linspace(dx, maxX, nPoints)

    # computing 1st derivatives
    upperLineDer = misc.derivative(func=upperLine, x0=x, n=1, dx=dx)
    lowerLineDer = misc.derivative(func=lowerLine, x0=x, n=1, dx=dx)

    # computing 2nd derivatives
    upperLineDerDer = misc.derivative(func=upperLine, x0=x, n=2, dx=dx)
    lowerLineDerDer = misc.derivative(func=lowerLine, x0=x, n=2, dx=dx)

    # curvature computation
    upperCurvature = np.abs(upperLineDerDer) / (1 + upperLineDer**2)**(3/2)
    lowerCurvature = np.abs(lowerLineDerDer) / (1 + lowerLineDer**2)**(3/2)

    # getting max curvature position
    if max(upperCurvature) > max(lowerCurvature):
        maxPos = np.argmax(upperCurvature)
        position = True
    else:
        maxPos = np.argmax(lowerCurvature)
        position = False

    # getting axial chord max curvature position
    xCurvature = x[maxPos]
    if position:
        yCurvature = upperLine(xCurvature)
        yPrime     = misc.derivative(func=upperLine, x0=xCurvature, n=1, dx=dx)
        y2Prime    = misc.derivative(func=upperLine, x0=xCurvature, n=2, dx=dx)
    else:
        yCurvature = lowerLine(xCurvature)
        yPrime     = misc.derivative(func=lowerLine, x0=xCurvature, n=1, dx=dx)
        y2Prime    = misc.derivative(func=lowerLine, x0=xCurvature, n=2, dx=dx)

    if plot:
        # plotting data
        fig = plt.figure()
        ax = fig.add_subplot(1,4,1)

        ax.plot(x, upperLine(x), 'r')
        ax.plot(x, lowerLine(x), 'b')
        ax.plot(xCurvature, yCurvature, 'ok')

        ax.set_title('Camberline')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(visible=True, linestyle='dotted')

        ax1 = fig.add_subplot(1,4,2)

        ax1.plot(x, upperLineDer, 'r')
        ax1.plot(x, lowerLineDer, 'b')
        ax1.plot(xCurvature, yPrime, 'ok')
        
        # ax1.set_aspect('equal')
        ax1.set_title(r'$y^{\prime}$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(visible=True, linestyle='dotted')

        ax2 = fig.add_subplot(1,4,3)

        ax2.plot(x, upperLineDerDer, 'r')
        ax2.plot(x, lowerLineDerDer, 'b')
        ax2.plot(xCurvature, y2Prime, 'ok')

        # ax2.set_aspect('equal')
        ax2.set_title(r'$y^{\prime \prime}$')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(visible=True, linestyle='dotted')

        ax3 = fig.add_subplot(1,4,4)

        ax3.plot(x, upperCurvature, 'r')
        ax3.plot(x, lowerCurvature, 'b')
        
        # ax2.set_aspect('equal')
        ax3.set_title(r'$\kappa$')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.grid(visible=True, linestyle='dotted')

        plt.tight_layout()
        plt.show()

def rotate(data: list | np.ndarray, theta: float) -> np.ndarray: 
    '''
    This function rotates data with respect to the origin with a theta angle (in degrees).

    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the blade coordinate points
    `theta`: float 
        number which defines the rotation angle. The angle must be expressed in degrees

    Returns
    -------
    `data`: list | np.ndarray   
        2D vector which stores the rotated blade coordinate points  
    '''
    
    # main rotation matrix parameters
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))
    
    # rotation matrix generation
    rotMatrix = np.array([[cos, -sin], [sin, cos]])

    # rotating data 
    for ii, coord in enumerate(data):
        coord = np.matmul(rotMatrix, coord) 
        data[ii, :] = coord / cos

    return data

def plotCoords(data: list | np.ndarray, ax: plt.Axes = None, theta: float = 0.0, flip: bool = False, base: bool = True) -> None:
    '''
    This function plots the coordinate based data of the blade geometry.
    '''

    # rotating data
    if theta != 0:  
        # main data computation
        cos = np.cos(np.deg2rad(theta))
        sin = np.sin(np.deg2rad(theta))
        # rotation matrix assembly
        rotMatrix = np.array([[cos, -sin], [sin, cos]])
        # rotating data 
        for ii, coord in enumerate(data):
            coord = np.matmul(rotMatrix, coord) 
            data[ii, :] = coord / cos
    
    # axes generation
    if not isinstance(ax, (plt.Axes)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    # getting lower side and upper side of the blade geometry 
    _, _, upperPart, lowerPart, upperChord, lowerChord = interpolateData(data=data, flip=flip)

    # plotting data
    if base:
        ax.plot(data[:,0], data[:,1],                                 color='c', linestyle='solid',  linewidth=2.5, label='DATA.FULL')
    ax.plot(upperPart[:,0] * upperChord, upperPart[:,1] * upperChord, color='r', linestyle='dashed', linewidth=2.5, label='DATA.UPPER-SIDE')
    ax.plot(lowerPart[:,0] * lowerChord, lowerPart[:,1] * lowerChord, color='b', linestyle='dashed', linewidth=2.5, label='DATA.LOWER-SIDE')
    
    # axes decoration
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.grid(visible=True, linestyle='dotted')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()

def interpolateData(data: list | np.ndarray, plot: bool = False) -> tuple:
    '''
    This function interpolates the geometry data from the target blade dataset.

    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the target blade geometry in [x, y] coordinates.
    `plot`: bool
        boolean value for the plotting of the operation results.

    Returns
    -------
    `upperLine`: interpolate.interp1d 
        function object which parametrizes the upper part of the blade. The upper part of the blade starts from the most left point of the geometry to the most right point of the geometry.
    `lowerLine`: interpolate.interp1d 
        function object which parametrizes the lower part of the blade. The lower part of the blade starts from the most left point of the geometry to the most right point of the geometry.
    `upperPart`: np.ndarray
        2D vector which stores the upper part coordinates of the blade. 
    `lowerPart`: np.ndarray
        2D vector which stores the lower part coordinates of the blade.
    '''

    # getting minimum position 
    minPos = np.argmin(data[:, 0])
    minX   = data[minPos, 0]
    minXy  = data[minPos, 1]

    # # updating data
    data[:,0] = data[:,0] - minX 
    data[:,1] = data[:,1] - minXy

    # setting up data
    upperPart = np.array(data[0:minPos+1, :])
    lowerPart = np.array(data[minPos::, :])

    # getting blade chord
    upperChord = np.max(upperPart[:, 0]) - np.min(upperPart[:, 0])
    lowerChord = np.max(lowerPart[:, 0]) - np.min(lowerPart[:, 0])
    
    # normalize data 
    upperPart = upperPart / upperChord 
    lowerPart = lowerPart / lowerChord 

    # interpolating data 
    upperLine = interpolate.interp1d(upperPart[:, 0], upperPart[:, 1])
    lowerLine = interpolate.interp1d(lowerPart[:, 0], lowerPart[:, 1])

    # testing data 
    xVec = np.linspace(0.01, 0.99, 100)

    for x in xVec:
        if upperLine(x) < lowerLine(x):
            # checking data properties
            # plotting data error
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            
            ax.plot(upperPart[:,0], upperPart[:,1], 'r', linewidth=2, label='UPPER-SIDE')
            ax.plot(lowerPart[:,0], lowerPart[:,1], 'b', linewidth=2, label='LOWER-SIDE')

            ax.plot([x], [upperLine(x)], 'or', label='UPPER-SIDE-ERROR_POINT')
            ax.plot([x], [lowerLine(x)], 'ob', label='LOWER-SIDE-ERROR_POINT')
 
            ax.set_aspect('equal')
            ax.grid(visible=True, linestyle='dotted')
            ax.legend(bbox_to_anchor=(1,1), loc="upper left")
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            plt.tight_layout()
            plt.show()

            print('x = {0:f}'.format(x))
            print(upperLine(x), lowerLine(x))

            raise ValueError('::: Wrong interpolation, something is wrong with the input geometry (`data`).')

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.plot(upperPart[:,0], upperPart[:,1], 'r', linewidth=2, label='UPPER-SIDE')
        ax.plot(lowerPart[:,0], lowerPart[:,1], 'b', linewidth=2, label='LOWER-SIDE')

        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.legend(bbox_to_anchor=(1,1), loc="upper left")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('BLADE SPLITTING (UPPER\LOWER SIDE)')

        plt.tight_layout()
        plt.show()

    return upperLine, lowerLine, upperPart, lowerPart, upperChord, lowerChord

def dataExtraction(
        x:           list | np.ndarray, 
        Nsuct:       int, 
        Npress:      int
    ) -> tuple:

    # initializing data
    stagger    = x[0]
    metalIn    = x[1]
    metalOut   = x[2]
    LEradius   = x[3] 
    wedgeAngle = x[-1]
    
    # getting the rest of the data
    Asuct  = x[4:3+Nsuct] 
    Apress = x[3+Nsuct:2+Nsuct+Npress]

    # print('Asuct  = ', Asuct)
    # print('Apress = ', Apress)
    # print(len(Asuct))
    # print(len(Apress))

    return stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle

def bladeDataExtraction(
        x:        list | np.ndarray,
        Nsuct:    int   = None, 
        Npress:   int   = None,
        blade:    Blade = None,
        TEradius: float = 0.0,
    ) -> float:

    # getting data 
    if x is not None:
        stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle = dataExtraction(x=x, Nsuct=Nsuct, Npress=Npress)
        # blade generation 
        blade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

    # coordinates extraction
    XPS, YPS, XSS, YSS, XCL, YCL = blade.coordinate()

    # inverting coordinates 
    XPS = np.flip(XPS)
    YPS = np.flip(YPS) 

    # concatenating data 
    X = np.concatenate((XPS, XSS))
    Y = np.concatenate((YPS, YSS))

    # getting minimum position 
    minX   = np.min(X)
    minPos = np.argmin(X)
    minXy  = Y[minPos]

    # updating data
    X   = X   - minX 
    Y   = Y   - minXy
    XCL = XCL - minX 
    YCL = YCL - minXy

    # merging data 
    bladeData      = np.stack((X, Y), axis=1)
    camberlineData = np.stack((XCL, YCL), axis=1)

    # getting upper part of the blade 
    bladeUpper = bladeData[0:minPos+1, :]
    bladeLower = bladeData[minPos::, :]

    # updating real chord 
    upperChord      = np.max(bladeUpper[:, 0]) - np.min(bladeUpper[:, 0])
    lowerChord      = np.max(bladeLower[:, 0]) - np.min(bladeLower[:, 0])
    camberlineChord = np.max(camberlineData[:,0]) - np.min(camberlineData[:,0])

    # normalizing data 
    bladeUpper     = bladeUpper / upperChord
    bladeLower     = bladeLower / lowerChord
    camberlineData = camberlineData / camberlineChord

    # linear data interpolation 
    bladeUpperLine = interpolate.interp1d(x=bladeUpper[:, 0], y=bladeUpper[:, 1])
    bladeLowerLine = interpolate.interp1d(x=bladeLower[:, 0], y=bladeLower[:, 1])
    camberlineLine = interpolate.interp1d(x=camberlineData[:, 0], y=camberlineData[:, 1])

    return bladeUpperLine, bladeLowerLine, camberlineLine, upperChord, bladeData, camberlineData, lowerChord, camberlineChord

def camberlineFunc(
        x:           list | np.ndarray, 
        yCamberline: list | np.ndarray, 
        nPoints:     int
    ) -> float: 
    '''
    This function computes the error between a set of coordinates and a parametrized camberline.
    '''

    # camberline object generation
    try:
        cLine = Camberline(stagger=x[0], metalIn=x[1], metalOut=x[2], chebyschev=False, nPoints=nPoints, origin=True)

        # computing root mean squared error
        RMSE = 0.0 

        for ii in range(len(yCamberline)):
            RMSE = RMSE + (yCamberline[ii] - cLine.y[ii])**2

        RMSE = np.sqrt(RMSE) / len(yCamberline)
    except:
        RMSE = np.NAN

    print('>>> RMSE = {0:.2E}'.format(RMSE))

    return RMSE

def bladeFunc(
        x:           list | np.ndarray,
        Nsuct:       int, 
        Npress:      int,
        upperLine:   interpolate.interp1d,
        lowerLine:   interpolate.interp1d,
        TEradius:    float,
        nPoints:     int  = 200
    ) -> float:

    # getting data 
    stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle = dataExtraction(x=x, Nsuct=Nsuct, Npress=Npress)
    
    # blade generation 
    try:
        blade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

        # plotting blade
        # blade.plot()

        # exit()

        # getting blade coordinates with the most left point as origin
        XPS, YPS, XSS, YSS, _, _ = blade.coordinate()

        # inverting coordinates 
        XPS = np.flip(XPS)
        YPS = np.flip(YPS) 

        # concatenating data 
        X = np.concatenate((XPS, XSS))
        Y = np.concatenate((YPS, YSS))

        # getting minimum position 
        minX   = np.min(X)
        minPos = np.argmin(X)
        minXy  = Y[minPos]

        # updating data
        X = X - minX 
        Y = Y - minXy

        # merging data 
        data = np.stack((X, Y), axis=1)

        # getting upper part of the blade 
        bladeUpper = data[0:minPos+1, :]
        bladeLower = data[minPos::, :]

        # updating real chord 
        upperChord = np.max(bladeUpper[:, 0]) - np.min(bladeUpper[:, 0])
        lowerChord = np.max(bladeLower[:, 0]) - np.min(bladeLower[:, 0])

        # normalizing data 
        bladeUpper = bladeUpper / upperChord
        bladeLower = bladeLower / lowerChord

        # linear data interpolation 
        bladeUpperLine = interpolate.interp1d(x=bladeUpper[:, 0], y=bladeUpper[:, 1])
        bladeLowerLine = interpolate.interp1d(x=bladeLower[:, 0], y=bladeLower[:, 1])

        # computing study points
        # x = np.linspace(0.01, 0.99, nPoints)
        x = [(1 - np.cos((2 * (ii + 1) - 1) / (2 * nPoints) * np.pi)) / 2  for ii in range(nPoints)]
        # adding extremes
        x = np.concatenate(([0], x, [1]))

        # evaluating blade data
        bladeUpperY = bladeUpperLine(x)
        bladeLowerY = bladeLowerLine(x)

        # evaluating target data 
        targetUpperY = upperLine(x)
        targetLowerY = lowerLine(x)

        # computing root mean squared error 
        RMSEupper = 0.0 
        for ii in range(nPoints):
            RMSEupper = RMSEupper + (bladeUpperY[ii] - targetUpperY[ii])**2

        RMSElower = 0.0 
        for ii in range(nPoints):
            RMSElower = RMSElower + (bladeLowerY[ii] - targetLowerY[ii])**2 

        # computing total RMSE
        RMSE = np.sqrt(RMSEupper + RMSElower) / (2 * nPoints)

    except:
        RMSE = np.NAN

    print("RMSE = {0:.3E}".format(RMSE))

    return RMSE

def boundsGenerator(
        stagger:             int | float, 
        metalInlet:          int | float,
        metalOutlet:         int | float,
        Nsuct:               int, 
        Npress:              int,
        staggerInterval:     float | int = 1e-1,
        metalInletInterval:  float | int = 3, 
        metalOutletInterval: float | int = 1e-2, 
        metalInBound:        list        = [-60, -1E-1],
        metalOutBound:       list        = [1E-1, 80],
        Abounds:             list        = [-1E-1, 1.4],
        LEradiusBounds:      list        = [1E-2, 0.3],
        wedgeAngleMax:       int | float = 40
    ) -> Bounds:
    '''
    This function generates the boundary values for the blade geometry optimizer.

    Parameters
    ----------
    `stagger`: float | int 
        value for the stagger angle.
    `Nsuct`: int 
        Kulfan parametrization for the suction side of the blade.
    `Npress`: int 
        Kulfan parametrization for the pressure side of the blade.

    Returns
    -------
    `bounds`: Bounds
        boundary object for the scipy optimization.
    '''

    # boundaries generation 
    lowerBounds = [] 
    upperBounds = [] 

    # stagger angle
    lowerBounds.append(max(stagger - staggerInterval, 0))
    upperBounds.append(stagger + staggerInterval)
    
    # metal inlet angle
    lowerBounds.append(metalInlet - metalInletInterval)
    upperBounds.append(min(metalInlet + metalInletInterval, -1.0))

    # metal outlet angle
    lowerBounds.append(max(metalOutlet - metalOutletInterval, -1.0))
    upperBounds.append(metalOutlet + metalOutletInterval)

    # LEradius bounds
    lowerBounds.append(max(LEradiusBounds[0], 0))
    upperBounds.append(np.abs(LEradiusBounds[1]))

    # suction side bounds
    for _ in range(Nsuct - 1):
        lowerBounds.append(Abounds[0])
        upperBounds.append(Abounds[1])

    # pressure side bounds
    for _ in range(Npress - 1):
        lowerBounds.append(Abounds[0])
        upperBounds.append(Abounds[1])

    # wedge angle bounds
    lowerBounds.append(5)
    upperBounds.append(np.abs(wedgeAngleMax))

    # print(lowerBounds)
    # print(upperBounds)

    # exit()
    
    # scipy bounds object generation 
    bounds = Bounds(lowerBounds, upperBounds)

    return bounds
 
def findCircle(
        point1: np.ndarray, 
        point2: np.ndarray,
        point3: np.ndarray, 
    ) -> float:
    '''
    This function computes the circle radius given 3 passing points. 
    '''

    # data allocation
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x3 = point3[0]
    y3 = point3[1]

    # data computation
    x12 = x1 - x2; 
    x13 = x1 - x3; 
 
    y12 = y1 - y2; 
    y13 = y1 - y3; 
 
    y31 = y3 - y1; 
    y21 = y2 - y1; 
 
    x31 = x3 - x1; 
    x21 = x2 - x1; 
 
    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2); 
 
    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2); 
 
    sx21 = pow(x2, 2) - pow(x1, 2); 
    sy21 = pow(y2, 2) - pow(y1, 2); 
 
    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
          ((y31) * (x12) - (y21) * (x13))));
             
    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
          (2 * ((x31) * (y12) - (x21) * (y13)))); 
 
    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1); 

    h = -g; 
    k = -f; 
    sqr_of_r = h * h + k * k - c; 
 
    # radius computation 
    radius = round(np.sqrt(sqr_of_r), 5); 

    return radius

def computeGuess(data: list | np.ndarray, upperData: list | np.ndarray, lowerData: list | np.ndarray, inletPos: int, outletPos: int) -> tuple:
    # stagger angle computation 
    TEpoint = (data[-1, :] + data[0, :]) / 2
    stagger = np.rad2deg(np.arctan(TEpoint[1] / TEpoint[0]))

    # metal inlet angle computation 
    inletSlopePoint = (upperData[inletPos, :] + lowerData[inletPos, :]) / 2
    metalInlet = np.rad2deg(np.arctan(inletSlopePoint[1] / inletSlopePoint[0]))

    # metal outlet angle computation
    outletSlopePoint = (upperData[- outletPos, :] + lowerData[- outletPos, :]) / 2
    metalOutlet = np.rad2deg(np.arctan((TEpoint[1] - outletSlopePoint[1]) / (TEpoint[0] - outletSlopePoint[0])))

    # leading edge radius approxiamation
    # if LEradius == None:
        # LEradius = findCircle(point1=lowerData[-3, :], point2=[0,0], point3=upperData[2,:])
    
    # wedge angle vector allocation
    upperVec = TEpoint - upperData[1,:]
    lowerVec = TEpoint - lowerData[-2,:]

    # wedge angle computation
    if (data[0, 0] - data[-1, 0] == 0) and (data[0, 1] - data[-1, 1] == 0):
        wedgeAngle = np.dot(upperVec, lowerVec)
        wedgeAngle = wedgeAngle / (np.linalg.norm(upperVec) * np.linalg.norm(lowerVec))
        wedgeAngle = np.rad2deg(np.arccos(wedgeAngle)) / 2
        wedgeAngle = max(1, wedgeAngle)
    else: 
        wedgeAngle = 15 

    print('TE point    = ', TEpoint)
    print('upper data  = ', upperData[1,:])
    print('lower data  = ', lowerData[-2,:])
    print('wedge angle = ', wedgeAngle)

    return stagger, metalInlet, metalOutlet, wedgeAngle

def camberlineAnalysis(data: list | np.ndarray, plot: bool = False, nPoints: int = 100) -> tuple[bool, np.ndarray]:
    '''
    This function checks the camberline curvature and makes changes over the camberline properties. 
    
    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the blade coordinate points
    `plot`: bool 
        boolean value for the plotting of the main operation results
    `nPoints`: int
        integer which defines the number of points for the camberline analysis

    Returns
    -------
    `flip`: bool    
        boolean value which defines if the blade coordinates have been flipped for the blade optimization
    `data`: np.ndarray
        2D vector which stores the rotated coordinates
    '''

    # getting minimum position 
    minPos = np.argmin(data[:, 0])
    minX   = data[minPos, 0]
    minXy  = data[minPos, 1]

    # updating data
    data[:,0] = data[:,0] - minX 
    data[:,1] = data[:,1] - minXy

    # trailing edge angle computation
    xTE = (data[0, 0] + data[-1, 0]) / 2
    yTE = (data[0, 1] + data[-1, 1]) / 2
    theta = np.rad2deg(np.arctan(yTE / xTE))

    # setting up data
    upperPart = np.array(data[minPos::, :])
    lowerPart = np.array(data[0:minPos+1, :])

    # getting blade chord
    upperChord = np.max(upperPart[:, 0]) - np.min(upperPart[:, 0])
    lowerChord = np.max(lowerPart[:, 0]) - np.min(lowerPart[:, 0])
    
    # normalize data 
    upperPart = upperPart / upperChord 
    lowerPart = lowerPart / lowerChord 

    # interpolating data 
    upperLine = interpolate.interp1d(upperPart[:, 0], upperPart[:, 1])
    lowerLine = interpolate.interp1d(lowerPart[:, 0], lowerPart[:, 1])

    x = np.linspace(0, 1, nPoints)
    yUpper = upperLine(x)
    yLower = lowerLine(x)

    # computing camberline 
    yCamberline = (yUpper + yLower) / 2

    # rotating data 
    camberlinePoints = np.stack([x, yCamberline], axis=1)
    studyCamberline = rotate(data=camberlinePoints, theta=-theta)

    # integral computation for flipping  
    integral = integrate.trapezoid(y=studyCamberline[:,1], x=studyCamberline[:,0])

    print('>>> INTEGRAL = {0:+.2E}'.format(integral))

    if integral > 0:
        flip = True 
        data = np.stack([np.flip(data[:,0]), - np.flip(data[:,1])], axis=1)
        
        # reallocating point data 
        yUpper      = - yUpper 
        yLower      = - yLower
        yCamberline = - yCamberline
        
        print('>>> BLADE GEOMETRY IS FLIPPED FOR INTERPOLATION')
    else:
        flip = False
        print('>>> BLADE GEOMETRY IS NOT FLIPPED FOR INTERPOLATION')

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.plot(x, yUpper,      color='r', label='UPPER-SIDE', linewidth=3)
        ax.plot(x, yLower,      color='b', label='LOWER-SIDE', linewidth=3)
        ax.plot(x, yCamberline, color='k', label='CAMBERLINE', linewidth=3)
        ax.plot(studyCamberline[:,0], studyCamberline[:,1], color='k', linestyle='dotted', label='ORIGINAL-CAMBERLINE-ROTATED', linewidth=3)
        
        if flip:
            ax.set_title('BLADE DATA HAS BEEN FLIPPED. INTEGRAL = {0:.2E}'.format(integral))
            ax.plot(data[:,0], data[:,1], color='c', linestyle='-.', label='ROTATED-BLADE')
        else:
            ax.set_title('BLADE DATA HAS NOT BEEN FLIPPED. INTEGRAL = {0:.3E}'.format(integral))
        
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    return flip, data

def optimizeCamberline__(upperLine: list | np.ndarray, lowerLine: list | np.ndarray, nPoints: int = 100, plot: bool = True) -> tuple:
    '''
    This function optimizes find a suitable camberline parametrizatio for the blade.
    '''

    # camberline coordinates computation 
    x = np.linspace(0, 1.0, nPoints)
    
    # isolating data
    yUpper = upperLine(x)
    yLower = lowerLine(x)
    
    # camberline ordinates
    yCamberline = (yUpper + yLower) / 2

    # optimization boundaries generation 
    lowerBounds = [] 
    upperBounds = [] 

    # stagger angle
    lowerBounds.append(0)
    upperBounds.append(60)
    
    # metal inlet angle
    lowerBounds.append(-50)
    upperBounds.append(1E-2)

    # metal outlet angle
    lowerBounds.append(-1E-2)
    upperBounds.append(80)

    # scipy bounds object generation 
    bounds = Bounds(lowerBounds, upperBounds)

    # arguments generation
    args = (yCamberline, nPoints)

    # optimization
    res = optimize.minimize(fun=camberlineFunc, x0=[30, -10, 40], args=args, method='Nelder-Mead', bounds=bounds)

    # data allocation
    stagger     = res.x[0]
    metalInlet  = res.x[1]
    metalOutlet = res.x[2]

    # camberline object generation
    cLine = Camberline(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chebyschev=True, origin=True)

    # plotting data
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        cLine.plot(ax=ax)
        ax.plot(x, yUpper, color='orange',  linestyle='solid', linewidth=2, label='UPPER-SIDE')
        ax.plot(x, yLower, color='skyblue', linestyle='solid', linewidth=2, label='LOWER-SIDE')
        ax.plot(x, yCamberline, color='r',  linestyle='-.',    linewidth=2, label='CAMBERLINE-COMPUTED')
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.tight_layout()
        plt.show()

    return stagger, metalInlet, metalOutlet, yCamberline

def optimizeCamberline(upperLine: interpolate.interp1d, lowerLine: interpolate.interp1d, xLE: float, nPoints: int = 100, plot: bool = True) -> tuple:
    '''
    This function optimizes find a suitable camberline parametrizatio for the blade.
    '''

    # camberline coordinates computation 
    x = np.linspace(xLE, 1.0, nPoints)
    
    # isolating data
    yUpper = upperLine(x)
    yLower = lowerLine(x)

    # camberline ordinates
    yCamberline = (yUpper + yLower) / 2

    # scaling data into [0, 1] interval
    xCamberline = x / (1.0 - xLE)
    yCamberline = yCamberline / (1.0 - xLE)

    # optimization boundaries generation 
    lowerBounds = [] 
    upperBounds = [] 

    # stagger angle
    lowerBounds.append(0)
    upperBounds.append(60)
    
    # metal inlet angle
    lowerBounds.append(-50)
    upperBounds.append(1)

    # metal outlet angle
    lowerBounds.append(-1)
    upperBounds.append(80)

    # scipy bounds object generation 
    bounds = Bounds(lowerBounds, upperBounds)

    # arguments generation
    args = (yCamberline, nPoints)

    # optimization
    res = optimize.minimize(fun=camberlineFunc, x0=[30, -10, 40], args=args, method='Nelder-Mead', bounds=bounds)

    # data allocation
    stagger     = res.x[0]
    metalInlet  = res.x[1]
    metalOutlet = res.x[2]

    # camberline object generation
    cLine = Camberline(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chebyschev=True, origin=True)

    # plotting data
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        cLine.plot(ax=ax)
        ax.plot(x, yUpper, color='orange',  linestyle='solid', linewidth=2, label='UPPER-SIDE')
        ax.plot(x, yLower, color='skyblue', linestyle='solid', linewidth=2, label='LOWER-SIDE')
        ax.plot(xCamberline, yCamberline, color='r',  linestyle='-.',    linewidth=2, label='CAMBERLINE-COMPUTED')
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.tight_layout()
        plt.show()

    return stagger, metalInlet, metalOutlet, yCamberline

def optimizeBlade(
        data:      list | np.ndarray, 
        Nsuct:     int, 
        Npress:    int, 
        LEradius:  float = 2.5e-2, 
        nPoints:   int   = 100, 
        inletPos:  int   = 3, 
        outletPos: int   = 3, 
        theta:     float | list = [10, 15], 
        method:    str   = 'Nelder-Mead',
        nMax:      int   = 2, 
        tol:       float = 2.5E-5,
        flip:      bool  = False,
        plot:      bool  = True,
    ) -> np.ndarray:
    '''
    This function converts a coordinate based blade geometry into a Kulfan parametrization based geometry.
    
    Parameters
    ----------
    `data`: list | np.ndarray 
        blade geometry coordinates. 
    `nPoints`: int  
        study points number.

    Returns
    -------
    `kulfanParameters`: np.ndarray
        Kulfan parametrization parameters: [stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius]
    '''

    # moving blade into origin
    data = bladeInOrigin(data)

    # camberline analysis and flipping data analysis
    flip, __data = camberlineAnalysis(data=data, plot=False)

    # intepolating data 
    upperLine, lowerLine, upperData, lowerData, upperChord, lowerChord = interpolateData(__data)

    # leading edge position
    bladeLEpos(upperLine=upperLine, lowerLine=lowerLine)
    exit()

    # trailing edge radius computation 
    TEradius = bladeTEradius(data=__data, chord=axialChord)

    # allocating data 
    minCost = np.Inf
    
    # angle allocation
    theta = [0]
    thetaBlade = 0 

    for angle in theta:
        # generating angle for the rotation of the blade in the optimization
        # thetaRot = - thetaBlade + angle

        # rotating data 
        # data = rotate(__data, thetaRot)

        # intepolating data 
        upperLine, lowerLine, upperData, lowerData, upperChord, lowerChord = interpolateData(__data, flip=False)

        # getting main guess
        _, _, _, wedgeAngle = computeGuess(data=__data, upperData=upperData, lowerData=lowerData, inletPos=inletPos, outletPos=outletPos)
        
        # computing guess for the camberline
        stagger, metalInlet, metalOutlet, yCamberlineOptimized = optimizeCamberline(data=__data, LEpos=LEpos, plot=True)

        # profile line 
        Asuct  = np.concatenate([[np.sqrt(LEradius * 2) * 1.2], 0.3 * np.ones((Nsuct  - 3,)), [0.15]])
        Apress = np.concatenate([[np.sqrt(LEradius * 2) * 1.2], 0.3 * np.ones((Npress - 3,)), [0.15]])

        # setting up initial guess
        x = [
            stagger,
            metalInlet,
            metalOutlet, 
            LEradius
        ] 

        x = np.concatenate((x, Asuct, Apress, [wedgeAngle]))
        print('x = ', x)

        # arguments generation
        args = (
            Nsuct, 
            Npress,
            upperLine,
            lowerLine,
            TEradius,
            nPoints
        )

        # boundaries generation    
        bounds = boundsGenerator(stagger=stagger, metalInlet=metalInlet, metalOutlet=metalOutlet, Nsuct=Nsuct, Npress=Npress)

        # optimization using Nelder-Mead method
        cost = bladeFunc(x, Nsuct, Npress, upperLine, lowerLine, TEradius, nPoints)
        print('cost = ', cost)
        # exit()

    if True:
        counter = 0
        while cost > tol and counter < nMax: 
            # optimizing blade
            res = optimize.minimize(fun=bladeFunc, x0=x, args=args, method=method, bounds=bounds)
            
            # allocating data
            x = res.x
            
            # getting final cost
            cost = bladeFunc(x, Nsuct, Npress, upperLine, lowerLine, TEradius, nPoints)
            
            # updating counter
            counter = counter + 1

        # allocating data
        stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle = dataExtraction(x=res.x, Nsuct=Nsuct, Npress=Npress)

        # blade generation 
        tempBlade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

        if cost < minCost:
            minCost = cost
            blade = tempBlade

            kulfanParameters = np.concatenate([res.x, [TEradius]])
            
            # printing data
            print('Kulfan Parameters: {0}'.format(np.array2string(res.x, precision=2)))

        # if minCost < tol:
            # break
    
    xVal = np.array([
        2.98e+01, -4.13e+01, 7.01e+01,  1.31e-02,  3.10e-01, -1.00e-01,  5.24e-01,
        4.39e-02,  1.68e-01,  2.61e-03,  6.59e-01,  1.38e-02,  1.15e-01,  7.71e-04,
        6.25e-01,  1.11e-01,  3.00e-01, -5.80e-02,  3.68e-01,  1.69e-01,  5.32e-01,
        -2.86e-02, 3.54e-01,  2.06e-01,  2.19e-01,  4.60e-01,  2.57e-01,  2.01e-01,
        6.03e-01, -1.06e-02,  9.81e-01, -9.93e-02,  9.35e-01,  1.01e-01,  2.31e-01,
        3.02e-01, -8.98e-02,  4.97e-01, -9.54e-02,  5.68e-01,  1.40e-01,  3.29e-02,
        2.21e-01,  3.82e-01,  5.42e-01, -1.00e-01, -8.37e-03,  8.49e-01, -3.74e-02,
        1.26e-01,  7.25e-01, -1.88e-02,  3.13e-01,  3.55e-02,  9.27e-01,  4.01e-02,
        3.73e-01,  1.13e-02,  9.79e-01,  3.98e-01,  5.96e-03,  1.03e+00,  
    ])

    xVal = res.x

    stagger     = 30
    metalInlet  = -40
    metalOutlet = +70
    chord       = 1.0
    chebyschev  = True
    origin      = True
    number      = 2
    pitch       = 3.0
    vector      = False
    normalized  = True
    A          = [0.2, 0.3, 0.15]
    LEradius   = 3e-2 / 2
    # TEradius   = 2.5e-2 / 2 
    wedgeAngle = 30

    blade = Blade(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chord=1, pitch=1, Asuct=A, Apress=A, TEradius=TEradius, LEradius=LEradius)    

    # plotting results
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # getting profile line for plotting 
        x_ = np.linspace(0, 1, 400)

        # plotting optimized camberline from data
        ax.plot(np.linspace(0, 1, len(yCamberlineOptimized)), yCamberlineOptimized, 'k--', linewidth=3, label='OPTMIZATION.CAMBERLINE')
        
        # plotting coordinates
        plotCoords(__data, ax, theta=0, flip=flip, base=True)
        
        # getting from optimization's result
        bladeUpperLine, bladeLowerLine, camberlineLine, upperChord, bladeData, camberlineData, lowerChord, camberlineChord = bladeDataExtraction(xVal, Nsuct, Npress, TEradius=TEradius)
        ax.plot(x_ * upperChord, bladeUpperLine(x_) * upperChord,                             color='orange',    linestyle='solid', linewidth=3, label='RESULT.UPPER-LINE')
        ax.plot(x_ * lowerChord, bladeLowerLine(x_) * lowerChord,                             color='lightblue', linestyle='solid', linewidth=3, label='RESULT.LOWER-LINE')
        ax.plot(camberlineData[:,0] * camberlineChord, camberlineData[:,1] * camberlineChord, color='grey',      linestyle='solid', linewidth=3, label='RESULT.CAMBERLINE')
        
        # getting data from target blade object
        # bladeUpperLine, bladeLowerLine, camberlineLine, upperChord, bladeData, camberlineData, lowerChord, camberlineChord = bladeDataExtraction(x=None, blade=blade)
        # ax.plot(x_ * upperChord, bladeUpperLine(x_) * upperChord,                             color='orange',    linestyle='dotted', linewidth=3, label='TARGET.UPPER-LINE')
        # ax.plot(x_ * lowerChord, bladeLowerLine(x_) * lowerChord,                             color='lightblue', linestyle='dotted', linewidth=3, label='TARGET.LOWER-LINE')
        # ax.plot(camberlineData[:,0] * camberlineChord, camberlineData[:,1] * camberlineChord, color='k',         linestyle='dotted', linewidth=3, label='TARGET.CAMBERLINE')
        
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')  
        
        plt.show()

    return blade, kulfanParameters, minCost
