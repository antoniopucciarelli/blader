import numpy                      as     np 
import matplotlib.pyplot          as     plt 
from   scipy                      import interpolate
from   scipy                      import optimize
from   scipy.optimize             import Bounds    
from   src.geometryLIB.blade      import Blade 
from   src.geometryLIB.camberline import Camberline

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
    if flip:
        if base:
            ax.plot(data[:,0], - data[:,1], color='c', linestyle='solid', linewidth=2, label='DATA')
        ax.plot(upperPart[:,0] * upperChord, upperPart[:,1] * upperChord, color='r', linestyle='-.', linewidth=2, label='UPPER-SIDE')
        ax.plot(lowerPart[:,0] * lowerChord, lowerPart[:,1] * lowerChord, color='b', linestyle='-.', linewidth=2, label='LOWER-SIDE')
    else: 
        if base:
            ax.plot(data[:,0], data[:,1], color='c', linestyle='solid', linewidth=2, label='DATA')
        ax.plot(upperPart[:,0] * upperChord, upperPart[:,1] * upperChord, color='r', linestyle='-.', linewidth=2, label='UPPER-SIDE')
        ax.plot(lowerPart[:,0] * lowerChord, lowerPart[:,1] * lowerChord, color='b', linestyle='-.', linewidth=2, label='LOWER-SIDE')
    
    # axes decoration
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.grid(visible=True, linestyle='dotted')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()

def interpolateData(data: list | np.ndarray, flip: bool = False) -> tuple:
    '''
    This function interpolates the geometry data from the target blade dataset.

    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the target blade geometry in [x, y] coordinates.

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
    minX   = np.min(data[:, 0])
    minPos = np.argmin(data[:, 0])
    minXy  = data[minPos, 1]

    # updating data
    data[:,0] = data[:,0] - minX 
    data[:,1] = data[:,1] - minXy

    # setting up data
    if flip:
        upperPart = np.array(data[minPos::, :])
        lowerPart = np.array(data[0:minPos+1, :])
    else:
        lowerPart = np.array(data[minPos::, :])
        upperPart = np.array(data[0:minPos+1, :])

    # getting blade chord
    upperChord = np.max(upperPart[:, 0]) - np.min(upperPart[:, 0])
    lowerChord = np.max(lowerPart[:, 0]) - np.min(lowerPart[:, 0])
    
    # normalize data 
    lowerPart = lowerPart / lowerChord 
    upperPart = upperPart / upperChord 

    # interpolating data 
    upperLine = interpolate.interp1d(upperPart[:, 0], upperPart[:, 1])
    lowerLine = interpolate.interp1d(lowerPart[:, 0], lowerPart[:, 1])

    # testing data 
    xVec = np.linspace(0.01, 0.99, 100)

    for x in xVec:
        if upperLine(x) < lowerLine(x):
            # checking data properties
            # # plotting data 
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            
            # ax.plot(upperPart[:,0], upperPart[:,1], 'r', linewidth=2, label='UPPER-SIDE')
            # ax.plot(lowerPart[:,0], lowerPart[:,1], 'b', linewidth=2, label='LOWER-SIDE')

            # ax.plot([x], [upperLine(x)], 'or', label='UPPER-SIDE-ERROR_POINT')
            # ax.plot([x], [lowerLine(x)], 'ob', label='LOWER-SIDE-ERROR_POINT')

            # ax.set_aspect('equal')
            # ax.grid(visible=True, linestyle='dotted')
            
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')

            # plt.show()

            print('x = {0:f}'.format(x))
            print(upperLine(x), lowerLine(x))

            raise ValueError('::: Wrong interpolation, something is wrong with the input geometry (`data`).')

    return upperLine, lowerLine, upperPart, lowerPart, upperChord, lowerChord

def dataExtraction(
        x:           list | np.ndarray, 
        Nsuct:       int, 
        Npress:      int, 
        TEradiusDOF: bool
    ) -> tuple:

    # initializing data
    stagger  = x[0]
    metalIn  = x[1]
    metalOut = x[2]
    LEradius = x[3] 

    # initializing data
    if TEradiusDOF:
        TEradius   = x[-1] 
        wedgeAngle = x[-2]
    else:
        TEradius   = 0.0
        wedgeAngle = x[-1]
    
    # getting the rest of the data
    Asuct  = x[4:4+Nsuct] 
    Apress = x[4+Nsuct:4+Nsuct+Npress]

    return stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius

def bladeDataExtraction(
        x:           list | np.ndarray,
        Nsuct:       int, 
        Npress:      int,
        TEradiusDOF: bool = True,
    ) -> float:

    # getting data 
    stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius = dataExtraction(x=x, Nsuct=Nsuct, Npress=Npress, TEradiusDOF=TEradiusDOF)

    # blade generation 
    # try:
    TEradius = 2.5E-2 / 2
    blade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

    XPS, YPS, XSS, YSS = blade.coordinate()

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

    return bladeUpperLine, bladeLowerLine, upperChord, lowerChord

def func(
        x:           list | np.ndarray,
        Nsuct:       int, 
        Npress:      int,
        upperLine:   interpolate.interp1d,
        lowerLine:   interpolate.interp1d,
        TEradiusDOF: bool = True,
        nPoints:     int  = 100
    ) -> float:

    # getting data 
    stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius = dataExtraction(x=x, Nsuct=Nsuct, Npress=Npress, TEradiusDOF=TEradiusDOF)
    
    TEradius = 2.5E-2 / 2
    
    # blade generation 
    # try:
    blade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

    XPS, YPS, XSS, YSS = blade.coordinate()

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
    x = np.linspace(0.01, 0.99, nPoints)

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

    # except:
    #     RMSE = 1e+2

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(X, Y)
    # ax.plot(XPS, YPS)
    # ax.plot(XSS, YSS)
    # ax.set_aspect('equal')
    # ax.grid(visible=True, linestyle='dotted')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')  
    # plt.show()

    print("RMSE = {0:.3E}".format(RMSE))

    return RMSE

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

def boundsGenerator(
        stagger:             int | float, 
        metalInlet:          int | float,
        metalOutlet:         int | float,
        Nsuct:               int, 
        Npress:              int,
        staggerInterval:     float | int = 10,
        metalInletInterval:  float | int = 10, 
        metalOutletInterval: float | int = 10, 
        metalInBound:        list        = [-60, -1E-1],
        metalOutBound:       list        = [1E-1, 80],
        Abounds:             list        = [-1E-1, 2],
        LEradiusBounds:      list        = [1E-2, 1],
        wedgeAngleMax:       int | float = 60,
        TEradiusDOF:         bool        = True,
        TEradiusMax:         float       = 2.5E-2,
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
    # lowerBounds.append(metalInBound[0])
    # upperBounds.append(metalInBound[1])
    lowerBounds.append(metalInlet - metalInletInterval)
    upperBounds.append(metalInlet + metalInletInterval)

    # metal outlet angle
    # lowerBounds.append(metalOutBound[0])
    # upperBounds.append(metalOutBound[1])
    lowerBounds.append(metalOutlet - metalOutletInterval)
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
    lowerBounds.append(0)
    upperBounds.append(np.abs(wedgeAngleMax))

    # TEradius bounds
    if TEradiusDOF:
        lowerBounds.append(0)
        upperBounds.append(np.abs(TEradiusMax))
    
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
    LEradius = 1.5E-2

    # wedge angle approximation 
    upperWedgeOutlet = np.rad2deg(np.arctan((TEpoint[1] - upperData[-2,1]) / (TEpoint[0] - upperData[-2,0])))
    lowerWedgeOutlet = np.rad2deg(np.arctan((TEpoint[1] - lowerData[2,1]) / (TEpoint[0] - lowerData[2,0])))
    wedgeAngle = (upperWedgeOutlet + lowerWedgeOutlet) / 2 - metalOutlet
    wedgeAngle = max(1, wedgeAngle)

    return stagger, metalInlet, metalOutlet, wedgeAngle

def optimizeCamberline(upperLine: list | np.ndarray, lowerLine: list | np.ndarray, nPoints: int = 100, plot: bool = True) -> tuple:
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
    upperBounds.append(-1)

    # metal outlet angle
    lowerBounds.append(20)
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
        ax.plot(x, yUpper, color='orange',  linestyle='solid', linewidth=2)
        ax.plot(x, yLower, color='skyblue', linestyle='solid', linewidth=2)
        ax.plot(x, yCamberline, color='r',  linestyle='-.',    linewidth=2)
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.tight_layout()
        plt.show()

    return stagger, metalInlet, metalOutlet 

def optimizer(
        data:      list | np.ndarray, 
        Nsuct:     int, 
        Npress:    int, 
        LEradius:  float = 2.5e-2, 
        nPoints:   int   = 100, 
        inletPos:  int   = 3, 
        outletPos: int   = 3, 
        theta:     float | list = [10, 15, 20], 
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

    # flipping data
    if flip:
        data[:, 1] = - data[:, 1]

    # getting minimum position 
    minX   = np.min(data[:, 0])
    minPos = np.argmin(data[:, 0])
    minXy  = data[minPos, 1]

    # updating data
    data[:,0] = data[:,0] - minX 
    data[:,1] = data[:,1] - minXy

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # plotCoords(data, ax, theta=0, flip=flip)
    # ax.set_aspect('equal')
    # ax.grid(visible=True, linestyle='dotted')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')  
    # plt.show()

    # setting up not modifiable object
    __data = data

    # angle computation
    if (__data[0, 1] + __data[-1,1]) != 0:
        thetaBlade = np.rad2deg(np.arctan(__data[0,1] / __data[0,0]))
        # thetaBlade = 0.0
    else:
        thetaBlade = 0.0

    # checking trailing edge DOF inside the optimization
    # if __data[-1, 0] == __data[0, 0] and __data[-1, 1] == __data[0, 1]:
    #     TEradiusDOF = False
    #     TEradius = 0.0
    #     print('>>> TE radius DOF de-activated')
    # else:
    #     TEradiusDOF = True
    #     TEradius = 2.5e-2 / 2
    #     print('>>> TE radius DOF activated')
    TEradiusDOF = False

    # allocating data 
    minCost = np.Inf
    # theta = [0]
    for angle in theta:
        # generating angle for the rotation of the blade in the optimization
        thetaRot = - thetaBlade + angle

        # main rotation matrix parameters
        cos = np.cos(np.deg2rad(thetaRot))
        sin = np.sin(np.deg2rad(thetaRot))
        
        # rotation matrix generation
        rotMatrix = np.array([[cos, -sin], [sin, cos]])

        # rotating data 
        for ii, coord in enumerate(__data):
            coord = np.matmul(rotMatrix, coord) 
            data[ii, :] = coord / cos

        # intepolating data 
        upperLine, lowerLine, upperData, lowerData, upperChord, lowerChord = interpolateData(data, flip=flip)

        # getting main guess
        stagger, metalInlet, metalOutlet, wedgeAngle = computeGuess(data=data, upperData=upperData, lowerData=lowerData, inletPos=inletPos, outletPos=outletPos)
        
        try:
            stagger, metalInlet, metalOutlet = optimizeCamberline(upperLine=upperLine, lowerLine=lowerLine, plot=True)
        except:
            pass 

        # profile line 
        Asuct  = np.concatenate([[np.sqrt(LEradius * 2) * 1.2], 0.3 * np.ones((Nsuct  - 3,)), [0.15]])
        Apress = np.concatenate([[np.sqrt(LEradius * 2) * 1.2], 0.3 * np.ones((Npress - 3,)), [0.15]])

        x = [
            stagger,
            metalInlet,
            metalOutlet, 
            LEradius
        ] 
        
        # setting up initial guess
        # if TEradiusDOF:
            # x = np.concatenate((x, Asuct, Apress, [wedgeAngle, TEradius]))
        # else:
            # x = np.concatenate((x, Asuct, Apress, [wedgeAngle]))

        x = np.concatenate((x, Asuct, Apress, [wedgeAngle]))
        print('x = ', x)
        # exit()

        # arguments generation
        args = (
            Nsuct, 
            Npress,
            upperLine,
            lowerLine,
            TEradiusDOF,
            nPoints
        )

        # boundaries generation    
        bounds = boundsGenerator(stagger=stagger, metalInlet=metalInlet, metalOutlet=metalOutlet, Nsuct=Nsuct, Npress=Npress, TEradiusDOF=TEradiusDOF)

        # optimization using Nelder-Mead method
        cost = func(x, Nsuct, Npress, upperLine, lowerLine, TEradiusDOF, nPoints)

        counter = 0
        while cost > tol and counter < nMax: 
            # optimizing blade
            res = optimize.minimize(fun=func, x0=x, args=args, method=method, bounds=bounds, tol=1e-7)
            
            # allocating data
            x = res.x
            
            # getting final cost
            cost = func(x, Nsuct, Npress, upperLine, lowerLine, TEradiusDOF, nPoints)
            
            # updating counter
            counter = counter + 1

        # allocating data
        stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius = dataExtraction(x=res.x, Nsuct=Nsuct, Npress=Npress, TEradiusDOF=TEradiusDOF)

        # blade generation 
        TEradius = 2.5E-2 / 2
        tempBlade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

        if cost < minCost:
            minCost = cost
            blade = tempBlade

            # allocating data
            if TEradiusDOF:
                kulfanParameters = res.x 
            else:
                kulfanParameters = np.concatenate([res.x, [TEradius]])
            
            # printing data
            print('Kulfan Parameters: {0}'.format(np.array2string(res.x, precision=2)))

        if minCost < tol:
            break

    # plotting results
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # getting profile line for plotting 
        x_ = np.linspace(0, 1, 400)
        bladeUpperLine, bladeLowerLine, upperChord, lowerChord = bladeDataExtraction(res.x, Nsuct, Npress, TEradiusDOF=TEradiusDOF)
        ax.plot(x_ * upperChord, bladeUpperLine(x_) * upperChord, 'orange',    linewidth=3, label='UPPER-LINE-BLADE')
        ax.plot(x_ * lowerChord, bladeLowerLine(x_) * lowerChord, 'lightblue', linewidth=3, label='LOWER-LINE-BLADE')
        plotCoords(data, ax, theta=-angle, flip=flip, base=False)
        
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')  
        
        plt.show()

    return blade, kulfanParameters, minCost
