import numpy                  as     np 
import matplotlib.pyplot      as     plt 
from   scipy                  import misc, integrate, interpolate, optimize 
from   scipy.optimize         import Bounds    
from   geometryLIB.blade      import Blade 
from   geometryLIB.camberline import Camberline

def chebyschev(start: float, end: float, nPoints: int):
    
    x = (1 - np.cos((2 * (np.arange(nPoints) + 1) - 1) / (2 * nPoints) * np.pi)) / 2    
    
    # adding extremes
    x = np.concatenate(([0], x, [1]))

    x =  start + x * (end - start)

    return x 

def bladeInOrigin(data: list | np.ndarray) -> tuple[np.ndarray, float, float, float]:
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

    return data, minPos, minX, minXy

def bladeTEradius(data: list | np.ndarray, chord: float = 1) -> float:
    '''
    Blade traling edge computation and normalization with respect to the blade axial chord.
    '''

    # trailing edge computation 
    TEradius = np.linalg.norm(data[0,:] - data[-1,:]) / (2 * chord)
    print('>>> TRAILING EDGE RADIUS = {0:.2E}'.format(TEradius))

    return TEradius

def bladeLEpos(
        upperLine: interpolate.interp1d, 
        lowerLine: interpolate.interp1d, 
        dx:        float = 1E-3, 
        maxX:      float = 0.1, 
        nPoints:   int   = 30, 
        plot:      bool  = False
    ) -> tuple[float, float, float, float]:
    '''
    This function computes the blade leading edge position inside the coordinate based blade representation.

    Parameters
    ----------
    `upperLine`: scipy.interpolate.interp1d
        function object which is the numerical interpolation of the upper side blade coordinates 
    `lowerLine`: scipy.interpolate.interp1d
        function object which is the numerical interpolation of the lower side blade coordinates
    `dx`: float
        derivatives step
    `maxX`: float 
        blade interval upper boundary
    `nPoints`: int 
        number of points used for the leading edge blade analysis
    `plot`: bool    
        boolean value for the blade properties plotting

    Results
    -------
    `xLE`: float
        leading edge `x` position
    `yLE`: float 
        leading edge `y` position
    `LEradius`: float 
        leading edge radius
    `axialChord`: float
        blade axial chord
    '''

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
    xLE = x[maxPos]
    if position:
        yLE     = upperLine(xLE)
        yPrime  = misc.derivative(func=upperLine, x0=xLE, n=1, dx=dx)
        y2Prime = misc.derivative(func=upperLine, x0=xLE, n=2, dx=dx)
    else:
        yLE     = lowerLine(xLE)
        yPrime  = misc.derivative(func=lowerLine, x0=xLE, n=1, dx=dx)
        y2Prime = misc.derivative(func=lowerLine, x0=xLE, n=2, dx=dx)

    # axial chord computation
    axialChord = 1 - xLE 

    # computing leading edge radius 
    LEradius = 1 / (np.abs(y2Prime) * axialChord) 

    # printing results
    print('>>> dx                    = {0:.3E}'.format(dx))
    print('>>> LEADING EDGE POSITION = [{0:.3E}, {1:.3E}]'.format(xLE, yLE))
    print('>>> LEADING EDGE RADIUS   = {0:.3E}'.format(LEradius))

    if plot:
        # plotting data
        fig = plt.figure()

        # axes generation
        ax = fig.add_subplot(1,4,1)
        # plotting 
        ax.plot(x,   upperLine(x), 'r',  linewidth=2, label='INTERPOLATED.UPPER-LINE')
        ax.plot(x,   lowerLine(x), 'b',  linewidth=2, label='INTERPOLATED.LOWER-LINE')
        ax.plot(xLE, yLE,          'ok', linewidth=2, label='COMPUTED.LEADING-EDGE')
        # ax properties
        ax.grid(visible=True, linestyle='dotted')
        ax.set_title('Camberline')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc='upper left')

        # axes generation
        ax1 = fig.add_subplot(1,4,2)
        # plotting
        ax1.plot(x,   upperLineDer, 'r',  linewidth=2, label='COMPUTED.DERIVATIVE1.UPPER-LINE')
        ax1.plot(x,   lowerLineDer, 'b',  linewidth=2, label='COMPUTED.DERIVATIVE1.LOWER-LINE')
        ax1.plot(xLE, yPrime,       'ok', linewidth=2, label='COMPUTED.DERIVATIVE1.LEADING-EDGE')
        # ax properties
        ax1.grid(visible=True, linestyle='dotted')
        ax1.set_title(r'$y^{\prime}$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend(bbox_to_anchor=(0.0, -0.5), loc='upper left')

        # axes generation
        ax2 = fig.add_subplot(1,4,3)
        # plotting
        ax2.plot(x,   upperLineDerDer, 'r',  linewidth=2, label='COMPUTED.DERIVATIVE2.UPPER-LINE')
        ax2.plot(x,   lowerLineDerDer, 'b',  linewidth=2, label='COMPUTED.DERIVATIVE2.LOWER-LINE')
        ax2.plot(xLE, y2Prime,         'ok', linewidth=2, label='COMPUTED.DERIVATIVE2.LEADING-EDGE')
        # ax properties
        ax2.grid(visible=True, linestyle='dotted')
        ax2.set_title(r'$y^{\prime \prime}$')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.legend(bbox_to_anchor=(0.0, -0.5), loc='upper left')

        # axes generation
        ax3 = fig.add_subplot(1,4,4)
        # plotting
        ax3.plot(x, upperCurvature, 'r', linewidth=2, label='COMPUTED.CURVATURE.UPPER-LINE')
        ax3.plot(x, lowerCurvature, 'b', linewidth=2, label='COMPUTED.CURVATURE.LOWER-LINE')
        # ax properties
        ax3.grid(visible=True, linestyle='dotted')
        ax3.set_title(r'$\kappa$')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.legend(bbox_to_anchor=(0.0, -0.5), loc='upper left')

        plt.tight_layout()
        plt.show()

    return xLE, yLE, LEradius, axialChord 

def bladeLEpos_(upperLine: interpolate.interp1d, lowerLine: interpolate.interp1d, dx: float = 1E-3, maxX: float = 0.1, nPoints: int = 30, plot: bool = True) -> tuple:
    '''
    This function computes the blade leading edge position inside the coordinate based blade representation.
    '''

    # complete data assembly
    x_           = chebyschev(0, maxX, 100)
    upperData    = np.stack([x_[2::], upperLine(x_[2::])], axis=1)
    lowerData    = np.stack([np.flip(x_[1::]), np.flip(lowerLine(x_[1::]))], axis=1)
    
    # filtering data for interpolation 
    upperData = upperData[0:np.argmax(upperData[:,1])-1, :]
    lowerData = lowerData[np.argmin(lowerData[:,1]):np.argmax(lowerData[:,1]), :]
    completeData = np.vstack([lowerData, upperData])

    # complete flipped line 
    X = np.linspace(min(completeData[:,1]), max(completeData[:,1]), 100)
    f = interpolate.CubicSpline(completeData[:,1], completeData[:,0])

    # curvature study
    xStudy     = completeData[10:-10, 1]
    fDer       = misc.derivative(func=f, x0=xStudy, n=1, dx=dx)
    fDerDer    = misc.derivative(func=f, x0=xStudy, n=2, dx=dx)
    fCurvature = np.abs(fDerDer) / (1 + fDer**2)**(3/2)

    # computing properties
    fMaxPos = np.argmax(fCurvature)
    xLE     = xStudy[fMaxPos]
    fPrime  = misc.derivative(func=f, x0=xLE, n=1, dx=dx)
    fPrime2 = misc.derivative(func=f, x0=xLE, n=2, dx=dx)

    # computing leading edge position 
    yLE = f(xLE) 

    # setting up leading edge position 
    LEpos = [yLE, xLE]

    # axial chord computation
    axialChord = 1 - LEpos[0] 

    # computing leading edge radius 
    LEradius = 1 / (np.abs(fPrime2) * axialChord) 

    print('>>> dx                    = {0:.3E}'.format(dx))
    print('>>> LEADING EDGE POSITION = [{0:.3E}, {1:.3E}]'.format(LEpos[0], LEpos[1]))
    if max(LEradius, 2.5E-2) == 2.5E-2:
        LEradius = 2.5E-2
        print('>>> LEADING EDGE RADIUS TOO LOW: APPROXIMATED TO {0:.3E}'.format(LEradius))
    else:
        print('>>> LEADING EDGE RADIUS   = {0:.3E}'.format(LEradius))

    if plot:
        # plotting data
        fig = plt.figure()
        
        ax0 = fig.add_subplot(1,4,1)
        ax0.plot(f(X), X, 'r',  linewidth=3, label='Y')
        ax0.plot(yLE, xLE, 'ok', linewidth=3)

        ax0.grid(visible=True, linestyle='dotted')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_aspect('equal')

        ax1 = fig.add_subplot(1,4,2)

        ax1.plot(xStudy, fDer, 'orange')
        ax1.plot(xLE, fPrime, 'ok')

        ax1.set_title(r'$y^{\prime}$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(visible=True, linestyle='dotted')

        ax2 = fig.add_subplot(1,4,3)

        ax2.plot(xStudy, fDerDer, 'orange')
        ax2.plot(xLE, fPrime2, 'ok')

        ax2.set_title(r'$y^{\prime \prime}$')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(visible=True, linestyle='dotted')

        ax3 = fig.add_subplot(1,4,4)

        ax3.plot(xStudy, fCurvature, 'orange')
        
        ax3.set_title(r'$\kappa$')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.grid(visible=True, linestyle='dotted')

        plt.tight_layout()
        plt.show()

    return LEpos[0], LEpos[1], LEradius, axialChord 

def rotate(data: list | np.ndarray, theta: float, resize: bool = False) -> np.ndarray: 
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
        if resize:
            data[ii, :] = coord * cos
        else:
            data[ii, :] = coord

    return data

def plotCoords(data: list | np.ndarray, ax: plt.Axes = None, theta: float = 0.0, resize: bool = False, base: bool = True) -> None:
    '''
    This function plots the coordinate based data of the blade geometry.
    '''

    # rotating data
    if theta != 0:  
        data = rotate(data=data, theta=theta, resize=resize)
    
    # axes generation
    if not isinstance(ax, (plt.Axes)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    # getting lower side and upper side of the blade geometry 
    _, _, upperPart, lowerPart, upperChord, lowerChord = interpolateData(data=data)

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
    data, minPos, _, _ = bladeInOrigin(data=data)

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
        # checking data properties
        if upperLine(x) < lowerLine(x):
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
            ax.set_title('ERROR OVER PROFILE LINES')

            plt.tight_layout()
            plt.show()

            print('>>> ERROR OVER PROFILE LINES')
            print('>>> x = {0:f}'.format(x))
            print('>>> UPPER = {0:+.3E} || LOWER = {1:+.3E}'.format(upperLine(x), lowerLine(x)))

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

    Parameters
    ----------
    `x`: list | np.ndarray
        1D study vector which stores the optimization variables: [stagger, metalInlet, metalOutlet]
    `yCamberline`: list | np.ndarray
        1D vector which stores the blade camberline properties 
    `nPoints`: int 
        integer for the camberline study

    Returns
    -------
    `RMSE`: float 
        root mean squared error of the computed camberline with respect to `yCamberline` 
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
        staggerInterval:     float | int = 5,
        metalInletInterval:  float | int = 5, 
        metalOutletInterval: float | int = 5, 
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

def computeGuess(
        data:      list | np.ndarray, 
        upperData: list | np.ndarray, 
        lowerData: list | np.ndarray, 
        inletPos:  int, 
        outletPos: int
    ) -> tuple:
    '''
    This function computes the initial guess for a camberline approximation.

    Parameters
    ----------
    `data`: list | np.ndarray
        2D array which stores the blade coordinates
    `upperData`: list | np.ndarray
        2D array which stores the upper line blade coordinates
    `lowerData`: list | np.ndarray
        2D array which sotres the lower line blade coordinates 
    `inletPos`: int 
        array position for the computation of the inlet metal angle 
    `outletPos`: int 
        array position for the computation of the outlet metal angle 

    Returns
    -------
    `stagger`: float 
        blade stagger angle 
    `metalInlet`: float 
        inlet metal angle
    `metalOutlet`: float 
        outlet metal angle 
    `wedgeAngle`: float
        trailing edge wedge angle 
    '''
    
    # stagger angle computation 
    TEpoint = (data[-1, :] + data[0, :]) / 2
    stagger = np.rad2deg(np.arctan(TEpoint[1] / TEpoint[0]))

    # metal inlet angle computation 
    inletSlopePoint = (upperData[inletPos, :] + lowerData[inletPos, :]) / 2
    metalInlet = np.rad2deg(np.arctan(inletSlopePoint[1] / inletSlopePoint[0]))

    # metal outlet angle computation
    outletSlopePoint = (upperData[- outletPos, :] + lowerData[- outletPos, :]) / 2
    metalOutlet = np.rad2deg(np.arctan((TEpoint[1] - outletSlopePoint[1]) / (TEpoint[0] - outletSlopePoint[0])))

    print('>>> COMPUTED DATA')
    print('>>> TRAILING EDGE POSITION   = {0}'.format(TEpoint))
    print('>>> STAGGER ANGLE            = {0:+.3E} '.format(stagger))
    print('>>> INLET METAL ANGLE        = {0:+.3E} '.format(metalInlet))
    print('>>> OUTLET METAL ANGLE       = {0:+.3E} '.format(metalOutlet))

    # wedge angle computation
    if (data[0, 0] - data[-1, 0] == 0) and (data[0, 1] - data[-1, 1] == 0):
        # wedge angle vector allocation
        upperVec = TEpoint - upperData[1,:]
        lowerVec = TEpoint - lowerData[-2,:]
        # wedge angle computation
        wedgeAngle = np.dot(upperVec, lowerVec)
        wedgeAngle = wedgeAngle / (np.linalg.norm(upperVec) * np.linalg.norm(lowerVec))
        wedgeAngle = np.rad2deg(np.arccos(wedgeAngle)) / 2
        wedgeAngle = max(1, wedgeAngle)
        print('>>> WEDGE ANGLE              = {0:+.3E} '.format(wedgeAngle))
    else: 
        wedgeAngle = 15 
        print('>>> WEDGE ANGLE (BY DEFAULT) = {0:+.3E} '.format(wedgeAngle))

    return stagger, metalInlet, metalOutlet, wedgeAngle

def camberlineAnalysis(data: list | np.ndarray, nPoints: int = 100, plot: bool = False) -> tuple[bool, np.ndarray]:
    '''
    This function checks the camberline curvature and makes changes over the camberline properties. 
    
    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the blade coordinate points
    `nPoints`: int
        integer which defines the number of points for the camberline analysis
    `plot`: bool 
        boolean value for the plotting of the main operation results

    Returns
    -------
    `flip`: bool    
        boolean value which defines if the blade coordinates have been flipped for the blade optimization
    `data`: np.ndarray
        2D vector which stores the rotated coordinates
    '''

    # getting minimum position 
    data, minPos, _, _ = bladeInOrigin(data=data)

    # trailing edge angle computation
    xTE = (data[0, 0] + data[-1, 0]) / 2
    yTE = (data[0, 1] + data[-1, 1]) / 2

    # computing angle between the left furthest point and the trailing edge 
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

    # evaluating data
    x      = np.linspace(0, 1, nPoints)
    yUpper = upperLine(x)
    yLower = lowerLine(x)

    # computing camberline 
    yCamberline = (yUpper + yLower) / 2

    # rotating data 
    camberlinePoints = np.stack([x, yCamberline], axis=1)
    studyCamberline  = rotate(data=camberlinePoints, theta=-theta)

    # integral computation for flipping  
    integral = integrate.trapezoid(y=studyCamberline[:,1], x=studyCamberline[:,0])
    print('>>> INTEGRAL = {0:+.2E}'.format(integral))

    # flipping data
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

def optimizeCamberline(
        upperLine:  interpolate.interp1d, 
        lowerLine:  interpolate.interp1d, 
        LEpos:      list | np.ndarray, 
        upperChord: float = 1, 
        lowerChord: float = 1, 
        nPoints:    int   = 100, 
        plot:       bool  = False
    ) -> tuple[float, float, float, np.ndarray]:
    '''
    This function optimizes find a suitable camberline parametrizatio for the blade.
    '''

    # camberline coordinates computation 
    x = np.linspace(LEpos[0], 1.0, nPoints)

    # isolating data
    yUpper = upperLine(x) - upperLine(x[0])
    yLower = lowerLine(x) - lowerLine(x[0])

    # camberline ordinates
    yCamberline = (yUpper + yLower) / 2

    # scaling data into [0, 1] interval
    xCamberline = np.linspace(0, 1, nPoints)
    # camberline in origin
    yCamberline = yCamberline - yCamberline[0]
    # scaling camberline because the leading edge position
    yCamberline = yCamberline / (1.0 - LEpos[0]) 

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

    # camberline properties printout
    print('>>> STAGGER      = {0:+.3E}'.format(stagger))
    print('>>> METAL INLET  = {0:+.3E}'.format(metalInlet))
    print('>>> METAL OUTLET = {0:+.3E}'.format(metalOutlet))

    # plotting data
    if plot:
        fig = plt.figure(figsize=(8,8))

        # camberline object generation
        cLine = Camberline(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chebyschev=True, origin=True)
        
        # chebyscev study points
        x_ = chebyschev(0, 1, nPoints)    

        # axes generation
        ax = fig.add_subplot(1,2,1)
        # camberline plotting
        cLine.plot(ax=ax, label='OPTIMIZATION.CAMBERLINE')
        # plotting blade profile lines and computed camberline
        ax.plot(x,           yUpper,      color='orange',  linestyle='solid', linewidth=3, label='DATA.UPPER-SIDE')
        ax.plot(x,           yLower,      color='skyblue', linestyle='solid', linewidth=3, label='DATA.LOWER-SIDE')
        ax.plot(xCamberline, yCamberline, color='r',       linestyle='-.',    linewidth=3, label='DATA.CAMBERLINE-UPLOW)')
        # ax propeties
        ax.legend(bbox_to_anchor=(0,-0.5), loc='upper left')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # axes generation
        ax1 = fig.add_subplot(1,2,2)
        # plotting scaled camberline 
        ax1.plot(x_ * upperChord,                     upperLine(x_) * upperChord,          color='orange',  linestyle='solid', linewidth=3, label='DATA.UPPER-SIDE')
        ax1.plot(x_ * lowerChord,                     lowerLine(x_) * lowerChord,          color='skyblue', linestyle='solid', linewidth=3, label='DATA.LOWER-SIDE')
        ax1.plot(cLine.x * (1 - LEpos[0]) + LEpos[0], cLine.y * (1 - LEpos[0]) + LEpos[1], color='k',       linestyle='solid', linewidth=3, label='OPTIMIZATION.CAMBERLINE')
        # ax properties
        ax1.legend(bbox_to_anchor=(0,-0.5), loc='upper left')
        ax1.grid(visible=True, linestyle='dotted')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()

    return stagger, metalInlet, metalOutlet, yCamberline

def optimizeBlade(
        data:      list | np.ndarray, 
        Nsuct:     int, 
        Npress:    int, 
        angle:     float = 0.0,
        LEradius:  float = 2.5e-2, 
        nPoints:   int   = 100, 
        inletPos:  int   = 3, 
        outletPos: int   = 3,  
        method:    str   = 'Nelder-Mead',
        nMax:      int   = 2, 
        tol:       float = 2.5E-5,
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
    data, _, _, _ = bladeInOrigin(data)

    # camberline analysis and flipping data analysis
    flip, __data = camberlineAnalysis(data=data, plot=False)

    # allocating data 
    minCost = np.Inf
    
    # rotating blade geometry for the optimization
    if angle != 0:
       __data = rotate(__data, angle, resize=False)

    # intepolating data 
    upperLine, lowerLine, upperData, lowerData, upperChord, lowerChord = interpolateData(__data)

    # computing wedge angle 
    _, _, _, wedgeAngle = computeGuess(data=__data, upperData=upperData, lowerData=lowerData, inletPos=inletPos, outletPos=outletPos)    

    # leading edge position
    xLE, yLE, LEradius, axialChord = bladeLEpos(upperLine=upperLine, lowerLine=lowerLine, plot=False)

    # camberline approximation
    stagger, metalInlet, metalOutlet, yCamberlineOptimized = optimizeCamberline(upperLine=upperLine, lowerLine=lowerLine, LEpos=[xLE, yLE], upperChord=upperChord, lowerChord=lowerChord, plot=False)

    # trailing edge radius computation 
    TEradius = bladeTEradius(data=__data, chord=axialChord)

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

    # setting up total guess array
    x = np.concatenate((x, Asuct, Apress, [wedgeAngle]))
    print('>>> GUESS VECTOR = {0} '.format(x))

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

    # checking camberline properties: if RMSE/cost computed -> good to go
    cost = bladeFunc(x, Nsuct, Npress, upperLine, lowerLine, TEradius, nPoints)
    print('>>> INITIAL RMSE = {0:.3E}'.format(cost))

    # setting study parameters
    counter = 0
    # optimization in different steps
    while cost > tol and counter < nMax: 
        # updating counter
        counter = counter + 1
        
        # optimizing blade
        res = optimize.minimize(fun=bladeFunc, x0=x, args=args, method=method, bounds=bounds)
        
        # allocating data
        x = res.x
        
        # getting final cost
        cost = bladeFunc(x, Nsuct, Npress, upperLine, lowerLine, TEradius, nPoints)
        print('>>> OPTIMIZATION #{0:d} => RMSE = {1:.3E}'.format(counter, cost))

    # blade generation 
    # allocating data
    stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle = dataExtraction(x=res.x, Nsuct=Nsuct, Npress=Npress)
    # blade object
    blade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)
    
    # setting up results value
    xVal = res.x

    # Kulfan parameters
    kulfanParameters = np.concatenate([res.x, [TEradius]])
    print('>>> KULFAN PARAMETERS = {0}'.format(np.array2string(res.x, precision=2)))

    # plotting results
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # getting profile line for plotting 
        x_ = chebyschev(0, 1, 300)

        # plotting optimized camberline from data
        ax.plot(np.linspace(0, 1, len(yCamberlineOptimized)), yCamberlineOptimized, 'k--', linewidth=3, label='OPTMIZATION.CAMBERLINE')
        
        # getting from optimization's result
        bladeUpperLine, bladeLowerLine, camberlineLine, upperChord, bladeData, camberlineData, lowerChord, camberlineChord = bladeDataExtraction(xVal, Nsuct, Npress, TEradius=TEradius)
        bladeData = rotate(bladeData, -angle, resize=True)
        ax.plot(bladeData[:,0], bladeData[:,1], 'k', linewidth=5, label='RESULT:BLADE')
        # ax.plot(x_ * upperChord, bladeUpperLine(x_) * upperChord,                             color='orange',    linestyle='solid', linewidth=3, label='RESULT.UPPER-LINE')
        # ax.plot(x_ * lowerChord, bladeLowerLine(x_) * lowerChord,                             color='lightblue', linestyle='solid', linewidth=3, label='RESULT.LOWER-LINE')
        # ax.plot(camberlineData[:,0] * camberlineChord, camberlineData[:,1] * camberlineChord, color='grey',      linestyle='solid', linewidth=3, label='RESULT.CAMBERLINE')
        
        # plotting coordinates
        plotCoords(__data, ax, theta=-angle, resize=True, base=True)
        
        # getting data from target blade object
        bladeUpperLine, bladeLowerLine, camberlineLine, upperChord, bladeData, camberlineData, lowerChord, camberlineChord = bladeDataExtraction(x=None, blade=blade)
        ax.plot(x_ * upperChord, bladeUpperLine(x_) * upperChord,                             color='orange',    linestyle='dotted', linewidth=3, label='TARGET.UPPER-LINE')
        ax.plot(x_ * lowerChord, bladeLowerLine(x_) * lowerChord,                             color='lightblue', linestyle='dotted', linewidth=3, label='TARGET.LOWER-LINE')
        ax.plot(camberlineData[:,0] * camberlineChord, camberlineData[:,1] * camberlineChord, color='k',         linestyle='dotted', linewidth=3, label='TARGET.CAMBERLINE')
        
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')  
        
        plt.show()

    return blade, kulfanParameters, minCost
