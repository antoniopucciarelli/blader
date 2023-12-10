import time
import numpy                  as     np 
import matplotlib.pyplot      as     plt 
from   scipy                  import misc, integrate, interpolate, optimize 
from   scipy.optimize         import Bounds    
from   geometryLIB.blade      import Blade 
from   geometryLIB.camberline import Camberline

def chebyschev(start: float, end: float, nPoints: int) -> np.ndarray:
    '''
    This function computes the Chebyschev nodes over an interval. 
    
    Parameters
    ----------
    `start`: float
        discretization starting point.
    `end`: float 
        discretization ending point.
    `nPoints`: int 
        number of points - 2 to be used for the discretization.
    
    Returns
    -------
    `x`: np.ndarray
        array which stores the Chebyschev nodes.
    '''
    
    # computing inner nodes
    x = (1 - np.cos((2 * (np.arange(nPoints) + 1) - 1) / (2 * nPoints) * np.pi)) / 2    
    
    # adding extremes
    x = np.concatenate(([0], x, [1]))

    # scaling nodes
    x = start + x * (end - start)

    return x 

def checkNAN(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    This function checks if the blade coordinates array have a NAN element in it.

    Parameters
    ----------
    `x`: np.ndarray
        array which stores the abscissa of the coordinates array.
    `y`: np.ndarray
        array which stores the ordinates of the coordinates array.
    
    Returns
    -------
    `x`: np.ndarray
        array which stores the abscissa of the coordinates array without NAN element.
    `y`: np.ndarray
        array which stores the ordinates of the coordinates array without NAN element.
    '''
    
    if str(x[0]) == 'nan':
        x = x[1::]
        y = y[1::]
    
    if str(x[-1]) == 'nan':
        x = x[0:-1]
        y = y[0:-1]

    return x, y

def bladeInOrigin(data: list | np.ndarray, scale: bool = True) -> tuple[np.ndarray, float, float, float]:
    '''
    This function displaces the blade data into the origin.

    Parameters
    ----------
    `data`: list or np.ndarray
        2D vector which stores the blade coordinates in [x, y] fashion.
    `scale`: bool
        boolean value which allows to scale the blade with respect to the blade axial chord.

    Returns
    -------
    `data`: np.ndarray
        2D array which stores the displaced blade coordinates.
    `minPos`: int 
        array position which defines the lowest `x` coordinate of the blade geometry.
    `minX`: float 
        abscissa which defines the lowest `x` coordinate of the blade geometry.
    `minXy`: float 
        ordinate relative to `minX`. 
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

    if scale: 
        maxX = max(data[:, 0])
        data = data / maxX

    return data, minPos, minX, minXy

def bladeTEradius(data: list | np.ndarray, chord: float = 1) -> float:
    '''
    Blade traling edge computation and normalization with respect to the blade axial chord.

    Parameters
    ----------
    `data`: list or np.ndarray
        2D vector which stores the blade coordinates in [x, y] fashion.
    `chord`: float
        blade chord for the trailing edge scaling.

    Returns
    -------
    `TEradius`: float 
        scaled trailing edge radius.
    '''

    # trailing edge computation 
    TEradius = np.linalg.norm(data[0,:] - data[-1,:]) / (2 * chord)
    print('>>> TRAILING EDGE RADIUS = {0:.2E}'.format(TEradius))

    return TEradius

def bladeLEpos(
        upperLine: interpolate._interpolate.interp1d, 
        lowerLine: interpolate._interpolate.interp1d, 
        dx:        float = 1E-3, 
        maxX:      float = 0.1, 
        nPoints:   int   = 30, 
        bothSide:  bool  = False,
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
    `bothSide`: bool 
        boolean value which enables the leading edge position of the blade in the lower part of the blade EVEN being reversed by afore by the program
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
    if bothSide:
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
    else:
        maxPos  = np.argmax(upperCurvature)
        xLE     = x[maxPos]
        yLE     = upperLine(xLE)
        yPrime  = misc.derivative(func=upperLine, x0=xLE, n=1, dx=dx)
        y2Prime = misc.derivative(func=upperLine, x0=xLE, n=2, dx=dx)

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
        fig = plt.figure(figsize=(10,10))

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
        ax.legend(bbox_to_anchor=(-0.4, -0.5), loc='upper left')

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
        fig.legend(bbox_to_anchor=(-0.1, -1), loc='upper left')

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
        # ax2.legend(bbox_to_anchor=(0.0, -0.5), loc='upper left')

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
        # ax3.legend(bbox_to_anchor=(-0.2, -0.5), loc='upper left')

        plt.tight_layout()
        plt.show()

    return xLE, yLE, LEradius, axialChord

def rotate(data: list | np.ndarray, theta: float, resize: bool = False) -> np.ndarray: 
    '''
    This function rotates data with respect to the origin with a theta angle (in degrees).

    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the blade coordinate points
    `theta`: float 
        number which defines the rotation angle. The angle must be expressed in degrees
    `resize`: bool
        boolean value for the resizing of the coordinates with respect to the cosine

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
            data[ii, :] = coord / cos
        else:
            data[ii, :] = coord

    return data

def plotCoords(
        data:   list | np.ndarray, 
        ax:     plt.Axes = None, 
        theta:  float    = 0.0, 
        resize: bool     = False, 
        base:   bool     = True
    ) -> None:
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
    _, _, _, _, upperPart, lowerPart, upperChord, lowerChord = interpolateData(data=data, kind='linear')

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

def interpolateData(
        data: list | np.ndarray, 
        kind: str = 'cubic', 
        plot: bool = False
    ) -> tuple[interpolate._interpolate.interp1d, interpolate._interpolate.interp1d, np.ndarray, np.ndarray, float, float]:
    '''
    This function interpolates the geometry data from the target blade dataset.

    Parameters
    ----------
    `data`: list | np.ndarray
        2D vector which stores the target blade geometry in [x, y] coordinates.
    `kind`: str 
        interpolation properties.
    `plot`: bool
        boolean value for the plotting of the operation results.

    Returns
    -------
    `upperLine`: interpolate._interpolate.interp1d 
        function object which parametrizes the upper part of the blade. The upper part of the blade starts from the most left point of the geometry to the most right point of the geometry.
    `lowerLine`: interpolate._interpolate.interp1d
        function object which parametrizes the lower part of the blade. The lower part of the blade starts from the most left point of the geometry to the most right point of the geometry.
    `upperPart`: np.ndarray
        2D vector which stores the upper part coordinates of the blade. 
    `lowerPart`: np.ndarray
        2D vector which stores the lower part coordinates of the blade.
    `upperChord`: float 
        upper side of the blade which defines the axial chord.
    `lowerChord`: float 
        lower side of the blade which defines the axial chord.
    '''

    # getting minimum position 
    data, minPos, _, _ = bladeInOrigin(data=data)

    # setting up data
    upperPart = np.array(data[0:minPos+1, :])
    lowerPart = np.array(data[minPos::, :])

    # getting blade chord
    upperChord = np.max(upperPart[:, 0]) - np.min(upperPart[:, 0])
    lowerChord = np.max(lowerPart[:, 0]) - np.min(lowerPart[:, 0])
    
    # interpolating data 
    upperLineReal = interpolate.interp1d(upperPart[:, 0], upperPart[:, 1], kind=kind)
    lowerLineReal = interpolate.interp1d(lowerPart[:, 0], lowerPart[:, 1], kind=kind)
    
    # normalize data 
    upperPart = upperPart / upperChord 
    lowerPart = lowerPart / lowerChord 

    # interpolating normalized data
    upperLine = interpolate.interp1d(upperPart[:, 0], upperPart[:, 1], kind=kind)
    lowerLine = interpolate.interp1d(lowerPart[:, 0], lowerPart[:, 1], kind=kind)
    
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

    return upperLine, lowerLine, upperLineReal, lowerLineReal, upperPart, lowerPart, upperChord, lowerChord

def dataExtraction(
        x:           list | np.ndarray, 
        Nsuct:       int, 
        Npress:      int
    ) -> tuple:
    '''
    This function extracts the blade parameters from an array.

    Parameters
    ----------
    `x`: list | np.ndarray
        Kulfan parameters for the blade geometry definition
    `Nsuct`: int 
        number of DOF for the suction side of the blade
    `Npress`: int 
        number of DOF for the pressure side of the blade

    Returns
    -------
    `stagger`: float 
        camberline stagger angle (in degrees)
    `metalIn`: float 
        camberline metal inlet angle (in degrees)
    `metalOut`: float 
        camberline metal outlet angle (in degrees)
    `LEradius`: float 
        blade leading edge radius
    `Asuct`: np.ndarray
        array which stores the suction side DOF
    `Apress`: np.ndarray
        array which stores the pressue side DOF
    `wedgeAngle`: float 
        blade wedge angle (in degrees)
    '''

    # initializing data
    stagger    = x[0]
    metalIn    = x[1]
    metalOut   = x[2]
    LEradius   = x[3] 
    wedgeAngle = x[-1]
    
    # getting the rest of the data
    Asuct  = x[4:3+Nsuct] 
    Apress = x[3+Nsuct:2+Nsuct+Npress]

    return stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle

def bladeDataExtraction(
        x:        list | np.ndarray,
        Nsuct:    int   = None, 
        Npress:   int   = None,
        blade:    Blade = None,
        TEradius: float = 0.0,
        kind:     str   = 'cubic'
    ) -> tuple[interpolate._interpolate.interp1d, interpolate._interpolate.interp1d, np.ndarray, np.ndarray, float, float, float]:
    '''
    This function extracts the main coordinates of a blade under Kulfan parametrization.

    Returns
    -------
    `x`: list | np.ndarray  
        array which stores Kulfan parameters
    `Nsuct`: int 
        suction side number of degree of freedom
    `Npress`: int 
        pressure side number of degree of freedom 
    `blade`: Blade
        `geometryLIB.blade.Blade` object
    `TEradius`: float 
        blade trailing edge radius
    `kind`: str
        string which defines the interpolation properties of data
    
    Returns
    -------
    `bladeUpperLine`: interpolate._interpolate.interp1d
        upper line coordinates interpolation object
    `bladeLowerLine`: interpolate._interpolate.interp1d
        lower line coordinates interpolation object
    `camberlineLine`: interpolate._interpolate.interp1d
        camberline coordinates interpolation object
    `bladeData`: np.ndarray
        2D array which stores the blade coordinates 
    `camberlineData`: np.ndarray
        2D array which store the camberline coordinates
    `upperChord`: float
        upper side axial chord
    `lowerChord`: float
        lower side axial chord
    `camberlineChord`: float 
        camberline axial chord
    '''

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
    bladeUpperLine = interpolate.interp1d(x=bladeUpper[:, 0],     y=bladeUpper[:, 1],     kind=kind)
    bladeLowerLine = interpolate.interp1d(x=bladeLower[:, 0],     y=bladeLower[:, 1],     kind=kind)
    camberlineLine = interpolate.interp1d(x=camberlineData[:, 0], y=camberlineData[:, 1], kind=kind)
 
    return bladeUpperLine, bladeLowerLine, camberlineLine, bladeData, camberlineData, upperChord, lowerChord, camberlineChord

def camberlineFunc(
        x:           list | np.ndarray, 
        yCamberline: list | np.ndarray, 
        nPoints:     int,
        printout:    bool = False
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
    `printout`: bool
        boolean value for the printing of the root mean squared error

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

    if printout:
        print('>>> RMSE = {0:.2E}'.format(RMSE))

    return RMSE

def bladeFunc(
        x:           list | np.ndarray,
        Nsuct:       int, 
        Npress:      int,
        upperLine:   interpolate.interp1d,
        lowerLine:   interpolate.interp1d,
        TEradius:    float,
        nPoints:     int  = 200,
        printout:    bool = False,
    ) -> float:
    '''
    This function computes the error between a set of coordinates and a parametrized blade.

    Parameters
    ----------
    `x`: list | np.ndarray
        1D study vector which stores the optimization variables: [stagger, metalInlet, metalOutlet]
    `Nsuct`: int 
        number of DOF for the discretization of the suction side
    `Npress`: int 
        number of DOF for the discretization of the pressure side
    `upperLine`: interpolate.interp1d
        function object for the camberline optimization. It represents the upper side of the blade
    `lowerLine`: interpolate.interp1d
        function object for the camberline optimization. It represents the lower side of the blade 
    `TEradius`: float 
        trailing edge radius for the blade 
    `nPoints`: int 
        integer for the camberline study
    `printout`: bool
        boolean value for the printing of the root mean squared error

    Returns
    -------
    `RMSE`: float 
        root mean squared error of the computed camberline with respect to `yCamberline` 
    '''

    # getting data 
    stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle = dataExtraction(x=x, Nsuct=Nsuct, Npress=Npress)
    
    # blade generation 
    try:
        blade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)

        # getting blade coordinates with the most left point as origin
        XPS, YPS, XSS, YSS, _, _ = blade.coordinate()
        
        # checking NAN elements
        XPS, YPS = checkNAN(XPS, YPS)
        XSS, YSS = checkNAN(XSS, YSS)

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
        x = chebyschev(start=0, end=1, nPoints=nPoints)

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

    if printout:
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
        LEradiusBounds:      list        = [1E-3, 0.3],
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

def computeWedgeAngle(
        data:      list | np.ndarray, 
        upperData: list | np.ndarray, 
        lowerData: list | np.ndarray
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
    `wedgeAngle`: float
        trailing edge wedge angle 
    '''

    # wedge angle computation
    TEpoint = (data[-1, :] + data[0, :]) / 2
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

    return wedgeAngle

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
    ) -> tuple[float, float, float, np.ndarray, float]:
    '''
    This function optimizes find a suitable camberline parametrizatio for the blade.

    Parameters
    ----------
    `upperLine`: interpolate.interp1d
        function object for the camberline optimization. It represents the upper side of the blade
    `lowerLine`: interpolate.interp1d
        function object for the camberline optimization. It represents the lower side of the blade 
    `LEpos`: list | np.narray
        leading edge position coordinates
    `upperChord`: float
        upper side axial chord
    `lowerChord`: float 
        lower side axial chord
    `nPoint`: int 
        integer value for the discretization of the axial chord
    `plot`: bool
        boolean value for the plotting of the optimization results
    
    Returns
    -------
    `stagger`: float
        angle (in degrees) for the discretization of the camberline
    `metalInlet`: float 
        angle (in degrees) for the discretization of the camberline
    `metalOutlet`: float 
        angle (in degrees) for the discretization of the camberline
    `yCamberline`: np.ndarray
        2D array which stores the camberline coordinates
    `camberlineCost`: float 
        root mean squared error for the camberline optimization
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

    # cost allocation 
    camberlineCost = camberlineFunc(x=res.x, yCamberline=yCamberline, nPoints=nPoints)

    # data allocation
    stagger     = res.x[0]
    metalInlet  = res.x[1]
    metalOutlet = res.x[2]

    # camberline properties printout
    print('>>> STAGGER           = {0:+.3E}'.format(stagger))
    print('>>> METAL INLET       = {0:+.3E}'.format(metalInlet))
    print('>>> METAL OUTLET      = {0:+.3E}'.format(metalOutlet))
    print('>>> OPTIMIZATION COST = {0:+.3E}'.format(camberlineCost))

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
        ax.plot(xCamberline, yCamberline, color='r',       linestyle='-.',    linewidth=3, label='DATA.CAMBERLINE-UPER/LOWER')
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

    return stagger, metalInlet, metalOutlet, yCamberline, camberlineCost

def optimizeGeometry(
        data:      list | np.ndarray, 
        Nsuct:     int, 
        Npress:    int, 
        angle:     float = 0.0,
        LEradius:  float = 2.5e-2, 
        nPoints:   int   = 100, 
        method:    str   = 'Nelder-Mead',
        nMax:      int   = 2, 
        tol:       float = 1.4e-4,
        NsuctLow:  int   = 4, 
        NpressLow: int   = 4,
        plot:      bool  = True,
        save:      bool  = False
    ) -> np.ndarray:
    '''
    This function converts a coordinate based blade geometry into a Kulfan parametrization based geometry.
    
    Parameters
    ----------
    `data`: list | np.ndarray 
        blade geometry coordinates
    `Nsuct`: int 
        number of DOF for the discretization of the suction side
    `Npress`: int 
        number of DOF for the discretization of the pressure side
    `angle`: float 
        angle (in degrees) which allows to increase convergence 
    `LEradius`: float 
        guess for the leading edge radius in the optimization 
    `nPoints`: int 
        study points number
    `inletPos`: int 
        integer which defines the position in the array for the wedge angle computation
    `outletPos`: int 
        integer which defines the position in the array for the wedge angle computation
    `method`: str 
        optimization method used for parametrizing the blade coordinates
    `nMax`: int 
        max number of optimization. If the optimizer converges, nMax allows to optimize again starting from the best optimized parameters
    `tol`: float 
        tolerance for the blade optimization
    `NsuctLow`: int 
        integer value for the first optimization of the blade at the blade suction side
    `NpressLow`: int 
        integer value for the first optimization of the blade at the blade pressure side 
    `plot`: bool
        boolean value which allows the function to plot the optimization results
    `save`: bool
        boolean value which allows the function to save the optimization results

    Returns
    -------
    `blade`: object 
        blade object, check geometryLIB.blade object
    `kulfanParameters`: np.ndarray
        Kulfan parametrization parameters: [stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius]
    `bladeData`: np.ndarray
        2D array which stores the blade coordinate
    `cost`: float 
        optimization cost
    `fig`: plt.Figure
        figure object
    `flip`: bool
        boolean value which defines if the target blade has been flipped for the parametrization
    '''

    # moving blade into origin
    data, _, _, _ = bladeInOrigin(data, scale=True)

    # camberline analysis and flipping data analysis
    flip, __data = camberlineAnalysis(data=data, plot=False)
        
    # rotating blade geometry for the optimization
    if angle != 0:
        __data = rotate(__data, angle, resize=False)

        # moving blade into origin
        __data, _, _, _ = bladeInOrigin(__data, scale=True)

    # intepolating data 
    upperLine, lowerLine, _, _, upperData, lowerData, upperChord, lowerChord = interpolateData(__data, plot=False)

    # leading edge position
    xLE, yLE, LEradius, axialChord = bladeLEpos(upperLine=upperLine, lowerLine=lowerLine, bothSide=False, plot=False)

    # camberline approximation
    stagger, metalInlet, metalOutlet, _, camberlineCost = optimizeCamberline(upperLine=upperLine, lowerLine=lowerLine, LEpos=[xLE, yLE], upperChord=upperChord, lowerChord=lowerChord, plot=False)
    print('>>> CAMBERLINE INTERPOLATION COST = {0:.3E}'.format(camberlineCost))

    # computing wedge angle 
    wedgeAngle = computeWedgeAngle(data=__data, upperData=upperData, lowerData=lowerData)    

    # trailing edge radius computation 
    TEradius = bladeTEradius(data=__data, chord=axialChord)

    # optimize blade for lower DOF  
    print('>>> OPTIMIZE FOR LOW DEGREE OF FREEDOM')
    print('>>> Nsuct  (LOW DOF) = {0:d}'.format(NsuctLow))
    print('>>> Npress (LOW DOF) = {0:d}'.format(NpressLow))
    
    # profile line 
    Asuct  = np.concatenate([[np.sqrt(LEradius * 2) * 1.2], 0.3 * np.ones((NsuctLow  - 3,)), [0.15]])
    Apress = np.concatenate([[np.sqrt(LEradius * 2) * 1.2], 0.3 * np.ones((NpressLow - 3,)), [0.15]])

    # setting up initial guess
    x = [
        stagger,
        metalInlet,
        metalOutlet, 
        LEradius
    ] 

    # setting up total guess array
    x = np.concatenate((x, Asuct, Apress, [wedgeAngle]))
    print('>>> GUESS VECTOR FOR LOW DOF OPTIMIZATION = {0} '.format(np.array2string(x)))

    # arguments generation
    args = (
        NsuctLow, 
        NpressLow,
        upperLine,
        lowerLine,
        TEradius,
        nPoints,
        False
    )

    # boundaries generation    
    bounds = boundsGenerator(stagger=stagger, metalInlet=metalInlet, metalOutlet=metalOutlet, Nsuct=NsuctLow, Npress=NpressLow)

    # checking camberline properties: if RMSE/cost computed -> good to go
    cost = bladeFunc(x, NsuctLow, NpressLow, upperLine, lowerLine, TEradius, nPoints)
    print('>>> INITIAL RMSE = {0:.3E}'.format(cost))
        
    # optimizing blade
    res = optimize.minimize(fun=bladeFunc, x0=x, args=args, method=method, bounds=bounds)
        
    # allocating data
    x = res.x
        
    # getting final low DOF cost
    cost = bladeFunc(x, NsuctLow, NpressLow, upperLine, lowerLine, TEradius, nPoints)
    print('>>> LOW DOF OPTIMIZATION => RMSE = {0:.3E}'.format(cost))

    # blade object generation 
    stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle = dataExtraction(x=res.x, Nsuct=NsuctLow, Npress=NpressLow)
    
    # blade object
    lowDOFblade = Blade(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=1.0, pitch=1.0, Asuct=Asuct, Apress=Apress, LEradius=LEradius, TEradius=TEradius, wedgeAngle=wedgeAngle, origin=True)
    
    # blade scaling and data allocation
    Asuct, Apress, LEradius = lowDOFblade.scale(Nsuct=Nsuct, Npress=Npress, plot=False)
    
    # setting up initial guess
    x = [
        stagger,
        metalInlet,
        metalOutlet, 
        LEradius
    ] 

    # setting up total guess array
    x = np.concatenate((x, Asuct[1:-1], Apress[1:-1], [wedgeAngle]))
    print('>>> GUESS VECTOR = {0} '.format(x))

    # arguments generation
    args = (
        Nsuct, 
        Npress,
        upperLine,
        lowerLine,
        TEradius,
        nPoints,
        False
    )

    # boundaries generation    
    bounds = boundsGenerator(stagger=stagger, metalInlet=metalInlet, metalOutlet=metalOutlet, Nsuct=Nsuct, Npress=Npress)

    # checking camberline properties: if RMSE/cost computed -> good to go
    cost = bladeFunc(x, Nsuct, Npress, upperLine, lowerLine, TEradius, nPoints, True)
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
        fig.suptitle(r'$\Theta = $' + '{0:.2f}'.format(angle) + r'$^{\circ}$' + ' || ' + r'$cost = $' + '{0:.2E}'.format(cost))
        ax = fig.add_subplot(1,1,1)

        # getting from optimization's result
        _, _, _, bladeData, _, upperChord, lowerChord, _ = bladeDataExtraction(xVal, Nsuct, Npress, TEradius=TEradius, kind='linear')
        ax.plot(bladeData[:,0] / lowerChord, bladeData[:,1] / lowerChord, 'k', linewidth=5, label='RESULT.BLADE')
    
        # plotting coordinates
        plotCoords(__data, ax, theta=0, resize=False, base=True)
        
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')  
        
        # saving figure
        if not save:
            plt.show()

        bladeData = rotate(bladeData, -angle, resize=False)
    else:
        fig = None 
        _, _, _, bladeData, _, upperChord, lowerChord, _ = bladeDataExtraction(xVal, Nsuct, Npress, TEradius=TEradius, kind='linear')
        bladeData = rotate(bladeData, -angle, resize=False)
        
    return blade, kulfanParameters, bladeData, cost, fig, flip

def optimizeBlade(
        data:       list | np.ndarray, 
        Nsuct:      int, 
        Npress:     int, 
        angle:      float = 0.0,
        deltaAngle: float = 2.5,
        LEradius:   float = 2.5e-2, 
        nPoints:    int   = 100,
        method:     str   = 'Nelder-Mead',
        nMax:       int   = 2, 
        tol:        float = 1.2e-4,
        plot:       bool  = True,
        save:       bool  = False
    ) -> tuple[Blade, np.ndarray, np.ndarray, float, plt.Figure, float]:
    '''
    Parameters
    ----------
    `data`: list | np.ndarray 
        blade geometry coordinates
    `Nsuct`: int 
        number of DOF for the discretization of the suction side
    `Npress`: int 
        number of DOF for the discretization of the pressure side
    `angle`: float 
        angle (in degrees) which allows to increase convergence 
    `LEradius`: float 
        guess for the leading edge radius in the optimization 
    `nPoints`: int 
        study points number
    `inletPos`: int 
        integer which defines the position in the array for the wedge angle computation
    `outletPos`: int 
        integer which defines the position in the array for the wedge angle computation
    `method`: str 
        optimization method used for parametrizing the blade coordinates
    `nMax`: int 
        max number of optimization. If the optimizer converges, nMax allows to optimize again starting from the best optimized parameters
    `tol`: float 
        tolerance for the blade optimization
    `NsuctLow`: int 
        integer value for the first optimization of the blade at the blade suction side
    `NpressLow`: int 
        integer value for the first optimization of the blade at the blade pressure side 
    `plot`: bool
        boolean value which allows the function to plot the optimization results
    `save`: bool
        boolean value which allows the function to save the optimization results

    Returns
    -------
    `blade`: object 
        blade object, check geometryLIB.blade object
    `kulfanParameters`: np.ndarray
        Kulfan parametrization parameters: [stagger, metalIn, metalOut, LEradius, Asuct, Apress, wedgeAngle, TEradius]
    `bladeData`: np.ndarray
        2D array which stores the blade coordinate
    `cost`: float 
        optimization cost
    `fig`: plt.Figure
        figure object
    `flip`: bool
        boolean value which defines if the target blade has been flipped for the parametrization
    `angle`: float 
        angle (in degrees) where the best parameters for the blade coordinates have been optimized
    '''

    if angle != 0: 
        blade, kulfanParameters, bladeData, cost, fig, flip = optimizeGeometry(data=data, Nsuct=Nsuct, Npress=Npress, angle=angle, LEradius=LEradius, nPoints=nPoints, method=method, nMax=nMax, tol=tol, plot=plot, save=save)
        print('>>> OPTIMIZATION COST = {0:+.3E}'.format(cost))
    else:
        # setting up initial data
        cost = np.Inf

        # looping over angle and cost
        while cost > tol and angle < 20:
            # storing cost 
            costTemp = cost

            # blade computation
            bladeTemp, kulfanParametersTemp, bladeDataTemp, cost, figTemp, flip = optimizeGeometry(data=data, Nsuct=Nsuct, Npress=Npress, angle=angle, LEradius=LEradius, nPoints=nPoints, method=method, nMax=nMax, tol=tol, plot=plot, save=save)
            
            print('>>> OPTIMIZING FOR THETA = {0:.2f}'.format(angle))
            print('>>> OPTIMIZATION COST    = {0:+.3E}'.format(cost))
            
            # updating cost 
            if cost < costTemp: 
                # saving data 
                blade            = bladeTemp
                bladeData        = bladeDataTemp
                kulfanParameters = kulfanParametersTemp
                fig              = figTemp 
                if flip:
                    bladeData[:,1] = - bladeData[:,1]

            # angle updating
            angle = angle + deltaAngle

        angle = angle - deltaAngle

    return blade, kulfanParameters, bladeData, cost, fig, angle
