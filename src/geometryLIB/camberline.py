#!/usr/bin/env python3
import warnings
import numpy       as     np 
from   matplotlib  import pyplot as plt 

class Camberline():
    '''
    Camberline class. This class is built using the **Kulfan formulation** :cite:p:`kulfan2008universal`.
    '''

    def __init__(
            self, 
            stagger:    float, 
            metalIn:    float, 
            metalOut:   float,
            chord:      float = 1.0, 
            nPoints:    int   = 100, 
            chebyschev: bool  = False,
            origin:     bool  = True
        ) -> None:
        '''
        Camberline initialization: main quantities. Following :cite:p:`kulfan2008universal`: 
            * `n` computation
            * `a` computation 
            * `b` computation 

        Parameters
        ----------
        `stagger`: float
            camberline stagger angle in degrees
        `metalIn`: float 
            inlet metal angle in degrees
        `metalOut`: float 
            outlet metal angle in degrees
        `chord`: float 
            axial chord for the airfoil
        `nPoints`: int
            number of discretization points for the camberline
        `chebyschev`: bool 
            boolean value for the allocation of the camberline points
        `origin`: bool
            boolean value which computes the leading edge of the camberline at the origin

        '''

        # checking input data
        self.__inputCheck(stagger, metalIn, metalOut, chord, nPoints, chebyschev, origin)

        # allocating data
        self.stagger    = stagger
        self.metalIn    = metalIn
        self.metalOut   = metalOut
        self.chord      = chord
        self.nPoints    = nPoints 
        self.chebyschev = chebyschev
        self.origin     = origin
        
        # setting up line variables' properties
        self.__n()
        self.__a()
        self.__b()
        
        # setting up x coords and computing y coords
        self.__x()
        self.__y()
        self.__scale()

        # computing normal and tangent vectors
        self.__vector()

    # checking inputs
    def __inputCheck(
            self, 
            stagger:    float, 
            metalIn:    float, 
            metalOut:   float, 
            chord:      float, 
            nPoints:    int, 
            chebyschev: bool, 
            origin:     bool
        ) -> None:
        '''
        Checking if the object properties are correct. 
        
        Parameters
        ----------
        `stagger`: float
            camberline stagger angle in degrees
        `metalIn`: float 
            inlet metal angle in degrees
        `metalOut`: float 
            outlet metal angle in degrees
        `chord`: float 
            axial chord for the airfoil
        `nPoints`: int
            number of discretization points for the camberline
        `chebyschev`: bool 
            boolean value for the allocation of the camberline points
        `origin`: bool
            boolean value which computes the leading edge of the camberline at the origin

        '''

        # checking input types 
        if not (isinstance(stagger, float) or isinstance(stagger, int)):
            raise TypeError('The stagger angle must be a float or a int. Type = {0}'.format(type(stagger)))

        if not (isinstance(metalIn, float) or isinstance(metalIn, int)):
            raise TypeError('The inlet metal angle must be a float or a int. Type = {0}'.format(type(metalIn)))
        
        if not (isinstance(metalOut, float) or isinstance(metalOut, int)):
            raise TypeError('The outlet metal angle must be a float or a int. Type = {0}'.format(type(metalOut)))

        if not (isinstance(chord, float) or isinstance(chord, int)):
            raise TypeError('The chord value must be a float or a int. Type = {0}'.format(type(chord)))

        if not isinstance(nPoints, int):
            raise TypeError('The number of discretization point has to be an integer. Type: {0}'.format(type(nPoints)))
        
        if not isinstance(chebyschev, bool):
            raise TypeError('The chebyschev value must be a boolean. Type = {0}'.format(type(chebyschev)))
        
        if not isinstance(origin, bool):
            raise TypeError('The origin value must be a boolean. Type = {0}'.format(type(chebyschev)))
        
        if chord <= 0: 
            raise ValueError('Chord value, chord, must be greater than 0. chord = {0}'.format(chord))

        if nPoints <= 2:
            raise ValueError('nPoints value must be grater than 2. nPoints = {0}'.format(nPoints))

        if stagger == 0:
            warnings.warn('::: Warning: the input stagger angle is not compatible with the Kulfan parametrization. To avoid this problem, the camberline is sketch as a straight line.'.format(stagger))

    def __xCheck(self) -> None:
        '''
        This function checks the normalized `x` distribution in the object. `x` must be bounded between [0, 1].
        
        Returns
        -------
        : None
        
        '''

        if any(self.x) > 1 or any(self.x) < 0:
            raise ValueError('::: The normalized axial chord does not respect the bounds [0, 1].')

    # setting up camberline parameters
    def __n(self) -> None:
        '''
        This function computes the `n` value used in the **Kulfan parametrization** :cite:p:`kulfan2008universal`. It uses the `camberline` object properties.  
        ''' 

        if self.stagger == 0:
            self.n = 0
            warnings.warn('::: n value in the Kulfan parametrization is set as 0.')
        else:
            self.n = (np.tan(np.deg2rad(self.metalOut)) + np.tan(np.deg2rad(self.metalIn))) / np.tan(np.deg2rad(self.stagger))

        if self.n < 0:
            print('>>> STAGGER   = {0:.2E}'.format(self.stagger))
            print('>>> METAL IN  = {0:.2E}'.format(self.metalIn))
            print('>>> METAL OUT = {0:.2E}'.format(self.metalOut))
            raise ValueError('The camberline cannot be computed: ``self.n`` = {self.n} <= 0')

    def __a(self) -> None:
        '''
        This function computes the ``a`` value used in the **Kulfan parametrization** :cite:p:`kulfan2008universal`. It uses the ``camberline`` object properties.  
        ''' 

        if self.n == 0:
            self.a = 0
            warnings.warn('::: a value in the Kulfan parametrization is set as 0.')
        else:
            self.a = np.tan(np.deg2rad(self.metalOut)) / self.n 

    def __b(self) -> None:
        '''
        This function computes the ``n`` value used in the **Kulfan parametrization** :cite:p:`kulfan2008universal`. It uses the ``camberline`` object properties.  
        ''' 
        
        if self.n == 0:
            self.b = 0
            warnings.warn('::: b value in the Kulfan parametrization is set as 0.')
        else:
            self.b = - np.tan(np.deg2rad(self.metalIn)) / self.n

    # setting up camberline formulation
    def __x(self) -> None:
        '''
        This function computes the `x` coordinates points distribution using Chebyshev nodes or a linear distribution. It automatically updates the `y` coordinates of the camberline.

        Returns
        -------
        : None
        '''

        # computing Chebyshev points 
        if self.chebyschev:
            x = (1 - np.cos((2 * (np.arange(self.nPoints) + 1) - 1) / (2 * self.nPoints) * np.pi)) / 2    
            # adding extremes
            self.x = np.concatenate(([0], x, [1]))
        else:
            self.x = np.linspace(0, 1, self.nPoints)

    def __y(self) -> None:
        '''
        This function computes the `y` coordinate components of the camberline starting from `x` values which represent the percentage in chord position. 

        Returns
        -------
        : None  
        '''

        # checking input
        self.__xCheck()

        # computing normalized y coordinates for the camberline  
        self.y = self.a * self.x**self.n + self.b * (1 - self.x)**self.n

        if self.origin:
            self.y = self.y - self.y[0]

    def __scale(self) -> None:
        '''
        This function scale the `x` and `y` coordinates points with respect to the actual camberline chord.

        Parameters
        ----------
        `printout`: bool
            boolean for printing nodes coordinates 

        Returns
        -------
        `None`
        '''

        # scaling data
        self.X = self.chord * self.x 
        self.Y = self.chord * self.y

    # setting up tangent/normal vectors 
    def __tangent(self) -> np.ndarray:
        '''
        This function computes the tangent to the camberline.

        Returns 
        -------
        `T`: np.ndarray 
            tangent value at `x` position. 

        '''

        # computing tangent
        if self.n == 0:
            T = np.zeros(self.x.shape)
        else:
            T = self.a * self.n * self.x**(self.n - 1) - self.b * self.n * (1 - self.x)**(self.n - 1)
            
        return T
    
    def __vector(self) -> None:
        '''
        This function computes the tangent and normal vectors to the camberline.
        
        Returns
        -------
        : None 

        '''
      
        # computing tangent 
        T = self.__tangent()
        tangentLength = np.sqrt(1 + T**2)

        # allocating data
        self.tangent = np.stack((1 / tangentLength, T / tangentLength), axis=1)
        self.normal  = np.stack((- T / tangentLength, 1 / tangentLength), axis=1)

    # updating camberline 
    def compute(self) -> None:
        '''
        This function updates the ``camberline`` object properties and recomputed, at the end of the allocation, the new camberline shape.

        Parameters
        ----------
        `stagger`: float
            camberline stagger angle in degrees
        `metalIn`: float 
            inlet metal angle in degrees
        `metalOut`: float 
            outlet metal angle in degrees
        `chord`: float 
            axial chord for the airfoil
        `nPoints`: int
            number of discretization points for the camberline
        `chebyschev`: bool 
            boolean value for the allocation of the camberline points
        
        '''

        # checking input data
        self.__inputCheck(self.stagger, self.metalIn, self.metalOut, self.chord, self.nPoints, self.chebyschev, self.origin)

        # setting up line variables' properties
        self.__n()
        self.__a()
        self.__b()
        
        # setting up x coords and computing y coords
        self.__x()
        self.__y()
        self.__scale()

        # computing normal and tangent vectors
        self.__vector()

    # local variables
    def coordinatePoint(self, x: float or list or np.ndarray, scale: bool, printout: bool = False) -> float or np.ndarray: 
        '''
        This function computes the `y` coordinate component of the camberline starting from a `x` value which represents the percentage in chord position. 
        
        Parameters
        ----------
        `x`: float or list or np.ndarray
            camberline `x` position normalized by the chord. 
        `scale`: bool 
            boolean value for scaling the blade properties.
        `printout`: bool 
            boolean value for the point properties printout.

        Returns
        -------
        `x`: float | list | np.ndarray 
            camberline `x` coordinate component. 
        `y`: float | list | np.ndarray 
            camberline `y` coordinate component. 
        `X`: float | list | np.ndarray 
            camberline `x` coordinate component scaled by the chord. 
        `Y`: float | list | np.ndarray 
            camberline `y` coordinate component scaled by the chord. 
        '''

        if isinstance(x, np.ndarray) or isinstance(x, list):
            if any(x < 0) or any(x > 1):
                raise ValueError('::: The normalized input data is out the working boundaries [0, 1].')
        else:
            if x < 0 or x > 1:
                raise ValueError('::: The normalized input data is out the working boundaries [0, 1].')
        
        # computing camberline 
        y = self.a * x**self.n + self.b * (1 - x)**self.n

        if scale: 
            X = x * self.chord
            Y = y * self.chord
        else:
            X = x 
            Y = y
            
        if printout:
            print('>>> POINT ANALYSIS:')
            print('    >>> a = {0:+.3E}'.format(self.a))
            print('    >>> n = {0:+.3E}'.format(self.n))
            print('    >>> b = {0:+.3E}'.format(self.b))
            print('    >>> x NORMALIZED = {0}'.format(x))
            print('    >>> y NORMALIZED = {0}'.format(y))
            print('    >>> x SCALED     = {0}'.format(X))
            print('    >>> y SCALED     = {0}'.format(Y))

        return x, y, X, Y

    def tangentVector(self, x: float or list or np.ndarray) -> np.ndarray:
        '''
        This function computes the tangent vector to the camberline.

        Parameters
        ----------
        `x` (np.array): camberline `x` position/s normalized by the chord. 
        
        Returns
        -------
        tangent vector/s (np.array): tangent value at `x` position. 

        '''

        if isinstance(x, np.ndarray) or isinstance(x, list):
            if any(x < 0) or any(x > 1):
                raise ValueError('::: The normalized input data is out the working boundaries [0, 1].')
        else:
            if x < 0 or x > 1:
                raise ValueError('::: The normalized input data is out the working boundaries [0, 1].')

        # computing tangent properties
        if self.n == 0:
            T = 0
        else:
            T = self.a * self.n * x**(self.n - 1) - self.b * self.n * (1 - x)**(self.n - 1)

        tangentLength = np.sqrt(1 + T**2)

        # vector generation
        if isinstance(x, np.ndarray) or isinstance(x, list):
            tangent = np.stack((1 / tangentLength, T / tangentLength), axis=1)
        else: 
            tangent = np.array([1 / tangentLength, T / tangentLength])  
            
        return tangent

    def normalVector(self, x: float or list or np.ndarray) -> np.ndarray:
        '''
        This function computes the normal vector to the camberline.

        Paramters
        ---------
        `x`: np.array 
            camberline `x` position/s normalized by the chord. 
        
        Returns
        -------
        normal vector/s: np.array 
            normal value at `x` position. 

        '''        
        if isinstance(x, np.ndarray) or isinstance(x, list):
            if any(x < 0) or any(x > 1):
                raise ValueError('::: The normalized input data is out the working boundaries [0, 1].')
        else:
            if x < 0 or x > 1:
                raise ValueError('::: The normalized input data is out the working boundaries [0, 1].')

        # computing tangent properties
        if self.n == 0:
            T = 0
        else:
            T = self.a * self.n * x**(self.n - 1) - self.b * self.n * (1 - x)**(self.n - 1)
            
        tangentLength = np.sqrt(1 + T**2)

        # vector generation
        if isinstance(x, np.ndarray) or isinstance(x, list):
            normal = np.stack((- T / tangentLength, 1 / tangentLength), axis=1)
        else: 
            normal = np.array([- T / tangentLength, 1 / tangentLength])   
            
        return normal

    # setting up points distance 
    def __computeLength(self, x0: float, x1: float, Npoints: int) -> float:
        '''
        This function computes the length between 2 points (defined in chord percentage `x0` and `x1`) belonging to the camberline.

        Parameters
        ----------
        `x0`: float 
            start point that belongs to the camberline `x` position/s normalized by the chord.
        `x1`: float 
            end point that belongs to the camberline `x` position/s normalized by the chord.
        `Npoints`: int
            number of discretization points between `x0` and `x1`. 

        Returns
        -------
        `linelength`: float
            line length 
        '''
        if x0 > x1 or x0 < 0 or x1 > 1 or Npoints <= 0:
            raise ValueError('Check x0 and x1:\n\tx0 < x1\n\t0 <= x0 < 1\n\t0 < x1 <= 1\nCheck Npoints:\n\tN > 0')
        else:
            # vector generation 
            xVec       = np.linspace(x0, x1, Npoints)
            lineLength = 0

            for ii in range(Npoints - 1): 
                lineLength = lineLength + np.sqrt((xVec[ii+1] - xVec[ii])**2 + (self.yCamberPoint(xVec[ii+1]) - self.yCamberPoint(xVec[ii]))**2)

        return lineLength

    def __curveRefinement(self, pointsInUnit: int = 150, Npoints: int = 3, printout: bool = False) -> float:
        r'''
        This function automatically refines the camberline. It adds points in order to keep the distance between each consecutive points lower than `1/pointsInUnit`. 

        Parameters
        ----------
        `pointsInUnit`: int 
            number of discretization points in a unit length curve. If `pointsInUnit = 200` a line long 2 will be discretized by 400 points. Every discretization point along the curve will have a distance such that: :math:`\sqrt{(x_1 - x_0)^2 + (y_{(x_1)} - y_{(x_0)})^2} \leq \frac{1}{200}`.
        `Npoints`: int 
            number of points that will be used to discretizate the line between 2 consecutive x.
        `printout`: bool
            printout boolean value for printing out computed points.

        Returns
        -------
        : int 
            number of points used for the line discretization. 

        '''
        from scipy.optimize import bisect
        
        # storing vectors
        xVec   = [0.0]
        lenVec = [0.0]

        # compute total length 
        totalLength = 0.0 
        for ii in range(len(self.x) - 1):
            totalLength = totalLength + self.computeLength(x0=self.x[ii], x1=self.x[ii+1], Npoints=Npoints)

        # each segment length
        length = totalLength / (pointsInUnit - 1)

        prevX = 0.0
        # looping over all the profile line
        for _ in range(pointsInUnit - 2):
            # gussing new X
            X = prevX + 1 / (pointsInUnit - 1) 
        
            # updating starting point in the length computing function
            lengthFunc = lambda x: self.computeLength(x0=prevX, x1=x, Npoints=Npoints) - length

            # checking if line end is not 
            if X >= 1:
                X = 1.0

            # computing delta length 
            deltaLength = self.computeLength(x0=prevX, x1=X, Npoints=Npoints) - length

            # setting b point in bisection function
            if deltaLength > 0:
                res = bisect(f=lengthFunc, a=prevX, b=X)
            else: 
                res = bisect(f=lengthFunc, a=prevX, b=1.0)

            # storing length into vector
            segmentLen = self.computeLength(x0=prevX, x1=X, Npoints=5) 
            lenVec.append(segmentLen)
            
            if printout:
                print('X             = {0:.3E}'.format(X))
                print('deltaLength   = {0:.3E}'.format(deltaLength))
                print('length target = {0:.3E}'.format(length))
                print('total length  = {0:.3E}'.format(totalLength))
                print('res           = {0:.3E}'.format(res))
                print('length        = {0:.3E}'.format(segmentLen))

            # setting up starting point 
            prevX = res
            
            # updating storing vector
            xVec.append(res)

        # setting the last point  
        xVec.append(1.0)

        # setting up the last segment length
        segmentLen = self.computeLength(prevX, 1.0, Npoints=5) 
        lenVec.append(segmentLen)

        # allocating points distribution 
        self.x = np.array(xVec)
        
        # recomputing data 
        self.yCamberLine(self.x)

        # allocating length vector 
        self.length = np.array(lenVec)

        # allocating total profile length 
        self.totalLength = sum(lenVec) 
        print('>>> SURFACE TOTAL LENGTH = {0:.3E}'.format(self.totalLength))

        return len(xVec)

    def rotate(self, theta: float | int) -> tuple:
        '''
        This function rotates the profile line coordinates by a theta angle (in degrees).

        Parameters
        ----------
        `theta`: float | int 
        '''

        # main values computation
        cos = np.cos(np.deg2rad(theta))
        sin = np.sin(np.deg2rad(theta))

        # rotation matrix generation
        rotMatrix = np.array([[cos, -sin], [sin, cos]])

        # data allocation
        dataNormalized = np.stack((self.x, self.y), axis=1)
        dataReal       = np.stack((self.X, self.Y), axis=1)

        # rotating data 
        for ii, _ in enumerate(dataNormalized):
            coord = np.matmul(rotMatrix, dataNormalized[ii, :]) 
            dataNormalized[ii, :] = coord / cos

            coord = np.matmul(rotMatrix, dataReal[ii, :])
            dataReal[ii, :] = coord / cos

        return dataNormalized[:, 0], dataNormalized[:, 1], dataReal[:, 0], dataReal[:, 1]
    
    # saving data 
    def save(self, fileName: str, saveInZero: bool = False) -> None:
        '''
        This function saves the camberline data inside a text file. 
        The saved data are:
            * `x`
            * `y`
            * `tangent`
            * `normal`

        Parameters
        ----------
        `fileName`: str
            name of the file where data will be stored.
        `saveInZero`: bool 
            enables saving the blade camberline with the leading edge at (0, 0)

        '''

        header = '{0:>5}{1:>14}{2:>17}{3:>14}{4:>13}{5:>14}'.format('x', 'y', 'x_tangent', 'y_tangent', 'x_normal', 'y_normal')

        with open(file=fileName, mode='w') as f:
            if saveInZero:
                np.savetxt(f, np.column_stack([self.x, self.y - self.y[0], self.tangent, self.normal]), fmt='%+.5E', delimiter="  ", header=header)
            else:
                np.savetxt(f, np.column_stack([self.x, self.y, self.tangent, self.normal]), fmt='%+.5E', delimiter="  ", header=header)

    # plotting data
    def plot(
            self, 
            ax:         plt.Axes,
            normalized: bool     = True,
            color:      str      = 'k', 
            marker:     str      = ' ', 
            mfc:        str      = 'k', 
            mec:        str      = 'k', 
            ms:         int      = 3, 
            label:      str      = 'camberline', 
            vector:     bool     = False,
            linewidth:  float    = 2,
            pitch:      float    = 0.0,
            number:     int      = 1
        ) -> None:
        '''
        This function plots the camberline.

        Parameters
        ----------
        `ax`: matplotlib.pyplot.axes
            plt.axes object 
        `color`: str
            line color
        `marker`: str
            marker style 
        `mfc`: str 
            marker foreground color 
        `mec`: str
            marker edge color 
        `ms`: int 
            marker size 
        `label`: str
            curve label
        `vector`: bool
            tangent and normal vectors
        `linewidth`: float 
            camberline linewidth in plot
        `plotInZero`: bool
            plots the camberline with the leading edge at (0, 0)
        `pitch`: float
            enables plotting mumtiple blades with setting up the blade pitch 
        `number`: int
            number of camberlines to plot if pitch > 0
        '''

        if not isinstance(number, int):
            raise TypeError('::: The number of blades to plot must be an integer (greater than 0).')
        elif number <= 0:
            raise ValueError('::: The number of blades to plot must be greater than 0.')

        if not isinstance(pitch, float or int):
            raise TypeError('::: The pitch value must be an integer or a float (greater than 0).')
        elif pitch < 0:
            raise ValueError('::: The pitch value must be greater than 0.')

        # plotting data
        if normalized:
            pitch = pitch / self.chord
            for ii in range(number):
                ax.plot(self.x, self.y + ii*pitch, color=color, marker=marker, mfc=mfc, ms=ms, mec=mec, linewidth=linewidth, label=label)
        else:
            for ii in range(number):
                ax.plot(self.X, self.Y + ii*pitch, color=color, marker=marker, mfc=mfc, ms=ms, mec=mec, linewidth=linewidth, label=label)

        if vector:
            if normalized:
                ax.plot(self.x, self.y, 'k', linewidth=1.5, alpha=0.7)
                ax.quiver(self.x, self.y, self.tangent[:, 0], self.tangent[:, 1], color='b')
                ax.quiver(self.x, self.y, self.normal[:, 0], self.normal[:, 1], color='r')
            else:
                ax.plot(self.X, self.Y, 'k', linewidth=1.5, alpha=0.7)
                ax.quiver(self.X, self.Y, self.tangent[:, 0], self.tangent[:, 1], color='b')
                ax.quiver(self.X, self.Y, self.normal[:, 0], self.normal[:, 1], color='r')