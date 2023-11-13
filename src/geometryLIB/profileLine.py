#!/usr/bin/env python3
import time 
import numpy          as     np 
from   matplotlib     import pyplot as plt 
from   enum           import Enum, auto 
from   src.geometryLIB.camberline     import Camberline
from   scipy.optimize import bisect

class Side(Enum):
    '''
    This object defines if the position of the ProfileLine object represents the upper side of the blade or the lower side of the blade.
    '''

    SS = auto()
    PS = auto()

class ProfileLine():
    '''
    Profile line object following **Kulfan parametrization**. 
    In this case the *profile line* stands for the **suction** or the **pressure side** of the airfoil starting from the **leading edge** generated by the camberline and ending at its **trailing edge**.
    '''

    def __init__(
            self, 
            A:          np.ndarray | list,
            LEradius:   float = None, 
            wedgeAngle: float = None,  
            TEradius:   float = 0.0, 
            position:   Side  = Side.PS, 
            Cinputs:    np.ndarray | list = [0.5, 1.0],
        ) -> None:
        '''
        Profile line initialization, main quantities. Following :cite:p:`kulfan2008universal`.

        Parameters
        ----------
        `A`: np.array
            array that defines the line parametrization 
        `LEradius`: float 
            leading edge radius. Set as `None`. If set as a number greater than 0, the `A` vector is updated with the respective leading edge value in Kulfan parametrization.  
        `wedgeAngle`: float
            set as `None`. If it is not `None`, the wedge angle will affect the last term of the profile line. It adds a DOF to the profile line.
        `position`: bool 
            boolean value that defines if the profile line is referred to the suction or pressure side of the airfoil:
                * `position = Side.PS` stands for the pressure side of the blade 
                * `position = Side.SS` stands for the suction side of the blade
        `TEradius`: float 
            trailing edge radius. Set as 0.0.  
        `Cinputs`: np.array
            array that stores the **Kulfan** shape space descriptor for the airfoil. Most of the aeronautical airfoils are described by `Cinput = [0.5, 1.0]` 
             
        '''

        # checking inputs
        self.__checkInput(A=A, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=position, Cinputs=Cinputs)
        
        # allocating traling edge radius
        self.TEradius = TEradius
        
        # allocating profile line position
        self.position = position

        # allocating profile line shape function properties 
        self.Cinputs = Cinputs

        # allocating data
        # leading edge radius study
        if LEradius != None:
            self.LEradius = LEradius 
            # computing Kulfan's representation of the leading edge radius in the formulation
            A0 = np.sqrt(LEradius * 2)
            # updating Kulfan parameters 
            A = np.concatenate(([A0], A))
        else:
            # computing back leading edge radius from the Kulfan parameters
            self.LEradius = A[0]**2 / 2

        # trailing edge radius study
        if wedgeAngle != None:
            self.wedgeAngle = wedgeAngle 
            # computing Kulfan's representation of the wedge angle in the formulation
            Alast = np.tan(np.deg2rad(self.wedgeAngle)) + self.TEradius
            # updating Kulfan parameters 
            A = np.concatenate((A, [Alast]))
        else:
            # computing back the wedge angle value from the Kulfan parameters
            self.wedgeAngle = np.rad2deg(np.arctan(A[-1] - self.TEradius))

        # allocating Kulfan parameters
        self.A = A
        
        # allocating parameters counter
        self.N = len(self.A) - 1

    def __checkInput(
            self, 
            A:          np.ndarray, 
            LEradius:   float, 
            wedgeAngle: float | None, 
            TEradius:   float, 
            position:   Side,
            Cinputs:    np.ndarray | list 
        ) -> None:
        '''
        This program checks the input for the generation of the profile line object. 
        
        Parameters
        ----------
        `LEradius`: float 
            leading edge radius 
        `A`: np.ndarray
            numpy array that defines the values of the Kulfan parametrization
        `wedgeAngle`: float 
            float value that defines the wedge/boat angle of the profile line
        `TEradius`: float 
            traling edge radius 
        `position`: Side 
            class that defines is the profile line defines the pressure side or the suction side of the blade
        `Cinput`: np.ndarray or list 
            values that define the behaviour of the class function for the profile line representation 
        '''

        if not isinstance(A, (list, np.ndarray)):
            raise TypeError('The blade is parametrized by a list or a numpy array.')

        if isinstance(LEradius, (float, int)):
            if LEradius < 0:
                raise ValueError('The leading edge parameter has to be greater than 0.')
        elif LEradius != None:
            raise TypeError('The leading edge parameter must be a float or an int or `None`.')

        if isinstance(wedgeAngle, (float, int)):
            if wedgeAngle < 0:
                raise ValueError('The wedge angle parameter has to be greater than 0.')
        elif wedgeAngle != None:
            raise TypeError('The wedge angle parameter must be a float or an int or `None`.')
        
        if isinstance(TEradius, (float, int)):
            if TEradius < 0:
                raise ValueError('The trailing edge parameter has to be greater than 0.')
        else:
            raise TypeError('The trailing edge parameter must be a float or an int.')
        
        if not isinstance(Cinputs, (list, np.ndarray)):
            raise TypeError('The Bernstein curves are shaped by a list or a numpy array.')
        else:
            if len(Cinputs) != 2:
                raise ValueError('The shape function must be parametrized by a 2 element array.')

        if not isinstance(position, Side):
            raise TypeError('The profile line descriptor is wrong. The profile line position is based on a Enum object.\nThis object has to be imported inside the code using:\n>>> from kulfanLIB.profileline import Side\n>>> pLine = ProfileLine(..., Side.PS, ...)')

    def __bernstein(self, x: float, i: int, N: int) -> float:
        '''
        Bernstein polynomial function following :cite:p:`kulfan2008universal`. 

        Parameters
        ----------
        `x`: float 
            study position, in chord percentage.
        `i`: int
            series index. 
        `N`: int 
            total DOF for the Bernstein formulation.
        '''
        
        K = np.math.factorial(N) / (np.math.factorial(i) * np.math.factorial(N - i))
        
        return K * x**i * (1 - x)**(N - i)

    def __C(self, x: float) -> float:
        '''
        Closure function following :cite:p:`kulfan2008universal`. 

        Parameters
        ----------
        `x`: float 
            study position, in chord percentage.
        '''
        return x**self.Cinputs[0] * (1 - x)**self.Cinputs[1]

    def __S(self, x: float, printout: bool = False) -> float:
        '''
        Profile definition following :cite:p:`kulfan2008universal`.

        Parameters
        ----------
        `x`: float 
            study position, in chord percentage.
        '''

        # initializing data
        Sval = 0.0 

        # computing thickness values related only to A values without C(x) thickness distribution
        for i in range(self.N + 1):
            Sval = Sval + self.A[i] * self.__bernstein(x, i, self.N) 

            if printout:
                print('>>> S({0:.2E}, {1:d}, {2:d}) = {3:+.3E}'.format(x, i, self.N, self.A[i] * self.__bernstein(x, i, self.N)))
    
        if printout:
            print('-----------------------------------')
            print('>>> PROFILE LINE DOF = {0:d}'.format(self.N + 1))
            print('>>> Sval             = {0:+.3E}\n'.format(Sval))

        return Sval 
    
    def __thickness(self, x: float) -> float:
        '''
        Thickness function following :cite:p:`kulfan2008universal`.

        Parameters
        ----------
        `x`: float 
            study position, in chord percentage.
        '''
        return self.__C(x) * self.__S(x) 

    def __thicknessDistribution(self, camberline: Camberline, x: list | np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Thickness distribution following :cite:p:`clark2019step`. 

        Parameters
        ----------
        `camberline`: kulfanLIB.camberline.Camberline
            camberline object.
        `x`: float 
            study position, in chord percentage.

        Results
        -------
        `X`: float 
            horizontal displacement with respect to the `x` camberline position.
        `Y`: float 
            vertical displacement with respect to the `y` camberline position. The `y` camberline position is relative to `x`. 

        '''

        # allocating data
        if not isinstance(x, (list, np.ndarray)):
            x      = camberline.x
            normal = camberline.normal
        elif any(x < 0) or any(x > 1):
            raise ValueError('::: x must be a list or a np.ndarray, bounded between [0, 1].')
        else:
            normal = camberline.normalVector(x=x)

        # allocating data
        B         = self.__bernstein(x, 0, 2)
        thickness = self.__thickness(x)  

        if self.position == Side.PS:
            # upper side study
            X = thickness * normal[:,0] * B  
            Y = thickness * (normal[:,1] * B + (1 - B))  
        elif self.position == Side.SS:
            # lower side study
            X = - thickness * normal[:,0] * B  
            Y = - thickness * (normal[:,1] * B + (1 - B))  
        else: 
            if not isinstance(self.position, Side):
                raise TypeError('The profile line descriptor is wrong. The profile line position is based on a Enum object.\nThis object has to be imported inside the code using:\n>>> from kulfanLIB.profileline import Side\n>>> pLine = ProfileLine(..., Side.PS, ...)')

        return X, Y 

    def __TEdistribution(self, camberline: Camberline, x: list | np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Trailing egde gap distribution along the profile using a linear distribution function.
        
        Parameters
        ----------
        `camberline`: kulfanLIB.camberline.Camberline
            camberline object.
        `x`: float 
            study position, in chord percentage.

        Results
        -------
        `X`: float 
            horizontal displacement with respect to the `x` camberline position.
        `Y`: float 
            vertical displacement with respect to the `y` camberline position. The `y` camberline position is relative to `x`. 
        '''

        # allocating data
        if not isinstance(x, (list, np.ndarray)):
            x      = camberline.x
            normal = camberline.normal
        elif any(x < 0) or any(x > 1):
            raise ValueError('::: x must be a list or a np.ndarray, bounded between [0, 1].')
        else:
            normal = camberline.normalVector(x=x)

        if self.position == Side.PS:
            X = x * self.TEradius * normal[:,0]
            Y = x * self.TEradius * normal[:,1]
        elif self.position == Side.SS:
            X = - x * self.TEradius * normal[:,0]
            Y = - x * self.TEradius * normal[:,1] 
        else: 
            if not isinstance(self.position, Side):
                raise TypeError('The profile line descriptor is wrong. The profile line position is based on a Enum object.\nThis object has to be imported inside the code using:\n>>> from kulfanLIB.profileline import Side\n>>> pLine = ProfileLine(..., Side.PS, ...)')

        return X, Y 

    def __compute(self, camberline: Camberline, x: list | np.ndarray = None) -> None:
        '''
        This function computes the profile (suction/pressure side) of the airfoil following :cite:p:`clark2019step` fomulation.
        
        Parameters
        ----------
        `camberline`: camberline.Camberline
            camberline object.
        `x`: float 
            study position, in chord percentage.

        Results
        -------
        `x`: list of np.ndarray 
            normalized x coordinates of the profile line
        `y`: list of np.ndarray 
            normalized y coordinates of the profile line
        `X`: list of np.ndarray 
            real x coordinates of the profile line
        `Y`: list of np.ndarray 
            real y coordinates of the profile line
        '''

        if not isinstance(camberline, Camberline):
            raise TypeError('::: It is required a camberline.Camberline object for the computation of the profile line coordinates.')
    
        if not isinstance(x, (list, np.ndarray)):
            # data allocation
            x = camberline.x
            y = camberline.y
            # computing thickness
            X,   Y   = self.__thicknessDistribution(camberline=camberline)
            XTE, YTE = self.__TEdistribution(camberline=camberline)
        elif any(x < 0) or any(x > 1):
            raise ValueError('::: x must be a list or a np.ndarray, bounded between [0, 1].')
        else:
            # data allocation
            _, y, _, _ = camberline.coordinatePoint(x=x, scale=False, printout=False)
            # computing thickness
            X,   Y   = self.__thicknessDistribution(camberline=camberline, x=x)
            XTE, YTE = self.__TEdistribution(camberline=camberline, x=x)

        # allocating data 
        # normalized coordintates
        x = x + X + XTE
        y = y + Y + YTE
        # coordinates
        X = x * camberline.chord
        Y = y * camberline.chord

        return x, y, X, Y
    
    def compute(self, camberline: Camberline) -> None:
        '''
        This function computes the profile (suction/pressure side) of the airfoil following :cite:p:`clark2019step` fomulation.
        
        Parameters
        ----------
        `camberline`: camberline.Camberline
            camberline object.
        '''

        # coordinates computation
        x, y, X, Y = self.__compute(camberline=camberline)

        # coordinates allocation
        self.x = x
        self.y = y
        self.X = X 
        self.Y = Y

    def computeCoords(self, x: list | np.ndarray, camberline: Camberline) -> None:
        '''
        This function computes the profile (suction/pressure side) of the airfoil following :cite:p:`clark2019step` fomulation.
        
        Parameters
        ----------
        `x`: list | np.ndarray
            array which contains the study points for the blade representation in axial coordinates.
        `camberline`: camberline.Camberline
            camberline object.

        Returns
        -------
        `x`: np.ndarray
            array which stores the normalized x coordinates of the profile line
        `y`: np.ndarray
            array which stores the normalized y coordinates of the profile line 
        `X`: np.ndarray
            array which stores the scaled x coordinates of the profile line 
        `Y`: np.ndarray 
            array which stores the scaled y coordinates of the profile line 
        '''

        if not isinstance(x, (list, np.ndarray)):
            raise TypeError('::: It is required a `x` variable which is a list or a np.ndarray.')
        elif any(x > 1) or any(x < 0):
            raise ValueError('::: It is required that any value of `x` is in the interval [0, 1].')

        if not isinstance(camberline, Camberline):
            raise TypeError('::: It is required a camberline.Camberline object for the computation of the profile line coordinates.')
        
        xCamberline, yCamberline, _, _ = camberline.coordinatePoint(x=x, scale=False)

        # computing thickness
        # allocating data
        normal    = camberline.normalVector(x)
        B         = self.__bernstein(x, 0, 2)
        thickness = self.__thickness(x)  

        if self.position == Side.PS:
            XTH = thickness * normal[:,0] * B  
            YTH = thickness * (normal[:,1] * B + (1 - B))  
            XTE = x * self.TEradius * normal[:,0]
            YTE = x * self.TEradius * normal[:,1]
        elif self.position == Side.SS:
            XTH = - thickness * normal[:,0] * B  
            YTH = - thickness * (normal[:,1] * B + (1 - B)) 
            XTE = - x * self.TEradius * normal[:,0]
            YTE = - x * self.TEradius * normal[:,1] 
        
        # allocating data 
        # normalized coordintates
        x = xCamberline + XTH + XTE
        y = yCamberline + YTH + YTE
        # coordinates 
        X = x * camberline.chord 
        Y = y * camberline.chord

        return x, y, X, Y, XTE, YTE, XTH, YTH 

    def scale(
            self,
            N:          int, 
            camberline: Camberline,
            overwrite:  bool = False
        ) -> np.ndarray:
        r'''
        This function converts the suction side or pressure side parametrization, using the Kulfan parametrization model. 
        It solves a linear system of equations based the previous curve shape (which remains constant) relative the new `N` parameters parametrization.

        Parameters
        ----------
        `N`: int 
            this variable sets the number of parameters used to sketch the `profile` keeping the same curve shape.
        `camberline`: camberline.camberline 
            profile camberline. It does not change during the fuction. It allows computing the profile coordinates.
        `overwrite`: bool
            boolean value wich updates the current object with the new parametrization and recomputes the blade coordinates.
           
        Returns
        -------
        `A`: np.array
            array that stores the new parametrization of dimension `N + 1`.
        '''

        if not isinstance(N, int):
            raise TypeError('::: N must be an integer greater than 2.')
        elif N < 2:
            raise ValueError('::: N must be an integer greater than 2.')
        
        if not isinstance(camberline, Camberline):
            raise TypeError('::: camberline must be a camberline.Camberline object.')

        if not isinstance(overwrite, bool):
            raise TypeError('::: overwrite parameter must be a boolean variable.')

        # profile line study
        # evaluation of the initial profile line at N + 1 chord points
        x = np.linspace(0, 1, N + 1)

        # computing and collecting normalized profile line points 
        _, y, _, _ = self.__compute(camberline=camberline, x=x)
        _, yTEthickness = self.__TEdistribution(camberline=camberline, x=x)

        # removing camberline from y
        _, yCamber, _, _ = camberline.coordinatePoint(x=x, scale=False)
        y = y - yCamber 

        # removing TE linear thickness distribution from y
        y = y - yTEthickness

        # solving a linear system of equations at each control point (x, y)
        # matrix generation
        matrix = np.zeros((N + 1, N + 1))

        # matrix assembly
        for ii in range(N + 1):
            # removing A0 thickness distribution from y
            bernsteinValA0 = self.__bernstein(x[ii], 0, N)
            thicknessValA0 = self.A[0] * bernsteinValA0 * self.__C(x[ii])

            # removing A wedge angle thickness distribution from y
            bernsteinValWedge = self.__bernstein(x[ii], N, N)
            thicknessValWedge = self.A[-1] * bernsteinValWedge * self.__C(x[ii])
            
            # computing normal and Bernstein value that weights the A0 and wedge angle influence
            normal = camberline.normalVector(x[ii])
            B      = self.__bernstein(x[ii], 0, 2)

            if self.position == Side.PS:
                thickness    = thicknessValA0 + thicknessValWedge
                thicknessVal = thickness * (normal[1] * B + (1 - B))
            elif self.position == Side.SS:
                thickness    = thicknessValA0 + thicknessValWedge
                thicknessVal = - thickness * (normal[1] * B + (1 - B))
            else:
                raise TypeError('Error on the profile line position. It has to be a kulfanLIB.profileline.Side object.')

            # removing A0 and wedge angle influence from y
            y[ii] = y[ii] - thicknessVal 

            # matrix element allocation
            for jj in range(N + 1):
                # computing Bernstein value for the thickness distribution
                bernsteinVal = self.__bernstein(x[ii], jj, N)
                # computing thickness with unit value
                thicknessVal = bernsteinVal * self.__C(x[ii])
                # computing normal 
                normal = camberline.normalVector(x[ii])
                # computing Bernstein distribution for normal fraction distribution
                B = self.__bernstein(x[ii], 0, 2)

                # setting up thickness contribute with respect to profile position (SS or PS) 
                if self.position == Side.PS:
                    matrix[ii, jj] = thicknessVal * (normal[1] * B + (1 - B))
                elif self.position == Side.SS:
                    matrix[ii, jj] = - thicknessVal * (normal[1] * B + (1 - B))
                else: 
                    raise TypeError('Error on the profile line position. It has to be a kulfanLIB.profileline.Side object.')

        # reshaping matrix and known vector
        matrix = matrix[1:-1, :]
        matrix = matrix[:, 1:-1]
        y      = y[1:-1]

        # solving linear system 
        A = np.linalg.solve(matrix, y)
        
        # inserting know data 
        # leading edge properties 
        A0 = self.A[0]
        A  = np.insert(A, 0, [A0])

        # trailing edge properties 
        Awedge = self.A[-1] 
        A      = np.concatenate((A, [Awedge]))

        if overwrite: 
            # overwriting data and recomputing blade coordinates
            self.A = A 
            self.N = len(self.A) - 1
            self.compute(camberline=camberline)

        return A

    def __computeLength(
            self, 
            x0:         float, 
            x1:         float, 
            camberline: Camberline, 
            Npoints:    int, 
            printout:   bool            = False, 
            ax:         plt.Axes | None = None
        ) -> float:
        '''
        This function computes the line length between 2 points (`x0` and `x1`) along the profile line. 

        Parameters
        ----------
        `x0`: float 
            study coordinate in chord percentage. 
        `x1`: float 
            study coordinate in chord percentage. 
        `camberline`: kulfanLIB.camberline.Camberline 
            camberline object. 
        `Npoints`: int 
            number of points for the line discretization. If `Npoints = 3`: between `x0` and `x1` there is one additional discretization point that allow to compute better the line length.
        `printout`: bool
            value that allows to printout data.
        `ax`: plt.Axes
            axis where data should be plotted.

        Returns
        -------
        `linelength`: float 
            line length following the profile line curve. 
        '''
        # vector generation 
        xVec       = np.linspace(x0, x1, Npoints)
        lineLength = 0

        for ii in range(Npoints - 1): 
            # thickness distribution
            X0, Y0 = self.thicknessDistribution(xVec[ii], camberline)
            X1, Y1 = self.thicknessDistribution(xVec[ii+1], camberline)

            # thickness distribution due to trailing edge 
            XTE0, YTE0 = self.TEdistribution(xVec[ii], camberline)
            XTE1, YTE1 = self.TEdistribution(xVec[ii+1], camberline)

            # point coorindates assignment
            y0 = camberline.yCamberPoint(xVec[ii])   + Y0 + YTE0 - camberline.yCamberPoint(0.0)
            y1 = camberline.yCamberPoint(xVec[ii+1]) + Y1 + YTE1 - camberline.yCamberPoint(0.0)
            x0 = xVec[ii]   + X0 + XTE0
            x1 = xVec[ii+1] + X1 + XTE1

            if ax is not None:
                plt.scatter([x0, x1], [y0, y1], color='k')

            if printout:
                print('X0 = ', X0)
                print('Y0 = ', Y0)
                print('XTE0 = ', XTE0)
                print('YTE0 = ', YTE0)
                print('x0 = ', x0)
                print('y0 = ', y0)

            # computing length
            lineLength = lineLength + np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        return lineLength

    def __curveRefinement(
            self, 
            camberline:   Camberline, 
            pointsInUnit: int  = 150, 
            Npoints:      int  = 3, 
            printout:     bool = False,
            timePrintout: bool = True
        ) -> float:
        r'''
        This function automatically refines the camberline. It adds points in order to keep the distance between each consecutive points lower than `1/pointsInUnit`. 

        Parameters
        ----------
        `camberline`: kulfanLIB.camberline.Camberline
            camberline object.
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
        
        # getting time 
        T = [time.time()]
        
        # storing vectors
        xVec   = [0.0]
        lenVec = [0.0]

        # compute total length 
        totalLength = self.computeProfileLength(camberline=camberline, x=[0, 1])
        
        # updating time 
        T.append(time.time())

        # each segment length
        length = totalLength / (pointsInUnit - 1)

        prevX = 0.0
        # looping over all the profile line
        for _ in range(pointsInUnit - 2):
            # gussing new X
            X = prevX + 1 / (pointsInUnit - 1) 
            
            # updating starting point in the length computing function
            lengthFunc = lambda x: self.computeLength(prevX, x, camberline=camberline, Npoints=Npoints) - length

            # computing delta length 
            deltaLength = self.computeLength(prevX, X, camberline=camberline, Npoints=Npoints) - length

            # setting b point in bisection function
            if deltaLength > 0:
                res = bisect(f=lengthFunc, a=prevX, b=X)
            else: 
                res = bisect(f=lengthFunc, a=prevX, b=1.0)

            # storing lenght into vector
            segmentLen = self.computeLength(prevX, X, camberline=camberline, Npoints=5) 
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
        segmentLen = self.computeLength(prevX, 1.0, camberline=camberline, Npoints=5) 
        lenVec.append(segmentLen)

        # allocating points distribution 
        self.x = np.array(xVec) 

        # allocating length vector 
        self.length = np.array(lenVec)

        # allocating total profile length 
        self.totalLength = sum(lenVec) 
        print('>>> SURFACE TOTAL LENGTH = {0:.3E}'.format(self.totalLength))

        # computing lines points 
        self.computeSurface(self.x, camberline)

        if timePrintout:
            # updating time 
            T.append(time.time())
            print('>>> ELAPSED TIME FOR THE SURFACE DISCRETIZATION = {0:.2E} s'.format(T[2] - T[0]))
            print('    TOTAL TIME FOR LENGTH COMPUTATION = {0:.2E} s'.format(T[1] - T[0]))
            print('    TOTAL TIME FOR DISCRETIZATION     = {0:.2E} s'.format(T[2] - T[1]))

        return len(xVec)

    def __computeProfileLength(
            self, 
            camberline: Camberline, 
            x:          np.ndarray | list, 
            Npoints:    int        = 10, 
            Nsegments:  int        = 180, 
            printout:   bool       = False
        ) -> float:       
        '''
        Computing the total length of the surface line. 

        Parameters
        ----------
        `camberline`: kulfanLIB.camberline.Camberline
            camberline object. 
        `x`: np.array 
            start point and end point for the length computation, in chord percentage. 
        `Npoints`: int 
            number of points that will be used to discretizate the line between 2 consecutive x.
        `Nsegments`: int 
            number of segments. 

        '''

        # setting up study vector 
        xVec = np.linspace(x[0], x[1], Nsegments)
        
        # initializing storing value 
        totalLength = 0.0

        for ii in range(Nsegments-1):
            totalLength = totalLength + self.computeLength(xVec[ii], xVec[ii+1], camberline, Npoints)
            if printout: 
                print(xVec[ii])
                print(xVec[ii+1])

        return totalLength 
    
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

    def save(self, fileName: str, fileNameComponent: str, saveInZero: bool = True, nPoints: int = 200) -> None:
        '''
        This function saves the camberline data inside a text file. 
        The saved data are:
            * `x`
            * `y`

        Parameters
        ----------
        `fileName`: str
            name of the file where data will be stored.
        `fileNameComponent`: str
            name of the file where thickness data will be stored.
        `saveInZero`: bool 
            enables saving the blade profile line with the leading edge at (0, 0)

        '''

        # header generation
        header = '{0:>5}{1:>14}'.format('x', 'y')

        # saving profile data
        with open(file=fileName, mode='w') as f:
            if saveInZero:
                np.savetxt(f, np.column_stack([self.X, self.Y - self.Y[0]]), fmt='%+.5E', delimiter="  ", header=header)
            else:
                np.savetxt(f, np.column_stack([self.X, self.Y]), fmt='%+.5E', delimiter="  ", header=header)

        # saving bernstein data points
        x = np.linspace(0, 1, nPoints)

        # header generation
        header = '{0:>5}{1:>18}'.format('x', 'bernstein00')
        for ii in range(1, self.N + 1):
            header = header + '{0:>12}{1:02d}'.format('bernstein', ii)

        # saving file 
        with open(file=fileNameComponent, mode='w') as f:

            bernsteinVec = [x] 

            for ii in range(self.N + 1):
                bernsteinVec.append(self.A[ii] * self.bernstein(x, ii, self.N)) 
            
            bernsteinVec = np.array(bernsteinVec)

            np.savetxt(f, bernsteinVec.T, fmt='%+.5E', delimiter="  ", header=header)

    def plot(
            self, 
            ax:         plt.Axes | None = None, 
            color:      str      | None = None, 
            linestyle:  str      = '-',
            plotInZero: bool     = False, 
            normalized: bool     = False,
            pitch:      float    = 0,
            linewidth:  float    = 2,
            number:     int      = 1
        ) -> None:
        '''
        This function plots the profile line.

        Parameters
        ----------
        `ax`: matplotlib.pyplot.axes
            plt.axes object. 
        `color`: str
            line color.
        `linestyle`: str
            line linestyle.
        `normalized`: bool 
            boolean value for the normalization of the profile line.
        `plotInZero`: bool
            if `True` moves the blade in the chart such that the leading edge is at `(0,0)`. If `False` keeps the blade free with respect to the outputs of the Kulfan parametrization.
        `pitch`: bool 
            it translates the profile from `0` to the pitch value. It works only if `plotInZero` is `True`.
        `linewidth`: float 
            profile line linewidth in plot
        `number`: int
            numeber of profile lines to plot if pitch > 0
        '''
        
        if self.position == Side.SS:
            label = 'SS'
        elif self.position == Side.PS:
            label = 'PS'
        else:  
            if not isinstance(self.position, Side):
                raise TypeError('The profile line descriptor is wrong. The profile line position is based on a Enum object.\nThis object has to be imported inside the code using:\n>>> from kulfanLIB.profileline import Side\n>>> pLine = ProfileLine(..., Side.PS, ...)')
        
        if plotInZero:
            if normalized:
                pitch = pitch * self.x[-1] / self.X[-1] 
                for ii in range(number):
                    ax.plot(self.x, self.y - self.y[0] + ii*pitch, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            else:
                for ii in range(number):
                    ax.plot(self.X, self.Y - self.Y[0] + ii*pitch, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        else:
            if normalized:
                pitch = pitch * self.x[-1] / self.X[-1]
                for ii in range(number):
                    ax.plot(self.x, self.y + ii*pitch, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            else:
                for ii in range(number):
                    ax.plot(self.X, self.Y + ii*pitch, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    