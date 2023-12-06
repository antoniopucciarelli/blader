import numpy as np 
import matplotlib.pyplot as plt 
from   geometryLIB import camberline, profileLine

class Blade():
    '''
    Blade class. This class is built using the **Kulfan formulation** :cite:p:`kulfan2008universal`.
    '''

    def __init__(
            self, 
            stagger:    int, 
            metalIn:    int, 
            metalOut:   int, 
            chord:      float, 
            pitch:      float,
            Asuct:      list | np.ndarray, 
            Apress:     list | np.ndarray, 
            TEradius:   float = 0.0, 
            LEradius:   float = None, 
            wedgeAngle: float = None,
            nPoints:    int   = 100,
            chebyschev: bool  = True, 
            origin:     bool  = True,
        ):

        # checking inputs
        self.__checkInput(Asuct=Asuct, Apress=Apress, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, pitch=pitch)

        # allocating global properties
        self.__stagger    = stagger 
        self.__metalIn    = metalIn
        self.__metalOut   = metalOut
        self.__Asuct      = Asuct 
        self.__Apress     = Apress
        self.__LEradius   = LEradius
        self.__TEradius   = TEradius
        self.__wedgeAngle = wedgeAngle
        self.__chord      = chord
        self.__pitch      = pitch
        self.__origin     = origin

        # camberline generation
        self.__camberline = camberline.Camberline(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=chord, nPoints=nPoints, chebyschev=chebyschev, origin=origin)
        
        # suction side generation
        self.__sLine = profileLine.ProfileLine(A=Asuct, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=profileLine.Side.SS)
        
        # pressure side generation
        self.__pLine = profileLine.ProfileLine(A=Apress, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=profileLine.Side.PS)

        # computing blade coordinates
        self.__sLine.compute(self.__camberline)
        self.__pLine.compute(self.__camberline)

    def __checkInput(self, Asuct: list | np.ndarray, Apress: list | np.ndarray, LEradius: float, wedgeAngle: float, TEradius: float, pitch: float) -> None:
        '''
        This function checks the blade input values. 
        '''

        if not isinstance(Asuct, (list, np.ndarray)):
            raise TypeError('::: The geometry array for the suction side must be a list or a np.ndarray.')

        if not isinstance(Apress, (list, np.ndarray)):
            raise TypeError('::: The geometry array for the pressure side must be a list or a np.ndarray.')

        if not isinstance(LEradius, (float, int)):
            if LEradius == None:
                if Asuct[0] != Apress[0]:
                    raise ValueError('::: Since the leading edge radius is set to `None`: the first geometry parameter in the suction and pressure side arrays of the blade must share the same value.')
            else:
                raise TypeError('::: The leading edge radius must be a float greater than 0.')
        elif LEradius < 0:
            raise ValueError('::: The leading edge radius must be a float greater than 0.')
        
        if not isinstance(wedgeAngle, (float, int)):
            if wedgeAngle == None:
                if Asuct[-1] != Apress[-1]:
                    raise ValueError('::: Since the wedge angle is set to `None`: the last geometry parameter in the suction and pressure side arrays of the blade must share the same value.')
            else:
                raise TypeError('::: The wedge angle must be a float greater than 0.')
        elif wedgeAngle < 0:
            raise ValueError('::: The wedge angle must be a float greater than 0.')

        if not isinstance(TEradius, (float, int)):
            raise TypeError('::: The trailing edge radius must be a flot greater than 0.')
        elif TEradius < 0:
            raise ValueError('::: The trailing edge radius must be a flot greater than 0.')
        
        if not isinstance(pitch, (float, int)):
            raise TypeError('::: The pitch must be a flot greater than 0.')
        elif pitch < 0:
            raise ValueError('::: The pitch must be a flot greater than 0.')

    def update(
            self, 
            stagger:    int               = None, 
            metalIn:    int               = None, 
            metalOut:   int               = None, 
            chord:      float             = None, 
            pitch:      float             = None,
            Asuct:      list | np.ndarray = None, 
            Apress:     list | np.ndarray = None, 
            TEradius:   float             = 0.0, 
            LEradius:   float             = None, 
            wedgeAngle: float             = None,
            nPoints:    int               = 100,
            chebyschev: bool              = True, 
            origin:     bool              = True,
        ) -> None:
        '''
        Updating blade propreties. 
        '''

        # checking data 
        if stagger == None: 
            stagger = self.__stagger 
        
        if metalIn == None:
            metalIn = self.__metalIn 

        if metalOut == None:
            metalOut = self.__metalOut 
        
        if not isinstance(Asuct, (list, np.ndarray)):
            Asuct = self.__Asuct 

        if not isinstance(Apress, (list, np.ndarray)):
            Apress = self.__Apress 

        if LEradius == None: 
            LEradius = self.__LEradius 

        if TEradius == None: 
            TEradius = self.__TEradius 

        if wedgeAngle == None:
            wedgeAngle = self.__wedgeAngle 

        if chord == None:
            chord = self.__chord

        if pitch == None:
            pitch = self.__pitch 

        if origin == None:
            origin = self.__origin

        # checking inputs 
        self.__checkInput(Asuct=Asuct, Apress=Apress, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, pitch=pitch)

        # allocating global properties
        self.__stagger    = stagger 
        self.__metalIn    = metalIn
        self.__metalOut   = metalOut
        self.__Asuct      = Asuct 
        self.__Apress     = Apress
        self.__LEradius   = LEradius
        self.__TEradius   = TEradius
        self.__wedgeAngle = wedgeAngle
        self.__chord      = chord
        self.__pitch      = pitch
        self.__origin     = origin

        # camberline generation
        self.__camberline = camberline.Camberline(stagger=stagger, metalIn=metalIn, metalOut=metalOut, chord=chord, nPoints=nPoints, chebyschev=chebyschev, origin=origin)
        
        # suction side generation
        self.__sLine = profileLine.ProfileLine(A=Asuct, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=profileLine.Side.SS)
        
        # pressure side generation
        self.__pLine = profileLine.ProfileLine(A=Apress, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=profileLine.Side.PS)

        # computing blade coordinates
        self.__sLine.compute(self.__camberline)
        self.__pLine.compute(self.__camberline)

    def coordinate(self) -> tuple: 
        '''
        This function extracts the blade coordinates from the blade object into a vector style representation.
        '''

        XPS = self.__pLine.X 
        YPS = self.__pLine.Y 
        XSS = self.__sLine.X 
        YSS = self.__sLine.Y 
        XCL = self.__camberline.X
        YCL = self.__camberline.Y

        return XPS, YPS, XSS, YSS, XCL, YCL

    def rotate(
            self, 
            theta:      float, 
            ax:         plt.Axes = None, 
            normalized: bool     = True, 
            pitch:      float    = 0.0, 
            number:     int      = 1, 
            flip:       bool     = False,
            Ccolor:     str      = 'k', 
            Pcolor:     str      = 'r', 
            Scolor:     str      = 'b', 
            linewidth:  int      = 2
        ) -> tuple:
        '''
        This function returns the blade coordinates rotated by a theta angle (in degrees).
        '''

        # rotating blade coordinates
        x_camberline, y_camberline, X_camberline, Y_camberline = self.__camberline.rotate(theta=theta)
        x_pLine, y_pLine, X_pLine, Y_pLine = self.__pLine.rotate(theta=theta)
        x_sLine, y_sLine, X_sLine, Y_sLine = self.__sLine.rotate(theta=theta)

        if ax is not None:
            if normalized: 
                pitch = pitch / self.__chord
            
                for ii in range(number):
                    if flip:
                        ax.plot(x_camberline, - y_camberline + ii * pitch, color=Ccolor, linewidth=linewidth)
                        ax.plot(x_pLine, - y_pLine + ii * pitch, color=Pcolor, linewidth=linewidth)
                        ax.plot(x_sLine, - y_sLine + ii * pitch, color=Scolor, linewidth=linewidth)
                    else:
                        ax.plot(x_camberline, y_camberline + ii * pitch, color=Ccolor, linewidth=linewidth)
                        ax.plot(x_pLine, y_pLine + ii * pitch, color=Pcolor, linewidth=linewidth)
                        ax.plot(x_sLine, y_sLine + ii * pitch, color=Scolor, linewidth=linewidth)
            else:
                for ii in range(number):
                    if flip:
                        ax.plot(X_camberline, - Y_camberline + ii * pitch, color=Ccolor, linewidth=linewidth)
                        ax.plot(X_pLine, - Y_pLine + ii * pitch, color=Pcolor, linewidth=linewidth)
                        ax.plot(X_sLine, - Y_sLine + ii * pitch, color=Scolor, linewidth=linewidth)
                    else:
                        ax.plot(X_camberline, Y_camberline + ii * pitch, color=Ccolor, linewidth=linewidth)
                        ax.plot(X_pLine, Y_pLine + ii * pitch, color=Pcolor, linewidth=linewidth)
                        ax.plot(X_sLine, Y_sLine + ii * pitch, color=Scolor, linewidth=linewidth)
           

        return x_camberline, y_camberline, X_camberline, Y_camberline, x_pLine, y_pLine, X_pLine, Y_pLine, x_sLine, y_sLine, X_sLine, Y_sLine

    def scale(
            self, 
            Nsuct:  int, 
            Npress: int, 
            plot:   bool = False
        ) -> None:
        '''
        This function scales the blade geometry.

        Parameters
        ----------
        `Nsuct`: int 
            new blade DOF for the suction side of the blade 
        `Npress`: int 
            new blade DOF for the suction side of the blade
        `nPoints`: int 
            suction side and pressure side discretization points
        `wedgeAngle`: float 
            wedge angle of the blade 
        `ax`: plt.Axes 
            matplotlib axes object
        '''

        # scaling data 
        Asuct  = self.__sLine.scale(N=Nsuct,  camberline=self.__camberline, overwrite=False)
        Apress = self.__pLine.scale(N=Npress, camberline=self.__camberline, overwrite=False) 

        # computing leading edge radius 
        LEradius = Asuct[0]**2 / 2

        # data allocation
        self.update(Asuct=Asuct, Apress=Apress) 

        if plot:
            # printing out data 
            self.printout()

            # plotting data 
            self.plot()

        return Asuct, Apress, LEradius

    def save(self, fileName: str) -> None:
        '''
        This function saves the normalized blade properties inside a text file. 
        '''
        
        with open(file=fileName + '.dat', mode='w') as f:
            header = 'camberline.x, camberline.y, pressureLine.x, pressureLine.y, suctionLine.x, suctionLine.y'
            data = np.stack([self.__camberline.x, self.__camberline.y, self.__pLine.x, self.__pLine.y, self.__sLine.x, self.__sLine.y], axis=1)
            np.savetxt(fname=f, X=data, delimiter=' ', header=header)

        print('>>> BLADE COORDINATES SAVED INTO: {0:s}.dat'.format(fileName))

        with open(file=fileName + '_coords.dat', mode='w') as f:
            header = 'x, y'
            data = np.stack([np.concatenate([np.flip(self.__pLine.x[1::]), self.__sLine.x]), np.concatenate([np.flip(self.__pLine.y[1::]), self.__sLine.y])], axis=1)
            np.savetxt(fname=f, X=data, delimiter=' ', header=header)

        print('>>> BLADE COORDINATES SAVED INTO: {0:s}_coords.dat'.format(fileName))

        with open(file=fileName + '_params.dat', mode='w') as f:
            header = 'stagger, inlet metal angle, outlet metal angle, LEradius, TEradius, Asuct[1:{0:d}], Apress[1:{1:d}], pitch'.format(len(self.__Asuct) - 1, len(self.__Apress) - 1)
            data = np.concatenate([[self.__stagger, self.__metalIn, self.__metalOut, self.__LEradius, self.__TEradius], self.__Asuct[1::], self.__Apress[1::], [self.__pitch]]) 
            np.savetxt(fname=f, X=data, delimiter=' ', header=header)
        
        print('>>> BLADE COORDINATES SAVED INTO: {0:s}_params.dat'.format(fileName))

    def printout(self) -> None:
        '''
        This function prints out the blade properties.
        '''

        print('>>> CAMBERLINE')
        print('    >>> STAGGER      ANGLE = {0:+.2E} deg'.format(self.__camberline.stagger))
        print('    >>> METAL INLET  ANGLE = {0:+.2E} deg'.format(self.__camberline.metalIn))
        print('    >>> METAL OUTLET ANGLE = {0:+.2E} deg'.format(self.__camberline.metalOut))
        print('>>> UPPER SIDE')
        print('    >>> # KULFAN PARAMETERS      = {0:d}'.format(self.__pLine.N))
        print('    >>> LEADING EDGE RADIUS      = {0:.2E}'.format(self.__pLine.LEradius))
        print('    >>> KULFAN PARAMETERS (FULL) = {0}'.format(np.array2string(self.__pLine.A, precision=2)))
        print('    >>> WEDGE ANGLE              = {0:.2E}'.format(self.__pLine.wedgeAngle))
        print('    >>> TRAILING EDGE RADIUS     = {0:.2E}'.format(self.__pLine.TEradius))
        print('>>> LOWER SIDE')
        print('    >>> # KULFAN PARAMETERS      = {0:d}'.format(self.__sLine.N))
        print('    >>> LEADING EDGE RADIUS      = {0:.2E}'.format(self.__sLine.LEradius))
        print('    >>> KULFAN PARAMETERS (FULL) = {0}'.format(np.array2string(self.__sLine.A, precision=2)))
        print('    >>> WEDGE ANGLE              = {0:.2E}'.format(self.__sLine.wedgeAngle))
        print('    >>> TRAILING EDGE RADIUS     = {0:.2E}'.format(self.__sLine.TEradius))

    def plot(
            self, 
            ax:         plt.Axes = None, 
            Ccolor:     str      = 'k', 
            SScolor:    str      = 'r', 
            PScolor:    str      = 'b',
            normalized: bool     = True, 
            number:     int      = 2,
        ) -> None:
        '''
        This function plots the blade. 
        '''

        # checking axes
        if not isinstance(ax, plt.Axes):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            newAxes = True 
        else:
            newAxes = False

        # plotting data
        self.__camberline.plot(ax=ax, normalized=normalized, color=Ccolor, pitch=self.__pitch, number=number)
        self.__sLine.plot(ax=ax, color=SScolor, plotInZero=self.__origin, pitch=self.__pitch, number=number, normalized=normalized)
        self.__pLine.plot(ax=ax, color=PScolor, plotInZero=self.__origin, pitch=self.__pitch, number=number, normalized=normalized)

        # axes decoration
        if newAxes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Blade')
            ax.grid(visible=True, linestyle='dotted')
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.show()