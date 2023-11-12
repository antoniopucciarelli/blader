#!/usr/bin/env python3
import matplotlib.pyplot     as plt 
import numpy                 as np 
import geometryLIB.camberline  as camberline 
import geometryLIB.profileLine as profileLine 

class inverseKulfan:
    def __init__(
            self, 
            x:        np.array, 
            y:        np.array, 
            A:        np.array, 
            N:        int, 
            stagger:  float, 
            metalIn:  float, 
            metalOut: float, 
            position: bool
        ):
        self.x           = x 
        self.y           = y 
        self.A           = A
        self.N           = N 
        self.metalIn     = metalIn 
        self.metalOut    = metalOut
        self.stagger     = stagger
        self.camberline  = camberline(stagger, metalIn, metalOut)
        self.profileLine = profileLine(N, self.A, position)
    
    def updateA(self, A: np.array):
        self.A             = A 
        self.profileLine.A = A

    def bernstein(self, x, i, N):
        '''
        Bernstein polynomial
        '''
        K = np.math.factorial(N) / (np.math.factorial(i) * np.math.factorial(N - i))
        return K * x**i * (1 - x)**(N - i)

    def C(self, x: np.array) -> float:
        '''
        Closure function 
        '''
        return x**0.5 * (1 - x)

    def findX(
            self, 
            nPoints:     int      = 10000, 
            tol:         float    = 1e-4, 
            plot:        bool     = False, 
            ax:          plt.Axes = None, 
            ax1:         plt.Axes = None, 
            color:       str      = 'k', 
            markerColor: list     = ['r', 'b'], 
            markerSize:  list     = [10, 5], 
            label:       str      = 'max error plot'
        ) -> list:
        '''
        This function finds the camberline points that are mapped to the profile line keeping A vector fixed.
        '''

        # setting up profile line parametrization --> x is the parametrization coordinate
        #                                         --> X is the profile/mapped coordinate 
        self.profileLine.x = np.linspace(0.0, 1.0, nPoints)
        # computing X coordinates using defined camberline and fixed A vector at the x positions (previous code line)
        self.profileLine.computeSurface(self.profileLine.x, self.camberline)

        # setting up camberline points 
        xPosition = np.linspace(0.0, 1.0, nPoints)
        
        # allocating storing vector
        xCamber = []
        error   = []

        # since the fuction is not based on a Newton method (or something similar) 
        #   the model passes all the points on the surface and then compares it to the ones 
        #   got from a line transformation 
        counter = 0 

        # checking all the external coordinates (self.x)
        vectorPosition = []
        for ii, xVal in enumerate(self.x): 
            
            found = False # variable that allows to check if a real coordinate is mapped between 2 study points  
            
            while counter < nPoints - 1 and found == False:
                counter = counter + 1 
                if self.profileLine.X[counter - 1] <= xVal and self.profileLine.X[counter] >= xVal:
                    found = True 
                    tol1  = np.abs(xVal - self.profileLine.X[counter - 1])
                    tol2  = np.abs(xVal - self.profileLine.X[counter]) 
                    if tol1 < tol2:
                        xCamber.append(xPosition[counter-1])
                    else: 
                        xCamber.append(xPosition[counter])
                elif self.profileLine.X[counter - 1] >= xVal and self.profileLine.X[counter] <= xVal:
                    found = True 
                    tol1  = np.abs(xVal - self.profileLine.X[counter - 1])
                    tol2  = np.abs(xVal - self.profileLine.X[counter]) 
                    if tol1 < tol2:
                        xCamber.append(xPosition[counter-1])
                    else: 
                        xCamber.append(xPosition[counter])
                
            if found == True and tol1 > tol and tol2 > tol:
                print('-- Warning some x camberline points are compute with a tolerance above: {0:.2E}'.format(tol))
            
            if found == True:
                vectorPosition.append(ii)
                error.append(max([tol1, tol2]))

        self.camberline.x = np.array(xCamber) 

        if plot:
            if ax is not None:
                ax.semilogy(np.linspace(0, 1.0, len(error)), error, color=color, label=label)
            
            if ax1 is not None:
                self.profileLine.computeSurface(self.camberline.x, self.camberline)
                ax1.scatter(self.profileLine.X, self.profileLine.Y, c=markerColor[0], s=markerSize[0])
                ax1.scatter(self.x,             self.y,             c=markerColor[1], s=markerSize[1])

        return vectorPosition

    def findA(
            self, 
            Ainterval: list, 
            Astart:    list = [0], 
            Aend:      list = [0], 
            tol:       float = 1e-3, 
            plot:      bool = False, 
            ax:        plt.Axes = None
        ) -> list:
        import itertools

        Avec = np.zeros((self.N+1, Ainterval))

        for ii in range(self.N+1):
            try:
                Avec[ii, :] = np.linspace(Astart[ii], Aend[ii], Ainterval)
            except:
                Avec[ii, :] = np.linspace(Astart[0], Aend[0], Ainterval)

        Aguess   = np.zeros((self.N+1,))
        error    = np.ones((Ainterval**(self.N+1),))
        minAvec  = np.zeros((Ainterval**(self.N+1), self.N+1))
        counter  = 0
        minError = np.inf

        for combination in itertools.product(*Avec):
            Aguess[:] = combination[:]
        
            # update A value 
            self.updateA(Aguess)
            print('-' * 45)
            print('-- new study: \nA = [', *self.A, ']', sep='\n\t')

            # find x 
            vectorPosition = self.findX(nPoints=1000, tol=1)
            
            # computing profile properties (X, Y) using Aguess
            self.profileLine.computeSurface(self.camberline.x, self.camberline)

            # error computation 
            profileError = 0.0 
            for ii in range(len(vectorPosition)):
                profileError = profileError + (self.profileLine.Y[ii] - self.y[vectorPosition[ii]])**2

            profileError = np.sqrt(profileError/len(vectorPosition))


            error[counter]   = profileError
            minAvec[counter] = Aguess
            counter          = counter + 1
        
            if profileError < minError:
                print('profileError ({0:.2E}) < ({1:.2E}) minError'.format(profileError, minError))
                minError = profileError.copy()
                minA     = Aguess.copy()
                if plot == True:
                    ax.plot(self.profileLine.X, self.profileLine.Y)
 
            print('minError = {0:.2E}'.format(profileError))
            
            if profileError < tol: 
                print('Error lower than tolerance {0:.2E}'.format(tol))
                print('^' * 35)
                break

        if plot == True:
            ax.plot(self.x, self.y, '--k')
            pos = np.where(error == error.min())
            self.updateA(minAvec[pos][0,:])
            _ = self.findX(nPoints=1000, tol=1)
            self.profileLine.computeSurface(self.camberline.x, self.camberline)
            ax.plot(self.profileLine.X, self.profileLine.Y, '--r')

        return minA, minError, error