#!/usr/bin/ python3
import os 
import sys
import unittest
import json 
import numpy             as np 
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf
from   scipy import interpolate 

# importing folder 
bladerDir = os.path.dirname(os.path.dirname(__file__))
srcDir    = os.path.join(bladerDir, 'src/')
testDir   = os.path.join(bladerDir, 'test/')
utilDir   = os.path.join(bladerDir, 'container/')

print(50*'-' + ' BLADER ' + 50*'-')
print('>>> CURRENT WORKING     DIRECTORY: {0:s}'.format(os.getcwd()))
print('>>> BLADER  WORKING FILE POSITION: {0:s}'.format(__file__))
print('>>> BLADER  WORKING     DIRECTORY: {0:s}'.format(bladerDir))
print('>>> BLADER  SOURCE      DIRECTORY: {0:s}'.format(srcDir))
print('>>> BLADER  TEST        DIRECTORY: {0:s}'.format(testDir))
print('>>> BLADER  UTILITIES   DIRECTORY: {0:s}'.format(utilDir))
print((100 + len(' BLADER '))*'-')

# setting up src directory
if os.path.exists(bladerDir):
    sys.path.insert(0, srcDir)
else:
    raise IsADirectoryError('INCORRECT PATH')

from geometryLIB import optimizer

class TestOptimizer(unittest.TestCase):
    def test_optimizer(self):
        # reading data from json file 
        try:
            file = './container/test_optimizer.json'
            with open(file) as f:
                data = json.load(f)
        except:
            file = 'test/container/test_optimzer.json'
            with open(file) as f:
                data = json.load(f)

        # pdf generation 
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(testDir, 'pdf', "testModules.pdf"))
    
        # blade optimization
        for fileName in data["fileName"]:
            # reading data from json file 
            fileName = os.path.join('./data/', fileName)
            print('>>> OPTIMIZING FILE: {0}'.format(fileName))

            with open(fileName) as f:
                bladeCoords = np.loadtxt(f)
                # plotting blade 
                bladeCoords = np.array(bladeCoords)
            
            # checking coordinates 
            _, bladeCoords = optimizer.camberlineAnalysis(data=bladeCoords, plot=False)

            # move coordinates into origin 
            bladeCoordsInOrigin, _, _, _ = optimizer.bladeInOrigin(data=bladeCoords, scale=True)

            # rotating blade 
            bladeCoordsRotated = optimizer.rotate(data=bladeCoordsInOrigin, theta=10)
            
            # moving blade into origin 
            bladeCoordsRotatedInOrigin, minPos, _, _ = optimizer.bladeInOrigin(data=bladeCoordsRotated, scale=True)
            
            # interpolating blade coordinates with a spline
            upperLine, lowerLine, upperLineReal, lowerLineReal, _, _, _, _ = optimizer.interpolateData(bladeCoordsRotatedInOrigin, kind='cubic', plot=False)

            # leading edge position
            xLE, yLE, LEradius, axialChord = optimizer.bladeLEpos(upperLine=upperLineReal, lowerLine=lowerLineReal, bothSide=False, plot=False)
            
            if True:
                # trailing edge angle computation
                xTE = (bladeCoordsRotatedInOrigin[0, 0] + bladeCoordsRotatedInOrigin[-1, 0]) / 2
                yTE = (bladeCoordsRotatedInOrigin[0, 1] + bladeCoordsRotatedInOrigin[-1, 1]) / 2

                # rotate with respect to trailing edge 
                rotCoords = bladeCoordsRotatedInOrigin

                # setting up data
                upperPart = np.array(rotCoords[minPos::, :])
                lowerPart = np.array(rotCoords[0:minPos+1, :])

                # getting blade chord
                upperChord = np.max(upperPart[:, 0]) - np.min(upperPart[:, 0])
                lowerChord = np.max(lowerPart[:, 0]) - np.min(lowerPart[:, 0])
                
                # normalize data 
                upperPart = upperPart / upperChord 
                lowerPart = lowerPart / lowerChord 

                # interpolating data 
                upperLineFunc = interpolate.interp1d(upperPart[:, 0], upperPart[:, 1])
                lowerLineFunc = interpolate.interp1d(lowerPart[:, 0], lowerPart[:, 1])

                # evaluating data
                x = optimizer.chebyschev(0, 1, 300)
                yUpper = upperLineFunc(x)
                yLower = lowerLineFunc(x)

                # computing camberline 
                yCamberline = (yUpper + yLower) / 2
            
                # rotating data 
                camberlinePoints = np.stack([x, yCamberline], axis=1)

            # optimizing camberline
            stagger, metalInlet, metalOutlet, _, camberlineCost = optimizer.optimizeCamberline(upperLine=upperLine, lowerLine=lowerLine, LEpos=[xLE, yLE], upperChord=upperChord, lowerChord=lowerChord, plot=True)
    

            # plotting function properties  
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)
            
            ax.plot(bladeCoords[:,0],                bladeCoords[:,1],                'k',       linewidth=3, label='INITIAL COORDINATES')
            ax.plot(bladeCoordsInOrigin[:,0],        bladeCoordsInOrigin[:,1],        'r--',     linewidth=3, label='IN ORIGIN COORDINATES')
            ax.plot(bladeCoordsRotated[:,0],         bladeCoordsRotated[:,1],         'skyblue', linewidth=3, label='ROTATED COORDINATES')
            ax.plot(bladeCoordsRotatedInOrigin[:,0], bladeCoordsRotatedInOrigin[:,1], 'g',       linewidth=3, label='ROTATED COORDINATES IN ORIGIN')
            
            ax.plot(x + 0, yCamberline + 0, 'k', label='CAMBERLINE')
            ax.plot(xLE, yLE, 'ko', label='LEADING EDGE POSITION')
            ax.plot(xTE, yTE, 'ko', label='TRAILING EDGE POSITION')
            ax.plot(x, yUpper, 'm--')
            ax.plot(x, yLower, 'm--')

            x = optimizer.chebyschev(0, 0.97, 300)
            ax.plot(x, upperLineReal(x), 'b--', linewidth=3, label='UPPER LINE')
            ax.plot(x, lowerLineReal(x), 'c--', linewidth=3, label='LOWER LINE')

            # setting up axes 
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(visible=True, linestyle='dotted')
            ax.legend(bbox_to_anchor=(1,1), loc='upper left')
            plt.tight_layout()

            plt.show()

            pdf.savefig(figure=fig)
                
        pdf.close()
