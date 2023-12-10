#!/usr/bin/ python3
import os 
import sys
import unittest
import json 
import numpy as np 
import matplotlib.backends.backend_pdf

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

        # data allocation 
        nMax    = data["nMax"]
        nPoints = data["nPoints"]

        # pdf generation 
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(testDir, 'pdf', "testBlade.pdf"))
    
        # blade optimization
        for fileName in data["fileName"]:
            # reading data from json file 
            fileName = os.path.join('./data/', fileName)
            print('>>> OPTIMIZING FILE: {0}'.format(fileName))
            with open(fileName) as f:
                bladeCoords = np.loadtxt(f)
                bladeCoords = np.array(bladeCoords)
   
            for Ncounter in range(min(len(data["N"]["SS"]), len(data["N"]["PS"]))):
                # DOF
                Nsuct  = data["N"]["SS"][Ncounter]
                Npress = data["N"]["PS"][Ncounter]

                print('>>> N SUCTION  SIDE = {0:d}'.format(Nsuct))
                print('>>> N PRESSURE SIDE = {0:d}'.format(Npress))
                
                # optimizing blade with respect to different DOF
                _, _, _, _, fig, _ = optimizer.optimizeBlade(bladeCoords, Nsuct, Npress, angle=0, nMax=nMax, nPoints=nPoints, plot=True, save=True)
                pdf.savefig(figure=fig)
                
        pdf.close()
