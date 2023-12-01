#!/usr/bin/ python3
import os 
import sys
import unittest
import json 
import numpy as np 
from   matplotlib import pyplot as plt 
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

from geometryLIB import camberline

class TestCamberline(unittest.TestCase):
    def test_camberline(self):
        # reading data from json file 
        try:
            file = './container/test_camberline.json'
            with open(file) as f:
                data = json.load(f)
        except:
            file = 'test/container/test_camberline.json'
            with open(file) as f:
                data = json.load(f)

        # data allocation 
        staggerVec  = np.linspace(data["staggerAngle"]["min"], data["staggerAngle"]["max"], data["staggerAngle"]["n"])
        metalInVec  = np.linspace(data["metalInAngle"]["min"], data["metalInAngle"]["max"], data["metalInAngle"]["n"])
        metalOutVec = np.linspace(data["metalOutAngle"]["min"], data["metalOutAngle"]["max"], data["metalOutAngle"]["n"])
        chordVec    = np.linspace(data["chord"]["min"], data["chord"]["max"], data["chord"]["n"])
        pitchVec    = np.linspace(data["pitch"]["min"], data["pitch"]["max"], data["pitch"]["n"])
        chebyschev  = data["chebyschev"]
        origin      = data["origin"]
        number      = data["number"]
        vector      = data["vector"]
        normalized  = data["normalized"]

        # pdf generation 
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(testDir, 'pdf', "testCamberline.pdf"))

        # camberline analysis
        cLineCounter = 0
        for metalInAngle in metalInVec:
            for metalOutAngle in metalOutVec:
                for staggerAngle in staggerVec:
                    for pitch in pitchVec:
                        for chord in chordVec:
                            # updating camberline counter 
                            cLineCounter = cLineCounter + 1

                            print('-'*50)
                            print('>>> CAMBERLINE STUDY')
                            print('>>>     METAL INLET  ANGLE = {0:.2f}'.format(metalInAngle))
                            print('>>>     METAL OUTLET ANGLE = {0:.2f}'.format(metalOutAngle))
                            print('>>>     STAGGER      ANGLE = {0:.2f}'.format(staggerAngle))
                            print('>>>     CHORD              = {0:.2f}'.format(chord))
                            print('>>>     PITCH              = {0:.2f}'.format(pitch))

                            # object generation
                            cLine = camberline.Camberline(stagger=staggerAngle, metalIn=metalInAngle, metalOut=metalOutAngle, chord=chord, chebyschev=chebyschev, origin=origin)

                            # axes generation
                            fig = plt.figure()
                            ax = fig.add_subplot(1,1,1)

                            # plotting data
                            cLine.plot(ax, normalized=normalized, number=number, pitch=pitch, vector=vector)

                            # ax properties
                            ax.set_aspect('equal')
                            ax.grid(visible=True, linestyle='dotted')
                            rowTab = [r'$\chi_1$', r'$\chi_2$', r'$\gamma$', r'$pitch$', r'$chord$']
                            colTab = np.round([metalInAngle, metalOutAngle, staggerAngle, pitch, chord], 3)
                            colTab = colTab.tolist()
                            for ii in range(3):
                                colTab[ii] = '{0:+.2f}'.format(colTab[ii]) + r'$^{\circ}$'
                            table = ax.table(cellText=[colTab], colLabels=rowTab, colWidths=[0.5/5, 0.5/5, 0.5/5, 0.5/5, 0.5/5], bbox=[0., 1.03, 1.0, 0.15], cellLoc='center')
                            table.auto_set_font_size(False)
                            table.set_fontsize(8)
                            fig.tight_layout()
                            pdf.savefig(figure=fig)
                            plt.close()

        pdf.close()