import os
import numpy as np
from   geometryLIB import optimizer

def main():
    # file name allocation
    fileName = 'airfoil.dat'
    fileName = 'coords.dat'
    
    with open(file=os.path.join(['../containter/', fileName]), mode='r') as f:
        data = np.loadtxt(f)
        data = np.array(data)

    angle   = 0
    Nsuct   = 4
    Npress  = 4
    nMax    = 4
    nPoints = 120

    _, _, _ = optimizer.optimizeBlade(data, Nsuct, Npress, angle=angle, LEradius=2.5E-2, method='Nelder-Mead', nMax=nMax, plot=True, nPoints=nPoints)

    # blade.save('blade1.txt')

if __name__ == '__main__':
    main()