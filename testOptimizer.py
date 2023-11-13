import numpy as np
from   src.geometryLIB import optimizer

def main():
    # file name allocation
    fileName = 'coords.dat'
    fileName = 'airfoil1.dat'
    
    with open(file=fileName, mode='r') as f:
        data = np.loadtxt(f)
        data = np.array(data)

    Nsuct  = 5
    Npress = 5
    nMax   = 4
    flip   = False

    # data[:, 1] = - data[:, 1]
    blade, _, _ = optimizer.optimizeBlade(data, Nsuct, Npress, LEradius=2.5E-2, method='Nelder-Mead', flip=flip, nMax=nMax)

    # blade.save('blade1.txt')

if __name__ == '__main__':
    main()