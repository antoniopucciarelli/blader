import numpy as np
from   src.geometryLIB import optimizer

def main():
    
    with open(file='airfoil1.dat', mode='r') as f:
        data = np.loadtxt(f)

    Nsuct  = 4
    Npress = 4
    nMax   = 4
    flip   = False

    blade, _, _ = optimizer.optimizer(data, Nsuct, Npress, LEradius=2.5E-2, method='Nelder-Mead', flip=flip, nMax=nMax)

    # blade.save('blade1.txt')

if __name__ == '__main__':
    main()