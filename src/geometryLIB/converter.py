#!/usr/bin/env python3
import matplotlib.pyplot       as     plt 
import numpy                   as     np 
from   scipy                   import optimize
from   geometryLIB             import camberline 
from   geometryLIB             import profileLine
from   geometryLIB.profileLine import Side 

def coordBlade(
        metalInlet:  float, 
        metalOutlet: float, 
        LEradius:    float = 3.0E-2, 
        TEradius:    float = 2.5E-2,
        Zw:          float = 0.8,
        plot:        bool  = True
    ) -> np.ndarray:

    # converting angles into radiants
    metalInlet  = - np.deg2rad(np.abs(metalInlet))
    metalOutlet = - np.deg2rad(np.abs(metalOutlet))

    # stagger angle 
    stagger = 0.95 * np.rad2deg(- np.arctan( (np.tan(metalInlet) - np.tan(metalOutlet)) / 2 )) + 5.0
    stagger = - stagger 
    
    # solidity computation
    sigma = 2 / Zw * np.cos(-metalOutlet)**2 * (np.tan(-metalOutlet) - np.tan(metalInlet)) 

    print(">>> stagger angle = ", stagger)
    print(">>> solidity      = ", sigma)

    metalInlet  = np.rad2deg(metalInlet)
    metalOutlet = - np.rad2deg(metalOutlet)

    cLine = camberline.Camberline(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chebyschev=True)

    metalInlet  =   np.abs(np.deg2rad(metalInlet))
    metalOutlet = - np.abs(np.deg2rad(metalOutlet))

    c = 1 / np.sin(np.deg2rad(stagger))
    H = c / np.cos(np.deg2rad(stagger))

    P1 = [LEradius * (1 - np.sin(metalInlet)), H - LEradius * np.cos(metalInlet)]
    P2 = [1 - TEradius * (1 - np.sin(metalOutlet)), - TEradius * np.cos(metalOutlet)]

    if plot:
        ax = plt.subplot()

        cLine.plot(ax=ax, plotInZero=True, pitch=sigma/2, number=2)

        ax.scatter(P1[0], P1[1], c='r')
        ax.scatter(P2[0], P2[1], c='b')

        ax.set_aspect('equal')
        ax.grid(visible=True, linestyle='dotted')

        plt.show()

    return stagger, sigma

def bladeOptimization(
        x:        np.ndarray,
        Nsuct:    int, 
        Npress:   int,
        TEradius: float, 
    ) -> float:

    # setting up blade components
    # camberline
    staggerAngle  = x[0]
    metalInAngle  = x[1]
    metalOutAngle = x[2]

    # setting up profile line properties
    A0         = x[3]
    Asuct      = x[4:5+Nsuct]
    Apress     = x[4+Nsuct:5+Npress+Nsuct]
    wedgeAngle = x[-3]

    # concatenating data 
    Asuct  = np.concatenate([[A0, Asuct]])
    Apress = np.concatenate([[A0, Apress]])

    # setting up leading edge position
    x0 = x[-2]
    y0 = x[-1]

    # camberline generation
    cLine = camberline.Camberline(stagger=staggerAngle, metalIn=metalInAngle, metalOut=metalOutAngle, chebyschev=True)

    # profile generation 
    sLine = profileLine.ProfileLine(N=Asuct.shape[0], A=Asuct, wedgeAngle=wedgeAngle, position=Side.SS, TEradius=TEradius)
    pLine = profileLine.ProfileLine(N=Apress.shape[0], A=Apress, wedgeAngle=wedgeAngle, position=Side.PS, TEradius=TEradius)

def kulfanConverter(
        Nsuct:         int, 
        Npress:        int,
        metalInAngle:  float,
        metalOutAngle: float,    
        staggerAngle:  float = 20,
        TEradius:      float = 2.5E-2 
    ) -> tuple:
    '''
    This function generates the Kulfan parametrized version of a blade represented in coordinates.
    '''

    coordGeometry = coordBlade()

    geometry = optimize.optimize()

def main():
    stagger, sigma = coordBlade(
        metalInlet  = 30, 
        metalOutlet = 72.5, 
        LEradius    = 3.0E-2, 
        TEradius    = 2.5E-2,
        Zw          = 0.8
    )

    # stagger, sigma = coordBlade(
        # metalInlet  = 20, 
        # metalOutlet = 65, 
        # LEradius    = 3.0E-2, 
        # TEradius    = 2.5E-2,
        # Zw          = 0.8
    # )

if __name__ == "__main__":
    main()