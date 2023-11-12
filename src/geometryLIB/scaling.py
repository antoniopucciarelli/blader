#!/usr/bin/env python3
import numpy      as     np   
from   matplotlib import pyplot as plt 
from   geometryLIB  import camberline
from   geometryLIB  import profileLine 
from   geometryLIB.profileLine import Side

def linearSolver(
        N:          int                     | None = None, 
        camberline: camberline.Camberline   | None = None, 
        profile:    profileLine.ProfileLine | None = None, 
        wedgeAngle: float                   | None = None,
        nPoints:    int                     = 1000
    ) -> np.ndarray:
    r'''
    This function converts the suction side or pressure side parametrization, using the Kulfan parametrization model. 
    It solves a linear system of equations based the previous curve shape (which remains constant) relative the new `N` parameters parametrization.

    Parameters
    ----------
    `camberline`: camberline.camberline 
        profile camberline. It does not change during the fuction. It allows computing the profile coordinates.
    `profile`: profileLine.profileLine 
        suction or pressure side curve. This curve is parametrized with :math:`A_i, i = 0:N_{profile}`.  
    `N`: int 
        this variable sets the number of parameters used to sketch the `profile` keeping the same curve shape.
    `nPoints`: int 
        number of points on which the root mean square error is computed from the computed blade geometry and target blade. This parameter is both for suction side and pressure side.
    
    Returns
    -------
    `A`: np.array
        array that stores the new parametrization of dimension `N + 1`.
    `profileNew`: profileLine.profileLine
        profile curve object with `N` parameters.
    '''

    # increasing DOF 
    if wedgeAngle is not None:
        N = N + 1

    # profile line study
    # evaluation of the initial profile line at N + 1 chord points
    x = np.linspace(0, 1, N + 1)

    # computing and collecting profile line points
    profile.computeSurface(x, camberline)
    y = profile.Y

    # removing camberline from y
    yCamber = camberline.yCamberPoint(x)
    y = y - yCamber 

    # removing TE linear thickness distribution from y
    _, yTEthickness = profile.TEdistribution(x, camberline)
    y = y - yTEthickness

    # solving a linear system of equations at each control point (x, y)
    # matrix generation
    matrix = np.zeros((N + 1, N + 1))

    # matrix assembly
    for ii in range(N + 1):
        # removing A0 thickness distribution from y
        bernsteinValA0 = profile.bernstein(x[ii], 0, N)
        thicknessValA0 = profile.A[0] * bernsteinValA0 * profile.C(x[ii])

        # removing A wedge angle thickness distribution from y
        bernsteinValWedge = profile.bernstein(x[ii], N, N)
        thicknessValWedge = profile.A[-1] * bernsteinValWedge * profile.C(x[ii])
        
        # computing normal and Bernstein value that weights the A0 and wedge angle influence
        _, normal = camberline.computeVec(x[ii])
        B         = profile.bernstein(x[ii], 0, 2)

        if profile.position == Side.PS:
            thickness    = thicknessValA0 + thicknessValWedge
            thicknessVal = thickness * (normal[1] * B + (1 - B))
        elif profile.position == Side.SS:
            thickness    = thicknessValA0 + thicknessValWedge
            thicknessVal = - thickness * (normal[1] * B + (1 - B))
        else:
            raise TypeError('Error on the profile line position. It has to be a kulfanLIB.profileline.Side object.')

        # removing A0 and wedge angle influence from y
        y[ii] = y[ii] - thicknessVal 

        # matrix element allocation
        for jj in range(N + 1):
            # computing Bernstein value for the thickness distribution
            bernsteinVal = profile.bernstein(x[ii], jj, N)
            # computing thickness with unit value
            thicknessVal = bernsteinVal * profile.C(x[ii])
            # computing normal 
            _, normal = camberline.computeVec(x[ii])
            # computing Bernstein distribution for normal fraction distribution
            B = profile.bernstein(x[ii], 0, 2)

            # setting up thickness contribute with respect to profile position (SS or PS) 
            if profile.position == Side.PS:
                matrix[ii, jj] = thicknessVal * (normal[1] * B + (1 - B))
            elif profile.position == Side.SS:
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
    A0 = profile.A[0]
    A  = np.insert(A, 0, [A0])

    # trailing edge properties 
    Awedge = profile.A[-1] 
    A      = np.concatenate((A, [Awedge]))

    # generating new suction line object 
    profileNew = profileLine.ProfileLine(N=N, A=A, wedgeAngle=None, position=profile.position, TEradius=profile.TEradius)
    
    # computing rms between the 2 curves
    profile.computeSurface(x = np.linspace(0, 1, nPoints), camberline = camberline)
    profileNew.computeSurface(x = np.linspace(0, 1, nPoints), camberline = camberline)
    
    # rms computation
    rmsX = np.sqrt(1/nPoints * sum((profile.X - profileNew.X)**2)) 
    rmsY = np.sqrt(1/nPoints * sum((profile.Y - profileNew.Y)**2))

    if profile.position == Side.PS:
        if wedgeAngle is not None:
            print('>>> SCALING SETUP\n>>> PREV.     N = {0:d}\n>>> NEW       N = {1:d}\n>>> WEDGE ANGLE = {2:.2f} deg\n>>> RMS ON THE PRESSURE SIDE\n    >>> RMS(X) = {3:.5E}\n    >>> RMS(Y) = {4:.5E}'.format(profile.N, profileNew.N, wedgeAngle, rmsX, rmsY))
        else:
            print('>>> SCALING SETUP\n>>> PREV.     N = {0:d}\n>>> NEW       N = {1:d}\n>>> WEDGE ANGLE = {2} deg\n>>> RMS ON THE PRESSURE SIDE\n    >>> RMS(X) = {3:.5E}\n    >>> RMS(Y) = {4:.5E}'.format(profile.N, profileNew.N, wedgeAngle, rmsX, rmsY))
        print('>>> Apress = {0}'.format(A))
        print(80 * '~')
    elif profile.position == Side.SS:
        if wedgeAngle is not None:
            print('>>> SCALING SETUP\n>>> PREV.     N = {0:d}\n>>> NEW       N = {1:d}\n>>> WEDGE ANGLE = {2:.2f} deg\n>>> RMS ON THE SUCTION SIDE\n    >>> RMS(X) = {3:.5E}\n    >>> RMS(Y) = {4:.5E}'.format(profile.N, profileNew.N, wedgeAngle, rmsX, rmsY))
        else:
            print('>>> SCALING SETUP\n>>> PREV.     N = {0:d}\n>>> NEW       N = {1:d}\n>>> WEDGE ANGLE = {2} deg\n>>> RMS ON THE SUCTION SIDE\n    >>> RMS(X) = {3:.5E}\n    >>> RMS(Y) = {4:.5E}'.format(profile.N, profileNew.N, wedgeAngle, rmsX, rmsY))
        print('>>> Asuct = {0}'.format(A))
        print(80 * '~')
    else: 
        raise TypeError('Error on the profile line position. It has to be a kulfanLIB.profileline.Side object.')

    if rmsX > 1e-10 or rmsY > 1e-10:
        if profile.N > profileNew.N:
            return A, profileNew
        else:
            raise ValueError('Root mean square error is too high')
    else:
        return A, profileNew

def scale(
        camberline:   camberline.Camberline   | None = None, 
        suctionLine:  profileLine.ProfileLine | None = None, 
        pressureLine: profileLine.ProfileLine | None = None, 
        Nsuct:        int      | None = None, 
        Npress:       int      | None = None, 
        nPoints:      int      = 1000,
        wedgeAngle:   float    | None = None, 
        ax:           plt.Axes | None = None
    ) -> tuple[list[float], list[float]]:
    r'''
    This function converts the suction side and pressure side parametrization, using the Kulfan parametrization model, into the same suction side and pressure side surface but parametrized with more parameters.
    `A0` is the same for `suctionLine` and `pressureLine` and keeps remaining the same also at the end of change of parameters. This because it is strictly related to the leading edge curvature which does not change during the transformation.

    Parameters
    ----------
    `camberline`: camberline.camberline 
        profile camberline. It does not change during the fuction. It allows computing the profile coordinates.
    `suctionLine`: profileLine.profileLine 
        suction curve. This curve is parametrized with :math:`A_i, i = 0:N_{suctionLine}`. 
    `pressureLine`: profileLine.profileLine 
        pressure curve. This curve is parametrized with :math:`A_i, i = 0:N_{pressureLine}`. 
    `Nsuct`: int 
        this variable sets the number of parameters used to sketch the `suctionLine` keeping the same curve shape.
    `Npress`: int 
        this variable sets the number of parameters used to sketch the `pressureLine` keeping the same curve shape.
    `nPoints`: int 
        number of points on which the root mean square error is computed from the computed blade geometry and target blade. This parameter is both for suction side and pressure side.
    `wedgeAngle`: float 
        trailing edge wedge angle.
    `ax`: plt.Axes
        axes where the target and computed curves are plotted. `None` if the plot is not needed.

    Returns
    -------
    `Asuct`: np.array
        array of dimension `Nsuct + 1` that parametrizes `suctionLine`.
    `Apress`: np.array 
        array of dimension `Npress + 1` that parametrizes `pressureLine`.
    '''

    print('\n>>> \033[41mBLADE SCALING\033[0m\n')

    if Nsuct is not None and suctionLine is not None:
        Asuct, suctionLineNew = linearSolver(Nsuct, camberline, suctionLine, wedgeAngle, nPoints)
        # plotting data 
        if ax is not None:
            camberline.plot(ax)
            suctionLine.plot(ax, color='red')
            suctionLineNew.plot(ax, color='orange', linestyle='--')
    else:
        print('>>> NO Nsuct SELECTED -> SUCTION SIDE CURVE IS NOT CHANGED IN PARAMETRIZATION')
        Asuct = None

    if Npress is not None and pressureLine is not None:
        Apress, pressureLineNew = linearSolver(Npress, camberline, pressureLine, wedgeAngle, nPoints)
        # plotting data 
        if ax is not None:
            camberline.plot(ax)
            pressureLine.plot(ax, color='blue')
            pressureLineNew.plot(ax, color='cyan', linestyle='--')
    else:
        print('>>> NO Npress SELECTED -> SUCTION SIDE CURVE IS NOT CHANGED IN PARAMETRIZATION')
        Apress = None

    # setting ax properties 
    if ax is not None:
        colTab = [r'$N_{suct}$', r'$N_{press}$']  
        cellTab = [[suctionLine.N, pressureLine.N], [suctionLineNew.N, pressureLineNew.N]]
        rowTab = ["PREVIOUS", "NEW"]
        ax.table(cellText=cellTab, colLabels=colTab, rowLabels=rowTab, colWidths=[0.5/3, 0.5/3], bbox=[0.35, 1.03, 0.55, 0.18], cellLoc='center')
        ax.set_aspect(aspect='equal')
        ax.grid(visible=True, linestyle='--')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()

    return Asuct, Apress

def convertBlade(
        X:         np.ndarray, 
        Nsuct:     int, 
        Npress:    int,
        NsuctNew:  int,
        NpressNew: int,
        TEradius:  float = 2.5E-2,
        plot:      bool  = True
    ) -> tuple[np.ndarray, tuple[list[float], list[float]]]:
    '''
    This function converts a blade geometry, using Kulfan parametrization, which wedge angle is not set as parameter to a new geometry where the wedge angle is set as paramter.

    Parameters
    ----------
    `X`: np.array
        array that stores the blade geometry in raw format without explicit wedge angle.
    `Nsuct`: int 
        integer that defines the DOFs for the suction side of the blade.
    `Npress: int
        integer that defines the DOFs for the pressure side of the blade.
    `NsuctNew`: int 
        integer that defines the new blade DOFs for the suction side.
    `NpressNew`: int 
        integer that defines the new blade DOFs for the pressure side. 
    `TEradius`: float 
        float value that defines the trailing edge radius of the blade.
    `plot`: bool
        boolean value for the plotting of the blade.
    '''

    # getting camberline properties
    staggerAngle  = X[0]
    metalInAngle  = X[1]
    metalOutAngle = X[2] 

    # getting leading edge radius
    LEradius = X[3]**2 / 2
    A0       = X[3]

    # getting pitch value
    pitch = X[-1]

    # wedge angle computation for using the suction side and the pressure side of the blade
    # the wedge angle is treated as an average of the 2 sides
    SSwedgeVal = X[3 + Nsuct]
    PSwedgeVal = X[3 + Nsuct + Npress]

    # converting data into angles 
    SSwedgeAngle = np.rad2deg(np.arctan(SSwedgeVal))
    PSwedgeAngle = np.rad2deg(np.arctan(PSwedgeVal))

    # averaging wedge angles 
    wedgeAngle = (SSwedgeAngle + PSwedgeAngle) / 2

    # getting original blade parameters
    Asuct0  = X[4:4+Nsuct]
    Apress0 = X[4+Nsuct:4+Nsuct+Npress]

    # getting new blade parameters
    Asuct  = X[4:3 + Nsuct]
    Apress = X[4 + Nsuct:3 + Nsuct + Npress]

    # setting up new X array
    camberlineArray = np.array([staggerAngle, metalInAngle, metalOutAngle])
    Aarray          = np.concatenate(([A0], Asuct, Apress)) 
    Xnew            = np.concatenate((camberlineArray, Aarray, [wedgeAngle, pitch]))

    # printing data 
    print('>>> EXTRACTED DATA')
    print('>>> INITIAL ARRAY = {0}'.format(X))
    print('>>> UPDATED ARRAY = {0}'.format(Xnew))
    # camberline properties
    print('>>> CAMBERLINE PROPERTIES')
    print('\tSTAGGER      ANGLE = {0:.3E}'.format(staggerAngle))
    print('\tMETAL INLET  ANGLE = {0:.3E}'.format(metalInAngle))
    print('\tMETAL OUTLET ANGLE = {0:.3E}'.format(metalOutAngle))
    # leading edge properties
    print('>>> LE RADIUS = {0:.3E} || {1:.3E}'.format(LEradius, A0))
    # pitch properties 
    print('>>> PITCH = {0:.3E}'.format(pitch))
    # wedge angle properties
    print('>>> WEDGE ANGLE PROPERTIES')
    print('\tSUCTION  SIDE WEDGE ANGLE = {0:.3E} || {1:.3E}'.format(SSwedgeAngle, SSwedgeVal))
    print('\tPRESSURE SIDE WEDGE ANGLE = {0:.3E} || {1:.3E}'.format(PSwedgeAngle, PSwedgeVal))
    print('\tUPDATED       WEDGE ANGLE = {0:.3E}'.format(wedgeAngle))
    # suction side properties 
    print('>>> SUCTION SIDE PROPERTIES')
    for ii in range(Nsuct-1):
        print('\tA{0:02d} = {1:.3E}'.format(ii+1, Asuct[ii]))
    # pressure side properties 
    print('>>> PRESSURE SIDE PROPERTIES')
    for ii in range(Npress-1):
        print('\tA{0:02d} = {1:.3E}'.format(ii+1, Apress[ii]))
    print(35 * '~')

    # plotting blade 
    if plot:
        # allocating original blade geometry
        # camberline generation
        cLine0 = camberline.Camberline(staggerAngle, metalInAngle, metalOutAngle, nPoints=300)
        # suction side generation 
        sLine0 = profileLine.ProfileLine(N=Nsuct, A=np.concatenate(([A0], Asuct0)), wedgeAngle=None, position=Side.SS, TEradius=TEradius)
        # pressure side generation
        pLine0 = profileLine.ProfileLine(N=Npress, A=np.concatenate(([A0], Apress0)), wedgeAngle=None, position=Side.PS, TEradius=TEradius)
        
        # computing original blade geometry
        sLine0.computeSurface(x=cLine0.x, camberline=cLine0)
        pLine0.computeSurface(x=cLine0.x, camberline=cLine0)

        # allocating new blade geometry 
        # camberline generation 
        cLine = camberline.Camberline(staggerAngle, metalInAngle, metalOutAngle, nPoints=300)
        # suction side generation 
        sLine = profileLine.ProfileLine(N=Nsuct, A=np.concatenate(([A0], Asuct)), wedgeAngle=wedgeAngle, position=Side.SS, TEradius=TEradius)
        # pressure side generation
        pLine = profileLine.ProfileLine(N=Npress, A=np.concatenate(([A0], Apress)), wedgeAngle=wedgeAngle, position=Side.PS, TEradius=TEradius)
        
        # computing geometry 
        sLine.computeSurface(x=cLine.x, camberline=cLine)
        pLine.computeSurface(x=cLine.x, camberline=cLine)

        # plotting blade
        fig = plt.figure()
        ax = fig.subplots(1,2)
        fig.suptitle('Swapping to wedge angle representation and blade scaling')

        # plotting original blade 
        cLine0.plot(ax=ax[0], color='black', marker=' ')
        sLine0.plot(ax=ax[0], color='red', linestyle='-')
        pLine0.plot(ax=ax[0], color='blue', linestyle='-')

        # plotting new blade 
        cLine.plot(ax=ax[0], color='black', marker='s')
        sLine.plot(ax=ax[0], color='orange', linestyle='--')
        pLine.plot(ax=ax[0], color='cyan', linestyle='--')

        # axes properties 
        titleMessage = 'Blade changes on trailing edge angle ' + r'$\beta$' + '\n' + r'$N_{{suct}} = {{{0:d}}}$'.format(Nsuct) + '\t' + r'$N_{{press}} = {{{0:d}}}$'.format(Npress)
        ax[0].set_title(titleMessage)
        ax[0].set_aspect('equal')
        ax[0].grid(visible=True, linestyle='--')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].legend(bbox_to_anchor=(1,1), loc='upper left')

        # scaling blade 
        XnewScaled = scale(
            camberline   = cLine, 
            suctionLine  = sLine, 
            pressureLine = pLine, 
            Nsuct        = NsuctNew, 
            Npress       = NpressNew, 
            nPoints      = 1000,
            wedgeAngle   = wedgeAngle, 
            ax           = ax[1]
        ) 

        plt.tight_layout()
        plt.show()

    else: 
        # scaling airfoil
        XnewScaled = scale(
            camberline   = cLine, 
            suctionLine  = sLine, 
            pressureLine = pLine, 
            Nsuct        = NsuctNew, 
            Npress       = NpressNew, 
            nPoints      = 1000,
            wedgeAngle   = wedgeAngle, 
            ax           = None
        ) 

    return Xnew, XnewScaled