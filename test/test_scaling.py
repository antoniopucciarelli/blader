import matplotlib.pyplot       as     plt 
from   geometryLIB             import camberline
from   geometryLIB             import profileLine
from   geometryLIB.profileLine import Side 

def main():
    # variables allocation/definition
    stagger     = 30
    metalInlet  = -40
    metalOutlet = +70
    chord       = 2.0
    chebyschev  = True
    origin      = True
    number      = 2
    pitch       = 3.0
    vector      = False
    normalized  = True

    # profile line 
    A          = [0.3, 0.3, 0.3]
    LEradius   = None#chord * 2.5e-2 / 2
    TEradius   = 2.5e-2 / 2 
    wedgeAngle = None#30

    # object generation
    # camberline 
    cLine = camberline.Camberline(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chord=chord, chebyschev=chebyschev, origin=origin)

    # profile line 
    pLine = profileLine.ProfileLine(A=A, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=Side.PS)
    pLine.compute(camberline=cLine)

    sLine = profileLine.ProfileLine(A=A, LEradius=LEradius, wedgeAngle=wedgeAngle, TEradius=TEradius, position=Side.SS)
    sLine.compute(camberline=cLine)

    A1 = [0.18, 0.25, 0.1, 0.15, 0.1]
    
    pLine1 = profileLine.ProfileLine(A=A1, position=Side.PS, TEradius=TEradius)
    pLine1.compute(camberline=cLine)

    sLine1 = profileLine.ProfileLine(A=A1, position=Side.SS, TEradius=TEradius)
    sLine1.compute(camberline=cLine)

    # axes generation
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plotting data
    cLine.plot(ax=ax, normalized=normalized, number=number, pitch=pitch, vector=vector)
    pLine.plot(ax=ax, color='b', normalized=normalized, pitch=pitch, number=number)
    sLine.plot(ax=ax, color='r', normalized=normalized, pitch=pitch, number=number)

    pLine1.plot(ax=ax, color='orange', normalized=normalized, pitch=pitch, number=number)
    sLine1.plot(ax=ax, color='c', normalized=normalized, pitch=pitch, number=number)

    # showing plot
    ax.set_aspect('equal')
    ax.grid(visible=True, linestyle='dotted')
    plt.show()

if __name__ == '__main__':
    main()